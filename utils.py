import os 
import json
import time
import queue
import logging
from typing import Dict, Union, Any, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
# from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import (
    BertModel,
    BertForSequenceClassification,
    GPT2Model,
    GPT2ForSequenceClassification,
    XLNetModel,
    XLNetForSequenceClassification,
    BartModel,
    BartForSequenceClassification,
)
from transformers.models.bert.modeling_bert import BertModel


logger = logging.getLogger(__name__)

   
COLOR_MAP = {
    'embedding': 'blue',
    'attention': 'purple',
    'ffn': 'brown',
    'dropout': 'grey',
    'backward': 'green',
}
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class Node:
    def __init__(
        self, 
        node_id: int, 
        num_gpus_per_node: int, 
        init_device: Optional[int] = None,
    ):
        self.node_id = node_id
        self.num_gpus_per_node = num_gpus_per_node
        # self.device_queues = [queue.Queue() for _ in range(num_gpus_per_node)]
        self.device_queues = [queue.PriorityQueue() for _ in range(num_gpus_per_node)]
        self.init_device = init_device if init_device is not None else 0
        self.last_device = init_device + num_gpus_per_node - 1
        self.device_ids = list(range(init_device, init_device + num_gpus_per_node))
        # Stats for recording
        self.task_allocation = []
        self.num_tasks = 0
        self.train_tasks = 0
        self.test_tasks = 0
        self.mean = 0
        self.std = 0
        self.M2 = 0  # Sum of squares of differences from the current mean
        
    def updata_task_stats(self, taskID: int, loss: float, length: int, computation_type: str):
        self.task_allocation.append({
            'taskID': taskID, 
            'type': computation_type, 
            'loss': loss,
            'length': length,
        })

    def update_length_stats(self, length: int, computation_type: str):
        self.num_tasks += 1
        if computation_type == 'training':
            self.train_tasks += 1
        elif computation_type == 'test':
            self.test_tasks += 1
        else:
            raise ValueError(f"Invalid computation type: {computation_type}")
        delta = length - self.mean
        self.mean += delta / self.num_tasks
        delta2 = length - self.mean
        self.M2 += delta * delta2
        self.std = np.sqrt(self.M2 / self.num_tasks) if self.num_tasks > 1 else 0
    
    def length_consistency(self, length: int):
        if self.num_tasks == 0 or self.std == 0:
            return 0
        return (1 / (self.std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((length - self.mean) / self.std) ** 2)
        
        
class Task:
    def __init__(
        self, 
        task_id: int, 
        query: Union[torch.Tensor, Dict[str, Any]], 
        rate_lambda: float,
        feedback: Optional[torch.Tensor] = None, 
        node_id: Optional[int] = None, 
        num_gpus_per_node: Optional[int] = None,
        require_training: Optional[bool] = None,
    ):
        self.task_id = task_id
        self.query = query
        self.rate_lambda = rate_lambda
        num_gpus_per_node = num_gpus_per_node if num_gpus_per_node is not None else 1
        self.hiddens = [query] + [None for _ in range(num_gpus_per_node - 1)]
        self.feedback = feedback
        self.node_id = node_id if node_id is not None else 0
        self.require_training = False if require_training is None else require_training
        # Define do_backward for selective training: initially set to require_training
        self.do_backward = False if require_training is None else require_training


def record_time(
    device: int, 
    event_type: str, 
    opt_type: str, 
    taskID: int,
    timing_info: Dict[str, List[float]], 
    verbose: bool = False,
) -> float:
    # event_type can be 'start' or 'end'
    timestamp = time.time()
    timing_info[f"{device}_{event_type}"].append((timestamp, opt_type, taskID))
    if verbose:
        print(f"\t[CUDA {device}] Task {event_type} at time {timestamp}")
    return timestamp
    

def get_total_params(module: torch.nn.Module):
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    return total_params


def get_transformer_layers(
    model: Union[
        BertForSequenceClassification, 
        GPT2ForSequenceClassification, 
        XLNetForSequenceClassification, 
        BartForSequenceClassification,
    ]
) -> List[Union[BertModel, GPT2Model, XLNetModel, BartModel]]:
    model_name = model.__class__.__name__.lower()
    layers = []
    if "bert" in model_name:
        layers = model.bert.encoder.layer
    elif "gpt2" in model_name:
        layers = model.transformer.h
    elif "xlnet" in model_name:
        layers = model.transformer.layer
    elif "bart" in model_name:
        encoder_layers = model.model.encoder.layers
        decoder_layers = model.model.decoder.layers
        layers = encoder_layers + decoder_layers
    return layers


def get_colors(index: List[str], color_map: dict = COLOR_MAP):
    return [
        color_map['embedding'] if 'embedding' in idx.lower() 
        else color_map['attention'] if 'attention' in idx.lower() or 'attn' in idx.lower()
        else color_map['layernorm'] if 'ln' in idx.lower() or 'layernorm' in idx.lower()
        else color_map['ffn'] if (
            'mlp' in idx.lower() or 
            'linear' in idx.lower() or 
            'pooler' in idx.lower() or 
            'intermediate' in idx.lower() or
            'output' in idx.lower()
        )
        else color_map['dropout'] if 'dropout' in idx.lower()
        else color_map['backward'] if 'backward' in idx.lower() or 'bp' in idx.lower()
        else 'red'  # default color 
        for idx in index]

 

# Plot the average latency distribution of each layer
def plot_layer_profiling(
    profile_res: pd.DataFrame, 
    model_name: str, 
    backward_res: pd.DataFrame = None,
    save_file: str = None,
    color_map: dict = COLOR_MAP,
    metric: str = 'inference latency',
    unit: str = 'seconds',
    figsize: Tuple[int, int] = (20, 6),
):
    # Assuming you have the DataFrame loaded as df (do not include the batch_size, input_length columns)
    if 'batch_size' in profile_res.columns and 'input_length' in profile_res.columns:
        res = profile_res.drop(columns=['batch_size', 'input_length'])
    else:
        res = profile_res
    averages = res.mean()
    
    # Determine the color of each bar based on its label
    colors = get_colors(averages.index)
    
    # Create custom patches for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) for key in color_map]

    # Plotting
    plt.figure(figsize=figsize)
    averages.plot(kind='bar', color=colors, width=0.5)
    
    # Also plot line graph
    plt.plot(averages, color='black', linestyle='-', linewidth=2)
    
    plt.ylabel(f'Average {metric} ({unit})', fontdict={'fontsize': 12})
    plt.xlabel('Layer', fontdict={'fontsize': 12})
    plt.title(f'Average {metric} per Layer for {model_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Add legend for the 6 layers
    plt.legend(handles=legend_elements, title="Layer type")
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show() 
    
    

def plot_layer_profiling_dist(
    profile_res: pd.DataFrame, 
    model_name: str, 
    save_file: str = None,
    color_map: dict = COLOR_MAP,
    metric: str = 'inference latency',
    unit: str = 'seconds',
    figsize: Tuple[int, int] = (20, 6),
):
    
    # Assuming you have the DataFrame loaded as df (do not include the batch_size, input_length columns)
    # If res has columns batch_size and input_length, drop them
    if 'batch_size' in profile_res.columns and 'input_length' in profile_res.columns:
        res = profile_res.drop(columns=['batch_size', 'input_length'])
    else:
        res = profile_res
    
    # Determine the color of each column based on its label
    column_colors = get_colors(res.columns)

    # Create custom patches for legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[key], label=key) for key in color_map]
    
    # Plotting
    plt.figure(figsize=figsize)
    
    # Boxplot
    boxprops = dict(linestyle='-', linewidth=1)
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    # res.boxplot(column=res.columns, vert=False, patch_artist=True, boxprops=boxprops, medianprops=medianprops)
    bp = res.boxplot(
        vert=True, 
        patch_artist=True, 
        boxprops=boxprops, 
        medianprops=medianprops, 
        showfliers=False, 
        return_type='dict',
    )
    
    # Coloring the boxes based on the determined colors
    for patch, color in zip(bp['boxes'], column_colors):
        patch.set_facecolor(color)
    
    plt.xlabel('Layer', fontdict={'fontsize': 12})
    plt.ylabel(f'{metric} ({unit})', fontdict={'fontsize': 12})
    plt.title(f'Distribution of {metric} per Layer for {model_name}')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Add legend for the layer types with a title
    plt.legend(handles=legend_elements, title="Layer type")
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, bbox_inches='tight')
    plt.show()
    
    


LABEL2METHOD = {
    "NaiveMix": "active",
    "Separate": "isolated",
    "LaMix-LLF": "interval",
    "LaMix-MLF": "interval-MLF",
}

def plot_dual(lambda_=50, label1='NaiveMix', label2=None, label3=None, label4=None, label5=None,
              figname=None, setting=None, load_balancing=None, num_nodes=2, legend=True, model='dialogpt-small', use_bubble=True,
              color1=sns.color_palette("deep")[0], 
              color2=sns.color_palette("deep")[1],):
    
    res1, res2, res3, res4, res5 = [], [], [], [], []
    model_schedule = f'{model}_{load_balancing}' if load_balancing is not None else model
    res_dir = f"prof/{num_nodes}_node/lambda_{lambda_}"
    for retrain_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if setting:
            metric = json.load(open(f"{res_dir}/dialogpt-small/metrics_dialogpt-small_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            metric = json.load(open(f"{res_dir}/dialogpt-medium/metrics_dialogpt-medium_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res2.append(metric)
        else:
            metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            if label2:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label2]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res2.append(metric)
            if label3:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label3]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res3.append(metric)
            if label4:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label4]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res4.append(metric)
            if model_schedule != model:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model_schedule}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res5.append(metric)
        
    res1 = pd.DataFrame(res1)
    res2 = pd.DataFrame(res2) if res2 else None
    res3 = pd.DataFrame(res3) if res3 else None
    res4 = pd.DataFrame(res4) if res4 else None
    res5 = pd.DataFrame(res5) if res5 else None

    # Let's plot the metrics, x-axis is retrain_rate, y-axis is the metric value
    os.makedirs("figure", exist_ok=True) 
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    
    ax2 = axes[0].twinx()
    # ax.yaxis.grid(True, linestyle='dotted', which='major', color='grey', alpha=0.5)
    line1, = axes[0].plot(res1["retrain_rate"], res1["loss"], label=label1, marker='v', color=color1)
    line2, = ax2.plot(res1["retrain_rate"], res1["response_time"] * 1000, label=label1, color=color2, marker='x')
    lines = [line1, line2]
    if res2 is not None:
        line1_1, = axes[0].plot(res2["retrain_rate"], res2["loss"], label=label2, marker='v', color=color1, linestyle='--')
        line2_2, = ax2.plot(res2["retrain_rate"], res2["response_time"] * 1000, label=label2, color=color2, marker='x', linestyle='--')
        lines += [line1_1, line2_2]
    if res3 is not None:
        line1_2, = axes[0].plot(res3["retrain_rate"], res3["loss"], label=label3, marker='v', color=color1, linestyle='dotted')
        line2_3, = ax2.plot(res3["retrain_rate"], res3["response_time"] * 1000, label=label3, marker='x', color=color2, linestyle='dotted')
        lines += [line1_2, line2_3]
    if res4 is not None:
        line1_3, = axes[0].plot(res4["retrain_rate"], res4["loss"], label=label4, marker='v', color=color1, linestyle='-.')
        line2_4, = ax2.plot(res4["retrain_rate"], res4["response_time"] * 1000, label=label4, marker='x', color=color2, linestyle='-.')
        lines += [line1_3, line2_4]
    if res5 is not None:
        line1_4, = axes[0].plot(res5["retrain_rate"], res5["loss"], label=label5, marker='v', color=color1, linestyle=(0, (1, 10)))
        line2_5, = ax2.plot(res5["retrain_rate"], res5["response_time"] * 1000, label=label5, marker='x', color=color2, linestyle=(0, (1, 10)))
        lines += [line1_4, line2_5]
    
    ax2.set_ylabel("Response time (ms)", fontsize=14)
    ax2.tick_params(axis='y', colors=color2)
    labels = [line.get_label() for line in lines]
    axes[0].set_ylabel("Eval loss", fontsize=14)
    axes[0].tick_params(axis='y', colors=color1)
    axes[0].set_xlabel("Retraining rate", fontsize=14)

    ax2 = axes[1].twinx()
    if use_bubble:
        line1, = axes[1].plot(res1["retrain_rate"], res1["bubble_rate"] * 100, label=label1, color=color1, marker='v')
    else:
        line1, = axes[1].plot(res1["retrain_rate"], res1["idleness"] * 1000, label=label1, color=color1, marker='v')
    line2, = ax2.plot(res1["retrain_rate"], res1["end2end_latency"], label=label1, color=color2, marker='x')
    lines = [line1, line2]
    if res2 is not None:
        if use_bubble:
            line1_1, = axes[1].plot(res2["retrain_rate"], res2["bubble_rate"] * 100, label=label2, color=color1, marker='v', linestyle='--')
        else:
            line1_1, = axes[1].plot(res2["retrain_rate"], res2["idleness"] * 1000, label=label2, color=color1, marker='v', linestyle='--')
        line2_2, = ax2.plot(res2["retrain_rate"], res2["end2end_latency"], label=label2, color=color2, marker='x', linestyle='--')
        lines += [line1_1, line2_2]
    if res3 is not None:
        if use_bubble:
            line1_2, = axes[1].plot(res3["retrain_rate"], res3["bubble_rate"] * 100, label=label3, color=color1, marker='v', linestyle='dotted')
        else:
            line1_2, = axes[1].plot(res3["retrain_rate"], res3["idleness"] * 1000, label=label3, color=color1, marker='v', linestyle='dotted')
        line2_3, = ax2.plot(res3["retrain_rate"], res3["end2end_latency"], label=label3, color=color2, marker='v', linestyle='dotted')
        lines += [line1_2, line2_3]
    if res4 is not None:
        if use_bubble:
            line1_3, = axes[1].plot(res4["retrain_rate"], res4["bubble_rate"] * 100, label=label4, color=color1, marker='v', linestyle='-.')
        else:
            line1_3, = axes[1].plot(res4["retrain_rate"], res4["idleness"] * 1000, label=label4, color=color1, marker='v', linestyle='-.')
        line2_4, = ax2.plot(res4["retrain_rate"], res4["end2end_latency"], label=label4, color=color2, marker='x', linestyle='-.')
        lines += [line1_3, line2_4]
    if res5 is not None:
        if use_bubble:
            line1_4, = axes[1].plot(res5["retrain_rate"], res5["bubble_rate"] * 100, label=label5, color=color1, marker='v', linestyle=(0, (1, 10)))
        else:
            line1_4, = axes[1].plot(res5["retrain_rate"], res5["idleness"] * 1000, label=label5, color=color1, marker='v', linestyle=(0, (1, 10)))
        line2_5, = ax2.plot(res5["retrain_rate"], res5["end2end_latency"], label=label5, color=color2, marker='x', linestyle=(0, (1, 10)))
        lines += [line1_4, line2_5]
        
    ax2.set_ylabel("End2end latency (s)", fontsize=14)
    ax2.tick_params(axis='y', colors=color2)
    labels = [line.get_label() for line in lines]
    # Create a single legend for both lines together
    if use_bubble:
        axes[1].set_ylabel("Bubble rate (%)", fontsize=14)
    else:
        axes[1].set_ylabel("GPU idles (ms)", fontsize=14)
    axes[1].tick_params(axis='y', colors=color1)
    axes[1].set_xlabel("Retraining rate", fontsize=14)
    if legend:
        ncol = 1
        if res2 is not None: ncol += 1
        if res3 is not None: ncol += 1
        if res4 is not None: ncol += 1
        if res5 is not None: ncol += 1
        fig.legend(lines, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.15), fontsize=11)
    plt.tight_layout()
    if figname:
        plt.savefig(f"figure/{figname}.pdf", bbox_inches='tight')
    else:
        plt.savefig("figure/dialogpt_retraining_lambda={lambda_}.pdf", bbox_inches='tight')
    plt.show()
       


def plot_single(lambda_=50, label1='NaiveMix', label2=None, label3=None, label4=None, label5=None,
                figname=None, setting=None, load_balancing=None, num_nodes=2, legend=True, model='dialogpt-small',
                color1=sns.color_palette("deep")[0], 
                color2=sns.color_palette("deep")[1],):
    
    res1, res2, res3, res4, res5 = [], [], [], [], []
    model_schedule = f'{model}_{load_balancing}' if load_balancing is not None else model
    res_dir = f"prof/{num_nodes}_node/lambda_{lambda_}"
    for retrain_rate in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        if setting:
            metric = json.load(open(f"{res_dir}/dialogpt-small/metrics_dialogpt-small_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            metric = json.load(open(f"{res_dir}/dialogpt-medium/metrics_dialogpt-medium_{setting}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res2.append(metric)
        else:
            metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
            metric["retrain_rate"] = retrain_rate
            res1.append(metric)
            if label2:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label2]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res2.append(metric)
            if label3:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label3]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res3.append(metric)
            if label4:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model}_{LABEL2METHOD[label4]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res4.append(metric)
            if model_schedule != model:
                metric = json.load(open(f"{res_dir}/{model}/metrics_{model_schedule}_{LABEL2METHOD[label1]}_poisson_{retrain_rate}.json"))
                metric["retrain_rate"] = retrain_rate
                res5.append(metric)
        
    res1 = pd.DataFrame(res1)
    res2 = pd.DataFrame(res2) if res2 else None
    res3 = pd.DataFrame(res3) if res3 else None
    res4 = pd.DataFrame(res4) if res4 else None
    res5 = pd.DataFrame(res5) if res5 else None

    # Let's plot the metrics, x-axis is retrain_rate, y-axis is the metric value
    os.makedirs("figure", exist_ok=True) 
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.5))
    # ax.yaxis.grid(True, linestyle='dotted', which='major', color='grey', alpha=0.5)
    ax.set_ylabel("Eval loss", fontsize=14)
    ax.tick_params(axis='y', colors=color1)
    ax2 = ax.twinx()
    line1, = ax.plot(res1["retrain_rate"], res1["loss"], label=label1, marker='v', color=color1)
    line2, = ax2.plot(res1["retrain_rate"], res1["end2end_latency"], label=label1, color=color2, marker='x')
    lines = [line1, line2]
    if res2 is not None:
        line1_1, = ax.plot(res2["retrain_rate"], res2["loss"], label=label2, marker='v', color=color1, linestyle='--')
        line2_2, = ax2.plot(res2["retrain_rate"], res2["end2end_latency"], label=label2, color=color2, marker='x', linestyle='--')
        lines += [line1_1, line2_2]
    if res3 is not None:
        line1_2, = ax.plot(res3["retrain_rate"], res3["loss"], label=label3, marker='v', color=color1, linestyle='dotted')
        line2_3, = ax2.plot(res3["retrain_rate"], res3["end2end_latency"], label=label3, color=color2, marker='x', linestyle='dotted')
        lines += [line1_2, line2_3]
    if res4 is not None:
        line1_3, = ax.plot(res4["retrain_rate"], res4["loss"], label=label4, marker='v', color=color1, linestyle='-.')
        line2_4, = ax2.plot(res4["retrain_rate"], res4["end2end_latency"], label=label4, color=color2, marker='x', linestyle='-.')
        lines += [line1_3, line2_4]
    if res5 is not None:
        line1_4, = ax.plot(res5["retrain_rate"], res5["loss"], label=label5, color=color1, marker='v', linestyle=(0, (1, 10)))
        line2_5, = ax2.plot(res5["retrain_rate"], res5["end2end_latency"], label=label5, color=color1, marker='x', linestyle=(0, (1, 10)))
        lines += [line1_4, line2_5]
        
    ax2.set_ylabel("End2end latency (s)", fontsize=14)
    ax2.tick_params(axis='y', colors=color2)
    labels = [line.get_label() for line in lines]
    ax.set_xlabel("Retraining rate", fontsize=14)
    if legend:
        ncol = 1
        if res2 is not None: ncol += 1
        if res3 is not None: ncol += 1
        if res4 is not None: ncol += 1
        if res5 is not None: ncol += 1
        fig.legend(lines, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.15), fontsize=11)
    plt.tight_layout()
    if figname:
        plt.savefig(f"figure/{figname}.pdf", bbox_inches='tight')
    else:
        plt.savefig("figure/single_dialogpt_retraining_lambda={lambda_}.pdf", bbox_inches='tight')
    plt.show()