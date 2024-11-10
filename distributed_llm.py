import os
import sys
sys.dont_write_bytecode = True
import pdb
import time
import json
import queue
import random
import argparse
import logging
import numpy as np
import gc
from tqdm import tqdm
import GPUtil
from GPUtil import GPU
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import defaultdict
from memory_profiler import profile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    set_seed,
    get_scheduler,
)
from utils import Node, Task, record_time
from models import (
    get_stages, 
    _prepare_inputs,
)


# torch.autograd.set_detect_anomaly(True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def run_experiment(args: argparse.Namespace, experimentID: int):
    print(f"\n ** Experiment {experimentID+1} **\n")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize and run the distributed model
    distributed_llm = DistributedLLM(args, experimentID=experimentID)
    distributed_llm.run()
    record_mode = distributed_llm.RECORD_MODE
    
    # Clean up resources explicitly
    del distributed_llm
    torch.cuda.empty_cache()
    gc.collect()

    # Rerun if necessary based on specific conditions
    if record_mode and args.run_mode == 'online':  # Assuming record_mode is a valid arg
        new_run = DistributedLLM(args, experimentID=experimentID)
        new_run.run()
        
        # Final clean up
        del new_run
        torch.cuda.empty_cache()
        gc.collect()
        
 

class DistributedLLM:

    def __init__(self, args: argparse.Namespace, experimentID: int = 0):
        self.n_samples = args.n_samples
        self.setting = args.setting
        self.priority = args.priority
        self.num_nodes = args.num_nodes
        self.num_gpus = args.num_gpus
        self.batch_size = args.batch_size
        self.rate_lambda = args.rate_lambda
        self.training_lambda = args.training_lambda
        self.output_dir = args.output_dir
        self.task_assignment = args.task_assignment
        self.dataset_name_or_path = args.dataset_name_or_path
        self.lr = args.lr
        self.workload = args.workload
        self.retraining_rate = args.retraining_rate  
        self.model_n = args.model_name
        self.save_length = args.save_length
        self.length_distribution = args.length_distribution
        self.length_heterogeneity = args.length_heterogeneity
        self.active_selection = args.active_selection
        self.profile_dir = args.profile_dir
        self.experimentID = experimentID
        self.run_mode = args.run_mode
        self.no_prior_profile = args.no_prior_profile
        
        os.makedirs(self.profile_dir, exist_ok=True)
        lh = 'default' if self.length_heterogeneity is None else self.length_heterogeneity
        self.profile_file = f"{self.profile_dir}/{self.model_n}_lambda={self.rate_lambda}_nodes={self.num_nodes}_retrain={self.retraining_rate}_LH={lh}.json"
        self.RECORD_MODE = False if self.run_mode == 'online' else True
        if self.task_assignment == 'workload':
            if self.profile_file is None or not os.path.exists(self.profile_file):
                self.RECORD_MODE = True
                
        if self.RECORD_MODE:
            self.retraining_rate = 0.5
            self.task_assignment = 'rr'
            self.active_selection = None
            self.eta, self.BT = None, None
            self.record_dict = defaultdict(list)
        else:
            if self.no_prior_profile or self.task_assignment != 'workload':
                self.eta, self.BT = None, None
                print("\n ** No offline profile used [pure ONLINE] **\n")
            else:
                file = json.load(open(self.profile_file))
                self.eta = file['eta'] if 'eta' in file else None # forward coefficient
                self.BT = file['BT'] if 'BT' in file else None # backward latency
                # self.profiled_losses = file['loss'] # list of offline training losses
                self.profiled_loss_dict = {int(taskID): float(instance['loss']) for (taskID, instance) in file['task_dict'].items()} # {taskID: loss}
                print("\n ** Offline profile loaded: eta={}, BT={} **\n".format(self.eta, self.BT))
        
        self.alpha = args.alpha
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.k = args.k
            
        # If dynamic method has decided the number of nodes, use that for 'rr+', 'random+', 'util+'
        if self.setting == 'isolated':
            self.isolated_split = args.isolated_split if args.isolated_split is not None else self.retraining_rate
            setting = f"isolated-split{self.isolated_split}"
        else:
            setting = self.setting
            
        self.used_nodes = self.num_nodes
        if '+' in self.task_assignment: # strong baselines (e.g., rr+, random+, util+)
            lh = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
            asl = f"active_{self.active_selection}" if self.active_selection is not None else "active_1.0"
            task_assignment = f"workload(a={self.alpha}|b={self.beta}|tau={self.epsilon})"
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{lh}-{self.length_distribution}_{self.retraining_rate}-{asl}_ID=0.json'
            
            if not os.path.exists(stats_f):
                raise ValueError(f'Cannot find dynamic result: {stats_f}')
            workload_res = json.load(open(stats_f))
            self.used_nodes = sum(x > 0 for x in workload_res["num_tasks (node)"].values())
            print(f"\n ** TASK ASSIGNMENT: {self.task_assignment} | Number of used nodes: {self.used_nodes} (out of {args.num_nodes}) **\n")
            
        if self.setting == 'isolated':
            if self.isolated_split != 100:
                num_train_nodes = max(1, round(self.used_nodes * self.isolated_split))
                if self.retraining_rate == 0:
                    num_train_nodes = 0
                num_test_nodes = max(1, self.used_nodes - num_train_nodes)
                if self.retraining_rate == 1:
                    num_test_nodes = 0
            else:
                num_test_nodes = self.used_nodes // 4
            
            self._test_nodes = list(range(num_test_nodes))
            self._train_nodes = list(range(num_test_nodes, self.used_nodes))
            print(f"** ISOLATED SYSTEM: Test nodes: {self._test_nodes}, Train nodes: {self._train_nodes} **")
        else:
            self._train_nodes = list(range(self.used_nodes))
            self._test_nodes = list(range(self.used_nodes))
        
        if self.priority is not None:
            self.ckpt_path = f'{self.output_dir}/stages-{self.task_assignment}_{self.model_n}_{setting}-{self.priority}_{self.workload}_{self.retraining_rate}'
        else:
            self.ckpt_path = f'{self.output_dir}/stages-{self.task_assignment}_{self.model_n}_{setting}_{self.workload}_{self.retraining_rate}'

        self.task_arrival = defaultdict(dict)
        self.task_trace = defaultdict(lambda: defaultdict(dict))
        self.train_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.user_task_record = defaultdict(dict)
        self.all_trace = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.node_accumulated_bubble = defaultdict(int)
        self.forward_start = torch.cuda.Event(enable_timing=True)
        self.forward_end = torch.cuda.Event(enable_timing=True)
        self.backward_start = torch.cuda.Event(enable_timing=True)
        self.backward_end = torch.cuda.Event(enable_timing=True)
        
        # Count number GPUs per node
        if self.num_gpus is None:
            self.num_gpus_per_node = torch.cuda.device_count() // self.num_nodes
        else:
            # Ensure the user-specified number of GPUs does not exceed the available GPUs
            available_gpus = torch.cuda.device_count()
            if self.num_gpus > available_gpus:
                raise ValueError(f"Requested {self.num_gpus} GPUs, but only {available_gpus} GPUs are available.")
        self.num_gpus_per_node = self.num_gpus // self.num_nodes

        # Define node instance
        self.distributed_nodes = {
            nodeID: Node(
                nodeID, 
                self.num_gpus_per_node, 
                init_device=nodeID * self.num_gpus_per_node,
            ) for nodeID in range(self.num_nodes)
        }
        self.memory_threshold = args.memory_threshold
        self.device_total_memory = torch.cuda.get_device_properties(0).total_memory
        self.timing_infos = {nodeID: defaultdict(list) for nodeID in range(self.num_nodes)}
        self.metrics = defaultdict(list)
        
        # Load the model and tokenizer
        self.access_token = args.access_token
        self.model_name_or_path = args.model_name_or_path
        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            token=self.access_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            token=self.access_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # print(f"Tokenizer: {self.tokenizer}, pad token id: {self.tokenizer.pad_token_id}, eos token id: {self.tokenizer.eos_token_id}, unk token id: {self.tokenizer.unk_token_id}")
        
        # Load datasets and dataloaders
        datasets = load_dataset(self.dataset_name_or_path)
        if self.RECORD_MODE:
            dataset = datasets['train']
        else:
            dataset = datasets['test']

        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples['query'], 
                padding=False, 
                truncation=True,
            )
            labels = self.tokenizer(
                examples['reference'], 
                padding=False, 
                truncation=True,
            )
            tokenized_inputs['labels'] = labels['input_ids']
            # tokenized_inputs['labels_attention_mask'] = labels['attention_mask']
            return tokenized_inputs
        
        # print("Dataset example (first 2 examples):", dataset[:2])
        dataset = dataset.map(
            tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=False,
        ).remove_columns(dataset.column_names)
        # print("Dataset tokenized example (first 2 examples):", dataset[:2])
        # print(f" ** pretrain name {self.model_name_or_path}, tokenized IDs range: ({min(min(dataset['input_ids']))}, {max(max(dataset['input_ids']))}) ** ")
        
        # Do sampling according to the length distribution
        input_lengths = [len(x) for x in dataset['input_ids']]
        self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length \
            = np.mean(input_lengths), np.std(input_lengths), np.median(input_lengths), min(input_lengths), max(input_lengths)
        print(" ** Original data length distribution: mean={}, std={}, medium={}, min={}, max={} **".format(
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length))
        if self.n_samples > 0:
            n_samples = min(self.n_samples, len(input_lengths))
            if self.length_heterogeneity is None:
                indices = random.sample(range(len(input_lengths)), n_samples)
                dataset = dataset.select(indices)
            else:
                indices = self._sample_subset_indices(input_lengths, n_samples, self.mean_length, self.length_heterogeneity)
                dataset = dataset.select(indices)
  
            subset_lengths = [len(x) for x in dataset['input_ids']]
            self.mean_length, self.std_length, self.medium_length, self.min_length, self.max_length \
                = np.mean(subset_lengths), np.std(subset_lengths), np.median(subset_lengths), min(subset_lengths), max(subset_lengths)
            print(f" ** Sampled {len(subset_lengths)} data points: mean={self.mean_length}, std={self.std_length}, medium={self.medium_length}, min={self.min_length}, max={self.max_length} **")
        
        # Data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )
        
        # Preloaded dataset
        self.distributed_preloaded_tasks, self.total_tasks, self.retraining_tasks, self.training_taskIDs, self.inference_taskIDs = self.get_preloaded_dataset(
            self.distributed_nodes, 
            self.dataloader, 
            retraining_rate=self.retraining_rate,
        )
        self.saving_steps = max(min(100, self.retraining_tasks // 2), 1)
        print(f" ** Total tasks: {self.total_tasks}, retraining tasks: {self.retraining_tasks}, saving steps: {self.saving_steps} ** ")
        self._training_step = 0
        self._trained_task_lengths = []
        
        # Select tasks to be trained
        if (self.active_selection is None) or self.RECORD_MODE: 
            self.backward_taskIDs = set(self.training_taskIDs)
        elif self.active_selection in ['random', 'first', 'last']:
            lh = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
            adapt_f = f'{self.output_dir}/metrics_{self.model_n}_{self.task_assignment}_{setting}-{self.priority}_{self.workload}-{lh}-{self.length_distribution}_{self.retraining_rate}-active_adaptive_ID=0.json'
            if os.path.exists(adapt_f):
                adapt_res = json.load(open(adapt_f))
                actual_trained_tasks = adapt_res['actual_retrained_tasks']
                print(f"\n ** Actual tasks to be trained: {actual_trained_tasks} (according to adaptive) **\n")
                if self.active_selection == 'random':
                    self.backward_taskIDs = set(random.sample(self.training_taskIDs, k=actual_trained_tasks))
                elif self.active_selection == 'first':
                    self.backward_taskIDs = set(self.training_taskIDs[:actual_trained_tasks])
                else:
                    self.backward_taskIDs = set(self.training_taskIDs[-actual_trained_tasks:])
            else:
                self.backward_taskIDs = set(self.training_taskIDs)
        elif '-' in self.active_selection: 
            # check if the string is a float number
            policy, ratio = self.active_selection.split('-')
            k = int(self.retraining_tasks * float(ratio))
            print("\n ** Selective training policy: {} | Ratio: {} | Number of tasks to be trained: {} **\n".format(policy, ratio, k))
            if policy == 'random':
                self.backward_taskIDs = set(random.sample(self.training_taskIDs, k=k))
            elif policy == 'length':
                # LCs = {taskID: self._compute_LC(self.distributed_preloaded_tasks[0][taskID].query['input_ids'].shape[1]) for taskID in training_taskIDs}
                # self.backward_taskIDs = set(sorted(LCs, key=lambda x: LCs[x], reverse=True)[:k]) # select the top-k tasks with the highest LC
                self.backward_taskIDs = set(sorted(self.training_taskIDs, key=lambda x: self.distributed_preloaded_tasks[0][x].query['input_ids'].shape[1])[:k])
            elif policy == 'loss':
                # We need to do backward for the top-k tasks with the highest losses
                if self.no_prior_profile:
                    self.backward_taskIDs = set(random.sample(self.training_taskIDs, k=k))
                else:
                    self.backward_taskIDs = set(sorted(self.training_taskIDs, key=lambda x: self.profiled_loss_dict[x], reverse=True)[:k])
            elif policy == 'first':
                self.backward_taskIDs = set(self.training_taskIDs[:k])
            elif policy == 'last':
                self.backward_taskIDs = set(self.training_taskIDs[-k:])
            elif policy == 'adaptive':
                # We either select the top-k tasks with the highest losses or the top-k tasks with the smallest lengths
                if self.no_prior_profile:
                    self.loss_taskIDs = set(random.sample(self.training_taskIDs, k=k))
                    self.length_taskIDs = set(random.sample(self.training_taskIDs, k=k))
                else:
                    self.loss_taskIDs = set(sorted(self.training_taskIDs, key=lambda x: self.profiled_loss_dict[x], reverse=True)[:k])
                    self.length_taskIDs = set(sorted(self.training_taskIDs, key=lambda x: self.distributed_preloaded_tasks[0][x].query['input_ids'].shape[1])[:k])
            else:
                raise ValueError(f"Invalid active selection: {self.active_selection}")
            
        # Save task length distribution for further analysis
        if self.save_length:
            length_dict = {taskID: task.query['input_ids'].shape[1] for taskID, task in enumerate(self.distributed_preloaded_tasks[0])}
            with open(f"{self.output_dir}/task_length_{self.model_n}_{setting}_{self.workload}_{self.retraining_rate}.json", 'w') as f:
                json.dump(length_dict, f, indent=4)

        # Stages
        self.distributed_stages = {
            nodeID: get_stages(
                self.config,
                token=self.access_token,
                model_name_or_path=self.model_name_or_path,
                num_stages=self.num_gpus_per_node,
                init_device=self.distributed_nodes[nodeID].init_device,
                timing_info=self.timing_infos[nodeID],
            ) for nodeID in range(self.num_nodes)
        }

        self.distributed_optimizers = {}
        for nodeID in range(self.num_nodes):
            all_parameters = []
            # Collect all parameters from stages in each node
            for stage in self.distributed_stages[nodeID]: 
                all_parameters.extend(list(stage.parameters()))
            self.distributed_optimizers[nodeID] = torch.optim.AdamW(all_parameters, lr=self.lr)
        
        self.distributed_schedulers = {
            nodeID: get_scheduler(
                "linear",
                optimizer=self.distributed_optimizers[nodeID], 
                num_warmup_steps=0, 
                num_training_steps=100,
            ) for nodeID in range(self.num_nodes)
        }
    
    
    def _sample_subset_indices(self, input_lengths: List[int], K: int, mu: float, std: float) -> List[int]:
        # Create an empty list to store the selected numbers
        selected_ids = set()
        lengths_dict = {} # {length: [idx1, idx2, ...]}
        for idx, length in enumerate(input_lengths):
            if length not in lengths_dict:
                lengths_dict[length] = [idx]
            else:
                lengths_dict[length].append(idx)

        # We draw K samples from the normal distribution
        for _ in range(K):
            sample = np.random.normal(mu, std)
            if sample in lengths_dict:
                selected_ids.add(lengths_dict[sample][0])
                lengths_dict[sample].pop(0) # pop the selected index
                if len(lengths_dict[sample]) == 0:
                    del lengths_dict[sample]
            else:
                # Find the number in 'numbers' that is closest to the sampled number
                closest_number = min(list(lengths_dict.keys()), key=lambda x: abs(x - sample))
                selected_ids.add(lengths_dict[closest_number][0])
                lengths_dict[closest_number].pop(0)
                if len(lengths_dict[closest_number]) == 0:
                    del lengths_dict[closest_number]
            
        return selected_ids

        
    def get_preloaded_dataset(
        self,
        distributed_nodes: Optional[Dict[int, Node]] = None, 
        dataloader: Optional[DataLoader] = None, 
        retraining_rate: Optional[float] = None,
    ) -> Tuple[Dict[int, List[Task]], int, int, List[int], List[int]]:
        
        print("Using preloaded data ...")
        distributed_nodes = distributed_nodes if distributed_nodes is not None else self.distributed_nodes
        dataloader = dataloader if dataloader is not None else self.dataloader
        retraining_rate = retraining_rate if retraining_rate is not None else self.retraining_rate
        distributed_preloaded_tasks = defaultdict(list)
        inference_taskIDs, training_taskIDs = [], []
        
        selected_data = []
        for i, batch in enumerate(dataloader):
            seq_length = batch['input_ids'].shape[1]
            # print(f"task {i}: {batch['input_ids'].shape} range ({batch['input_ids'].min()}, {batch['input_ids'].max()})")
            if batch['input_ids'].max() >= self.tokenizer.vocab_size:
                raise ValueError(f"Index error of token ID {batch['input_ids'].max()}")
            selected_data.append((seq_length, batch))
        inference_tasks = int(len(selected_data) * (1-retraining_rate))
        inference_ids = random.sample(range(len(selected_data)), k=inference_tasks)
            
        # Define the order of arrival input sequence length
        if self.length_distribution == 'ascending':
            selected_data.sort(key=lambda x: x[0])
        elif self.length_distribution == 'descending':
            selected_data.sort(key=lambda x: x[0], reverse=True)
        elif self.length_distribution == 'bursty':  # one short one long, ...
            selected_data.sort(key=lambda x: x[0])
            mid_index = len(selected_data) // 2
            short_data, long_data = selected_data[:mid_index], selected_data[mid_index:]
            # Rearrange sentences in groups of bptt
            tmp = []
            bptt = 1
            for i in range(0, max(len(short_data), len(long_data)), 1):
                tmp.extend(short_data[i:i+bptt])
                tmp.extend(long_data[i:i+bptt])
            selected_data = tmp
        elif self.length_distribution == 'random':
            pass
        else:
            raise ValueError(f"Invalid length distribution: {self.length_distribution}")
        
        
        # If workload is 'alternate', we need to create a list of varying lambda values (e.g., 1, 1, ..., 5, 5, ..., 30, 30, ...), each for X consecutive tasks
        if self.rate_lambda == -1 and retraining_rate != 1:
            print(f"\n ** Frequency-alternated {inference_tasks} serving requests **\n")
            lambda_values = [6, 12, 29, 15, 8, 20, 11, 6, 24, 19, 30, 14]
            total_lambda = sum(lambda_values)
            k = inference_tasks / total_lambda  # Proportionality constant

            # Initialize variables
            tasks_per_lambda_dict = {}
            total_assigned_tasks = 0
            inference_lambdas = []

            # Calculate tasks per lambda proportional to lambda values
            for lam in lambda_values:
                tasks = int(k * lam)
                tasks_per_lambda_dict[lam] = tasks
                total_assigned_tasks += tasks

            # Adjust for any remaining tasks due to integer rounding
            remaining_tasks = inference_tasks - total_assigned_tasks
            if remaining_tasks > 0:
                # Distribute the remaining tasks starting from the largest lambda
                sorted_lambdas = sorted(lambda_values, reverse=True)
                idx = 0
                while remaining_tasks > 0:
                    lam = sorted_lambdas[idx % len(sorted_lambdas)]
                    tasks_per_lambda_dict[lam] += 1
                    remaining_tasks -= 1
                    idx += 1

            # Construct a list of dictionaries to store ranges and node requirements
            lambda_ranges = []
            current_start_id = 0  # Initial task ID
            for lam in lambda_values:
                repetitions = tasks_per_lambda_dict[lam]
                inference_lambdas.extend([lam] * repetitions)
                test_nodes = self.num_nodes // 4 if lam <= 15 else self.num_nodes // 2
                current_end_id = current_start_id + repetitions - 1  # Calculate the end ID for this lambda range
                
                # Append the range to lambda_ranges
                lambda_ranges.append({
                    'lambda': lam,
                    'start_id': current_start_id,
                    'end_id': current_end_id,
                    'test_nodes': test_nodes
                })
                
                # Update the start ID for the next lambda
                current_start_id = current_end_id + 1

            if self.isolated_split == 100:
                # Now lambda_ranges contains the ranges and test node requirements
                lambda_ranges[-1]['end_id'] = len(selected_data) - 1
                self.ID2test = {}
                for taskID in range(len(selected_data)):
                    matched = False
                    for entry in lambda_ranges:
                        if entry['start_id'] <= taskID <= entry['end_id']:
                            self.ID2test[taskID] = entry['test_nodes']
                            matched = True
                            break
                    if not matched:
                        raise ValueError(f'No lambda intervals match task {taskID}')
                print(f'\nLambda Ranges and Test Node Requirements: {lambda_ranges}')


        # print(f'inference lambdas {len(inference_lambdas)}, inference_ids {len(inference_ids)}')
        j = 0
        for i, (_, batch) in enumerate(selected_data):
            # 10% of the time, produce a task with feedback
            require_training = i not in set(inference_ids)
            if require_training: 
                training_taskIDs.append(i)
                lamda = self.training_lambda
            else:
                inference_taskIDs.append(i)
                lamda = self.rate_lambda if self.rate_lambda != -1 else inference_lambdas[j]
                j += 1
                
            for nodeID, node in distributed_nodes.items():
                task = Task(
                    task_id=i,
                    rate_lambda=lamda,
                    query=_prepare_inputs(batch, device=node.init_device),
                    feedback=_prepare_inputs(batch['labels'], device=node.last_device),
                    node_id=nodeID,
                    num_gpus_per_node=node.num_gpus_per_node,
                    require_training=require_training,
                )
                distributed_preloaded_tasks[nodeID].append(task)
        
        return distributed_preloaded_tasks, len(selected_data), len(training_taskIDs), training_taskIDs, inference_taskIDs


    def producer(
        self,
        taskQueue: queue.Queue, 
        function: str = 'training',
    ) -> None:
        if function == 'training':
            IDs = self.training_taskIDs
        elif function == 'serving':
            IDs = self.inference_taskIDs
        elif function == 'mixed':
            IDs = list(range(len(self.distributed_preloaded_tasks[0])))
        else:
            raise ValueError(f'Unrecognized function type {function}! Must be one of the following: training, serving, mixed.')
        
        # Produce using the dataset
        for taskID in IDs:
            if self.workload == 'all':
                time.sleep(0)
            else:
                time.sleep(random.expovariate(self.distributed_preloaded_tasks[0][taskID].rate_lambda))
            # 10% of the time, produce a task with feedback
            # print("Producing task {} with input length {}".format(taskID, task.query['input_ids'].shape[1]))
            # Essentially, we are using preloaded data (task ID)
            taskQueue.put(taskID)
            # We record and calculate the response time for each user task (no retraining)
            release = time.time()
            if not self.distributed_preloaded_tasks[0][taskID].require_training:
                self.user_task_record[taskID]['release'] = release
            self.task_arrival[taskID]['release'] = release
            
        taskQueue.put(None)  # Signal the end of the dataset
        print(f"Producer finished producing {function} tasks")
        
        
    def _compute_LC(
        self, 
        task_length: int,
        mean_length: int = None,
        std_length: int = None,
    ): # Gaussian probability based on task length as length heterogeneity
        mean_length = self.mean_length if mean_length is None else mean_length
        std_length = self.std_length if std_length is None else std_length
        return (1 / (std_length * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((task_length - mean_length) / std_length) ** 2)
        
    # @profile
    def _check_do_backward(
        self, 
        taskID: int,
        task_length: int,
        training_step: int,
        total_training_tasks: int,
        # nodeID: int,
    ) -> bool:
        # print("Selective algorithm will be implemented!")
        # LC = self._compute_node_LC(nodeID, task_length)
        if self.active_selection is None:
            return True
        # if '-' in self.active_selection: 
        # check if the string is a float number
        policy, _ = self.active_selection.split('-')
        if policy != 'adaptive':
            do_backward = taskID in self.backward_taskIDs
        else:
            if self.no_prior_profile:
                P_l = self._compute_LC(task_length)
                P_adjusted = self.k * P_l + (1 - self.k) * (1 - (training_step / total_training_tasks))
                # P_adjusted = P_l
            else:
                P_adjusted = self.k * int(taskID in self.loss_taskIDs) + (1 - self.k) * int(taskID in self.length_taskIDs)
            do_backward = P_adjusted > random.random()
        
        return do_backward
        
        
    # @profile 
    def _compute_rewards(self, nodeID: int, taskID: int, do_backward: bool = False):
        """
        Predict the execution traces for the incoming task and compute the reward (bubble rate & response time).
        """
        length = self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape[1]
        # Initialize bubble rate and response time
        fb = self.task_arrival[taskID]['release']
        br, bi, init_id, finished_ids = 0, 0, 0, []
        previous_taskIDs = list(self.all_trace[nodeID].keys())
        previous_train_taskIDs = list(self.train_trace[nodeID].keys())
        eta = self.eta if self.eta is not None else 2e-5
        BT = self.BT if self.BT is not None else 0.07
            
        # Get/Estimate previous forward execution traces
        if previous_taskIDs:
            last_taskID = previous_taskIDs[-1]
            arrival_gap = self.task_arrival[taskID]['release'] - self.task_arrival[last_taskID]['release']
        else:
            last_taskID = None
            arrival_gap = 0
            
        for stageID in range(self.num_gpus_per_node):
            pfe = self.all_trace[nodeID][last_taskID][stageID]['fe'] if last_taskID is not None else fb
            fb = max(pfe, fb)
            fe = fb + eta * length ** 2  # f_e = f_b + eta * L^2
            i = init_id
            
            for i in range(init_id, len(previous_train_taskIDs)):
                ptID = previous_train_taskIDs[i]
                if fe <= self.train_trace[nodeID][ptID][stageID]['bb']:
                    br += eta * length ** 2 # update BR
                    break
                fb = max(fb, self.train_trace[nodeID][ptID][stageID]['be']) # forward starts at least at the end of the backward
                fe = fb + eta * length ** 2 # f_e = f_b + eta * L^2
                if stageID == 0: # this training task has finished
                    finished_ids.append(ptID)
                    
            # Write the forward and backward execution traces
            self.task_trace[nodeID][stageID]['fb'] = fb
            self.task_trace[nodeID][stageID]['fe'] = fe
            bi += max(fb - pfe - (i - init_id) * BT, 0) # update BI
            init_id = i + 1 # update the initial training check index for the next stage
        
        if do_backward:
            bb = fe # backward starts at the end of the forward
            for stageID in range(self.num_gpus_per_node-1, -1, -1):
                self.task_trace[nodeID][stageID]['bb'] = bb
                self.task_trace[nodeID][stageID]['be'] = bb + BT
                # bi += max(bb - self.task_trace[nodeID][stageID]['fe'], 0) # update BI
                bb += BT
        
        # Clear the finished training tasks
        for ptID in finished_ids:
            self.train_trace[nodeID].pop(ptID)
            self.all_trace[nodeID].pop(ptID)
                    
        # Compute reward (bubble profit and response time)
        response = fe - self.task_arrival[taskID]['release']
        fbi = max(self.epsilon, bi/self.num_gpus_per_node - arrival_gap)
        lc = self.distributed_nodes[nodeID].length_consistency(length)
        reward = (br - fbi + self.beta * lc) / (self.alpha * response + self.epsilon)
        # reward = (br - fbi) / (self.alpha * response + self.beta * lc + self.epsilon)
        
        return reward
            
        
    # @profile
    def _assign_node(self, node_list: List[int], taskID: int, do_backward: bool = False):
        
        if len(node_list) == 1:
            return node_list[0]
        if 'random' in self.task_assignment or self.RECORD_MODE: # Random assignment
            return random.choice(node_list)
        elif 'rr' in self.task_assignment: # Round-robin
            return node_list[taskID % len(node_list)]
        elif 'util' in self.task_assignment: # LUF: choose the node with the least average utilization across all its GPUs
            # gputils = self._get_gpu_utilization()
            # return min(node_list, key=lambda nodeID: gputils[nodeID])
            gpus: List[GPU] = GPUtil.getGPUs()
            return min(node_list, key=lambda nodeID: np.mean([gpus[device].memoryUtil for device in self.distributed_nodes[nodeID].device_ids]))
            # # return min(node_list, key=lambda nodeID: np.mean([torch.cuda.memory_allocated(device) for device in self.distributed_nodes[nodeID].device_ids]))
        elif self.task_assignment == 'workload':
            # # Choose the node with the least workload (number of tasks in the first device queue)
            # return min(node_list, key=lambda nodeID: self.distributed_nodes[nodeID].device_queues[0].qsize())
            reward_list = []
            for nodeID in node_list:
                try:
                    reward = self._compute_rewards(nodeID, taskID, do_backward=do_backward)
                except Exception as e:
                    reward = random.random()
                    logging.error(f"[node {nodeID} | task {taskID}] Bubble calculation error occurred: {e}")
                reward_list.append(reward)
                # br_list.append(br)
                # bi_list.append(bi)
            
            # Greedy: assign task to the node with the highest reward (lowest average utilization)
            best_index = np.argmax(reward_list) 
            # make it more smart, if some reward values are the same, randomly choose one
            # best_index = np.random.choice(np.where(reward_list == np.max(reward_list))[0])
            # # Instead of greedy, let's use sampling with probability proportional to the reward
            # reward_list = F.softmax(torch.FloatTensor(reward_list), dim=0).numpy()  
            # best_index = np.random.choice(range(len(reward_list)), p=reward_list)
            nodeID = node_list[best_index]
            # self.node_accumulated_bubble[nodeID] += bi_list[best_index] - br_list[best_index]
            return nodeID 
        else:
            raise ValueError(f"Invalid task assignment method: {self.task_assignment}")
        
    # @profile
    def globalScheduler(
        self, 
        taskQueue: queue.Queue, 
    ):
        # Global scheduler
        while True:
            taskID: int = taskQueue.get() # ID
            if taskID is None:
                print("Global scheduler finished scheduling tasks")
                for node in self.distributed_nodes.values():
                    # node.device_queues[0].put(None)
                    node.device_queues[0].put((float('inf'), float('inf'))) # for priority_queue, use a large number to signal the end
                break
            
            # Active selection
            select_start = time.time()
            if self.distributed_preloaded_tasks[0][taskID].require_training:
                do_backward = self._check_do_backward(
                    taskID, 
                    self.distributed_preloaded_tasks[0][taskID].query['input_ids'].shape[1], 
                    self._training_step, 
                    self.retraining_tasks,
                ) if self.task_assignment == 'workload' else True
            else:
                do_backward = False
            self.metrics['active_selection'].append(time.time() - select_start)
            
            # Task assignment
            assign_start = time.time()
            if self.setting != 'isolated':
                nodeID = self._assign_node(node_list=list(range(self.used_nodes)), taskID=taskID, do_backward=do_backward)
            else:
                ################ Dynamic separate ##############
                test_nodes = self._test_nodes
                train_nodes = self._train_nodes
                if self.rate_lambda == -1 and self.isolated_split == 100:  # dynamic
                    # if self.num_gpus != 8:
                    #     raise ValueError(f'Dynamic separate only support 8 GPUs! Currently have {self.num_gpus} GPUs!')
                    # if taskID <= self.ID_thresholds[0]:
                    #     num_test_nodes = self.used_nodes // 4
                    # elif self.ID_thresholds[0] < taskID <= self.ID_thresholds[1]:
                    #     num_test_nodes = self.used_nodes // 2
                    # else:
                    #     num_test_nodes = self.used_nodes * 3 // 4
                    num_test_nodes = self.ID2test[taskID]
                    test_nodes = list(range(num_test_nodes))
                    train_nodes = list(range(num_test_nodes, self.used_nodes))
                    print(f' ** Dynamic separate: taskID {taskID}, test_nodes {test_nodes}, train_nodes {train_nodes} ** ')
                ################ Dynamic separate ##############
                
                if self.distributed_preloaded_tasks[0][taskID].require_training:  # assign to one of the training nodes
                    nodeID = self._assign_node(node_list=train_nodes, taskID=taskID, do_backward=do_backward)
                else: # assign to one of the test nodes
                    nodeID = self._assign_node(node_list=test_nodes, taskID=taskID, do_backward=do_backward) 

            self.distributed_preloaded_tasks[nodeID][taskID].do_backward = do_backward
                    
            # Update train trace and all trace with the task trace
            if self.task_assignment == 'workload':
                self.all_trace[nodeID][taskID] = self.task_trace[nodeID].copy()
                if do_backward:
                    self.train_trace[nodeID][taskID] = self.task_trace[nodeID].copy()
                        
            self.metrics['node_assignment'].append(time.time() - assign_start)

            # Each node queue store task IDs
            if self.setting == 'interval':
                seq_length = self.distributed_preloaded_tasks[0][taskID].query['input_ids'].shape[1]
                if self.priority is None or self.priority == 'LLF':
                    priority = seq_length
                elif self.priority == 'MLF':
                    priority = -seq_length
                else:
                    raise ValueError(f"Invalid priority type: {self.priority}")
                self.distributed_nodes[nodeID].device_queues[0].put((priority, taskID))
            else:
                self.distributed_nodes[nodeID].device_queues[0].put((taskID, taskID))
            
            # Record the node allocation 
            self.distributed_nodes[nodeID].update_length_stats(
                length=self.distributed_preloaded_tasks[nodeID][taskID].query['input_ids'].shape[1],
                computation_type='training' if do_backward else 'test',
            )
            # print("Global scheduler scheduled task {} (requre_training={}) to node {}".format(taskID, self.distributed_preloaded_tasks[0][taskID].require_training, nodeID))
    
    
    def _check_device_availability(self, device: int, threshold: float = 0.8):
        """
        Check if the device has enough available memory.
        Args:
        - device: The device to check.
        - threshold: The maximum allowed memory utilization ratio.
        Returns:
        - is_available: Boolean indicating if the device is available.
        """
        # Get device memory status
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = self.device_total_memory - allocated_memory
        # Calculate the available memory ratio
        available_ratio = available_memory / self.device_total_memory
        # Check if the available memory ratio is above the threshold
        return available_ratio > (1 - threshold)

    
    
    def _wait_for_device_availability(self, device: int, check_interval: float = 0.1, threshold: float = 0.8):
        """
        Wait until the device is available based on memory usage.
        Args:
        - device: The device to wait for.
        - check_interval: How often to check the device status (in seconds).
        - threshold: The maximum allowed memory utilization ratio.
        """
        wait_count = 0
        while not self._check_device_availability(device, threshold):
            # print(f"Waiting for device {device} to become available...")
            time.sleep(check_interval)
            wait_count += 1
            if wait_count > 100:
                print(f"Device {device} is not available after waiting for {wait_count * check_interval} seconds")
                break

            
    def forward(
        self, 
        task: Task,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        stageID: int,
        nodeID: int,
        device: int, 
        timing_info: Dict[str, List[float]],
    ) -> Tuple[torch.Tensor, ...]:
        
        self._wait_for_device_availability(device, threshold=self.memory_threshold)
        ################################ Print usage ############################
        # for name, input in inputs.items():
        #     if isinstance(input, torch.Tensor):
        #         print(f'[stage {stageID}] {name}: {input.shape} - range ({input.min()}, {input.max()})')
        #     else:
        #         print(f'[stage {stageID}] {name}: {input}')
        ##########################################################################
        try:
            if task.require_training: # this is a retraining task
                fb = record_time(device, 'start', 'forward_grad', task.task_id, timing_info)
                self.task_trace[nodeID][stageID]['fb'] = fb
                if task.task_id in self.train_trace[nodeID]: # this training task has been completely recorded
                    self.train_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task
                    self.all_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task

                tuple_outputs = self.distributed_stages[nodeID][stageID](**inputs, labels=task.feedback)
                # torch.cuda.synchronize() # synchronize the device
                fe = record_time(device, 'end', 'forward_grad', task.task_id, timing_info)
                
                self.task_trace[nodeID][stageID]['fe'] = fe
                if task.task_id in self.train_trace[nodeID]: # this training task has been completely recorded
                    self.train_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
                    self.all_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
           
            else: # this is a user (test) task
                fb = record_time(device, 'start', 'forward', task.task_id, timing_info)
                self.task_trace[nodeID][stageID]['fb'] = fb
                if task.task_id in self.all_trace[nodeID]: # this task has been completely recorded
                    self.all_trace[nodeID][task.task_id][stageID]['fb'] = fb # update the already recorded task
                    
                if stageID == 0: # first stage
                    self.user_task_record[task.task_id]['start'] = fb
                    
                with torch.no_grad():
                    # print(f'current stage {stageID}: {self.distributed_stages[nodeID][stageID]}')
                    tuple_outputs = self.distributed_stages[nodeID][stageID](**inputs, labels=task.feedback)
                    # print(f'outputs: {tuple_outputs}')
                # torch.cuda.synchronize() # synchronize the device
                fe = record_time(device, 'end', 'forward', task.task_id, timing_info)
                
                self.task_trace[nodeID][stageID]['fe'] = fe
                if task.task_id in self.all_trace[nodeID]: # this task has been completely recorded
                    self.all_trace[nodeID][task.task_id][stageID]['fe'] = fe # update the already recorded task
                
                if stageID == self.num_gpus_per_node - 1: # last stage
                    self.user_task_record[task.task_id]['end'] = fe
                    logging.info(f"[Node {nodeID}] Task {task.task_id} finished forward pass!")
                    
            if self.eta is None:
                self.eta = (fe - fb) / (task.query['input_ids'].shape[1] ** 2)
                    
            if self.RECORD_MODE:
                # self.forward_end.record()
                # torch.cuda.synchronize()
                # Profile forward time (seconds) per stage
                # self.eta = self.forward_start.elapsed_time(self.forward_end) / (task.query['input_ids'].shape[1] ** 2 * 1e3)
                self.record_dict['etas'].append(((fe - fb) / (task.query['input_ids'].shape[1] ** 2), task.task_id))
                self.record_dict['FTs'].append((fe - fb, task.task_id))
                
        except Exception as e:
            logging.error(f"[Node {nodeID} - stage {stageID} - device {device}] Forward error occurred: {e}")
            tuple_outputs = None
        
        return tuple_outputs
    

    def device_inference(
        self,
        stageID: int,
        nodeID: int,
        timing_info: Dict[str, List[float]],
        preloaded_tasks: List[Task], 
        deviceQueue: Union[queue.Queue, queue.PriorityQueue],
        nextdeviceQueue: Optional[queue.Queue] = None,
        init_device: Optional[int] = None,
    ):
        raise NotImplementedError("device_inference method must be implemented")             


    def node_inference(
        self,
        nodeID: int,
        node: Node,
    ):
        # We use num_gpus_per_node workers to simulateously get task from the queue and inference
        with ThreadPoolExecutor(max_workers=self.num_gpus_per_node) as executor:
            # futures = []
            for stageID in range(self.num_gpus_per_node):
                future = executor.submit(
                    self.device_inference, 
                    stageID,
                    nodeID,
                    self.timing_infos[nodeID], 
                    self.distributed_preloaded_tasks[nodeID],
                    node.device_queues[stageID],
                    nextdeviceQueue=node.device_queues[stageID+1] if stageID != len(self.distributed_stages[nodeID]) - 1 else None,
                    init_device=node.init_device,
                )
            #     futures.append(future)
            # for future in futures:
            #     try:
            #         # Set a timeout for each task. Adjust the timeout value as needed.
            #         future.result(timeout=60)  # Timeout set to 60 seconds
            #     except TimeoutError:
            #         # Handle the timeout, for example, by logging an error, retrying the task, or skipping it.
            #         print(f"Task execution exceeded the timeout limit and was aborted. Node ID: {nodeID}")
                
        print("Node {} finished inference".format(node.node_id))


    def run_stages_concurrently(self):
        with ThreadPoolExecutor(max_workers=len(self.distributed_nodes)) as executor:
            for nodeID, node in self.distributed_nodes.items():
                future = executor.submit(self.node_inference, nodeID, node)
        
        
    def save_timing_info(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if self.setting == 'isolated':
            setting = f"isolated-split{self.isolated_split}"
        else:
            setting = self.setting
        
        length_heterogeneity = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
        active_selection = f"active_{self.active_selection}" if self.active_selection is not None else "active_1.0"
        # task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta}|tau={self.epsilon})" if self.task_assignment == 'workload' else self.task_assignment
        task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta}|tau={self.epsilon})" if (self.task_assignment == 'workload' or '+' in self.task_assignment) else self.task_assignment
        for nodeID, timing_info in self.timing_infos.items():
            if self.no_prior_profile:
                stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-prior-profile_node{nodeID}.json'
            else:
                stats_f = f'{self.output_dir}/timing_info_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_node{nodeID}.json'
            with open(stats_f, 'w') as f:
                json.dump(timing_info, f, indent=4)
        
        
    def calculate_metrics(
        self, 
        metrics: Optional[Dict[str, Union[float, int]]] = None,
    ):
        metrics = metrics if metrics is not None else self.metrics
        if self.setting == 'isolated':
            setting = f"isolated-split{self.isolated_split}"
        else:
            setting = self.setting
            
        # Calculate metrics
        global_min_time, global_max_time = float('inf'), float('-inf')
        total_idles = []
        total_latencies = []
        total_runtime = 0
        node_idles = defaultdict(list)
        node_latencies = defaultdict(list)
        node_timelines = {}
        for nodeID, node in self.distributed_nodes.items():
            timing_info = {k: [[t[0], t[1]] for t in v] for k, v in self.timing_infos[nodeID].items()}
            if not timing_info:
                continue
            
            node_min_time, node_max_time = float('inf'), float('-inf')
            for gpu_id in range(self.num_gpus_per_node):
                min_t, max_t = float('inf'), float('-inf')
                gpu_idx = node.init_device + gpu_id
                starts = timing_info.get(f"{gpu_idx}_start", [])
                ends = timing_info.get(f"{gpu_idx}_end", [])
                if len(starts) == 1:
                    idles = []
                else:
                    idles = [start - end for (start, _), (end, _) in zip(starts[1:], ends[:-1]) if (start > end)]
                total_idles.extend(idles)
                node_idles[nodeID].extend(idles)
                
                tasks = list(zip(starts, ends))
                for i, ((start, start_label), (end, _)) in enumerate(tasks):
                    metrics[start_label].append(end - start)
                    min_t = min(min_t, start)
                    max_t = max(max_t, end)
                total_latencies.append(max_t - min_t)
                node_latencies[nodeID].append(max_t - min_t)
                global_min_time = min(global_min_time, min_t)
                global_max_time = max(global_max_time, max_t)
                node_min_time = min(node_min_time, min_t)
                node_max_time = max(node_max_time, max_t)
                
            node_timelines[nodeID] = (node_min_time, node_max_time)
            if node_min_time == float('inf') or node_max_time == float('-inf'):
                continue
            else:
                total_runtime += node_max_time - node_min_time
                    
        bubble_rate = sum(total_idles) / sum(total_latencies) if sum(total_latencies) > 0 else 0
        for key, value in metrics.items():
            if key == 'loss':
                losses = value
            metrics[key] = sum(value) / len(value)
        
        # Calculate response times
        metrics['num_tasks'] = self.total_tasks
        metrics['retrain_tasks'] = self.retraining_tasks
        metrics['actual_retrained_tasks'] = len(self._trained_task_lengths)
        metrics['user_tasks'] = len(self.user_task_record)
        metrics['bubble_rate'] = bubble_rate 
        metrics['total_runtime'] = total_runtime
        metrics['end2end_latency'] = global_max_time - global_min_time
        metrics['throughput'] = self.total_tasks / (global_max_time - global_min_time)
        metrics['num_tasks (node)'] = {nodeID: self.distributed_nodes[nodeID].num_tasks for nodeID in self.distributed_nodes}
        metrics['retrain_tasks (node)'] = {nodeID: self.distributed_nodes[nodeID].train_tasks for nodeID in self.distributed_nodes}
        metrics['user_tasks (node)'] = {nodeID: self.distributed_nodes[nodeID].test_tasks for nodeID in self.distributed_nodes}
        metrics['bubble_rate (node)'] = {
            nodeID: sum(idles) / sum(latencies) if sum(latencies) > 0 else 0 
            for nodeID, idles, latencies in zip(node_idles.keys(), node_idles.values(), node_latencies.values())
        }
        metrics['end2end_latency (node)'] = {nodeID: node_timelines[nodeID][1] - node_timelines[nodeID][0] for nodeID in node_timelines}
        metrics['throughput (node)'] = {nodeID: self.distributed_nodes[nodeID].num_tasks / (node_timelines[nodeID][1] - node_timelines[nodeID][0]) for nodeID in node_timelines}
        metrics['node_timelines'] = node_timelines
        metrics['idles_sum'] = sum(total_idles)
        metrics['idles_sum (node)'] = {nodeID: sum(idles) for nodeID, idles in node_idles.items()}
        metrics['idles_avg'] = sum(total_idles) / len(total_idles) if total_idles else 0
        metrics['idles_avg (node)'] = {nodeID: sum(idles) / len(idles) if idles else 0 for nodeID, idles in node_idles.items()}
        metrics['length_statistics'] = {
            'mean': self.mean_length,
            'std': self.std_length,
            'medium': self.medium_length,
            'min': self.min_length,
            'max': self.max_length,
        }
        metrics['length_statistics (node)'] = {}
        for nodeID in self.distributed_nodes:
            node_lengths = [task['length'] for task in self.distributed_nodes[nodeID].task_allocation]
            if not node_lengths:
                continue
            medium_length = np.median(node_lengths)
            min_length = min(node_lengths)
            max_length = max(node_lengths)
            metrics['length_statistics (node)'][nodeID] = {
                'mean': self.distributed_nodes[nodeID].mean,
                'std': self.distributed_nodes[nodeID].std,
                'medium': medium_length,
                'min': min_length,
                'max': max_length,
            } 
        
        if self.user_task_record:
            # total_response_time, total_wait_time, total_inference_time = 0, 0, 0
            response_times, wait_times, latencies = [], [], []
            user_global_min_time, user_global_max_time = float('inf'), float('-inf')
            for taskID, record_dict in self.user_task_record.items():
                # total_response_time += record_dict['end'] - record_dict['release']
                # total_wait_time += record_dict['start'] - record_dict['release']
                # total_inference_time += record_dict['end'] - record_dict['start']
                if 'start' not in record_dict or 'end' not in record_dict:
                    print(f"Unrecorded user request (ID={taskID})!")
                    continue
                user_global_min_time = min(user_global_min_time, record_dict['start'])
                user_global_max_time = max(user_global_max_time, record_dict['end'])
                response_times.append(record_dict['end'] - record_dict['release'])
                wait_times.append(record_dict['start'] - record_dict['release'])
                latencies.append(record_dict['end'] - record_dict['start'])
                
            metrics['user_wait_avg'] = sum(wait_times) / len(self.user_task_record)
            metrics['user_inference_avg'] = sum(latencies) / len(self.user_task_record)
            metrics['user_response_avg'] = sum(response_times) / len(self.user_task_record)
            metrics['user_end2end_latency'] = user_global_max_time - user_global_min_time
            metrics['user_throughput'] = len(self.user_task_record) / (user_global_max_time - user_global_min_time)
            metrics['user_responses'] = response_times # list
            
        metrics['idles'] = total_idles # list
        metrics['losses'] = losses # list
        # metrics['losses (node)'] = {nodeID: list(self.distributed_nodes[nodeID].losses.values()) for nodeID in self.distributed_nodes} 
        metrics['task_stats (node)'] = {
            nodeID: self.distributed_nodes[nodeID].task_allocation 
            for nodeID in self.distributed_nodes
        }   
        
        # Save metrics
        os.makedirs(self.output_dir, exist_ok=True)
        length_heterogeneity = f"hetero{self.length_heterogeneity}" if self.length_heterogeneity is not None else "hetero_default"
        active_selection = f"active_{self.active_selection}" if self.active_selection is not None else "active_1.0"
        task_assignment = f"{self.task_assignment}(a={self.alpha}|b={self.beta}|tau={self.epsilon})" if (self.task_assignment == 'workload' or '+' in self.task_assignment) else self.task_assignment
        if self.no_prior_profile:
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_no-prior-profile_ID={self.experimentID}.json'
        else:
            stats_f = f'{self.output_dir}/metrics_{self.model_n}_{task_assignment}_{setting}-{self.priority}_{self.workload}-{length_heterogeneity}-{self.length_distribution}_{self.retraining_rate}-{active_selection}_ID={self.experimentID}.json'
        with open(stats_f, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {stats_f}")
        
        
    def run(self):
        # Run the stages concurrently
        if self.RECORD_MODE:
            print("\n ** Running in RECORD mode **\n")
        
        task_queue = queue.Queue()
        num_workers = 2
        if self.training_taskIDs:
            num_workers += 1
        if self.inference_taskIDs:
            num_workers += 1

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            if self.inference_taskIDs:
                executor.submit(
                    self.producer,
                    task_queue, 
                    'serving',
                )
            if self.training_taskIDs:
                executor.submit(
                    self.producer,
                    task_queue, 
                    'training',
                )
            # future1 = executor.submit(
            #     self.producer,
            #     task_queue, 
            #     'mixed',
            # )
            executor.submit(
                self.globalScheduler,
                task_queue,
            )
            executor.submit(self.run_stages_concurrently)
        
        if self.RECORD_MODE:        
            # Save recorded dict
            losses = [loss for loss, _ in self.record_dict['loss']]
            # Quantile (25, 50, 75) for record_dict['loss]
            self.record_dict['loss_stats'] = {'mean': np.mean(losses), 'std': np.std(losses)}
            for quantile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
                self.record_dict['loss_stats'][f'{quantile}%'] = np.percentile(losses, quantile)
                
            loss_dict = {taskID: loss for (loss, taskID) in self.record_dict['loss']}
            length_dict = {taskID: length for (length, taskID) in self.record_dict['length']}
            
            # First get the forward dict where each taskID corresponds to multiple forward times
            tmp = defaultdict(list)
            for (ft, taskID) in self.record_dict['FTs']:
                tmp[taskID].append(ft)
            # Then take the sum of the forward times for each taskID
            forward_dict = {taskID: np.mean(fts) for taskID, fts in tmp.items()}
            eta_dict = {taskID: et for et, taskID in self.record_dict['etas']}
            backward_dict = {taskID: bt for bt, taskID in self.record_dict['BTs']}
            self.record_dict['eta'] = np.mean(list(eta_dict.values()))
            self.record_dict['BT'] = np.mean(list(backward_dict.values()))
            
            self.record_dict['task_dict'] = {
                taskID: {
                    'loss': loss_dict.get(taskID, 'None'),
                    'length': length_dict.get(taskID, 'None'),
                    'forward': forward_dict.get(taskID, 'None'),
                    'backward': backward_dict.get(taskID, 'None'),
                    'eta': eta_dict.get(taskID, 'None'),
                } for taskID in loss_dict
            }
            # Remove etas, FTs, BTs, loss, length
            for key in ['etas', 'FTs', 'BTs', 'loss', 'length']:
                self.record_dict.pop(key)
            
            # os.makedirs(self.profile_file, exist_ok=True)
            with open(self.profile_file, 'w') as f:
                json.dump(self.record_dict, f, indent=4)
            
        else:
            # Delete checkpoint file in the disk if self.ckpt_path is not None
            if self.ckpt_path is not None:
                for j in range(self.num_gpus_per_node):
                    if os.path.exists(f"{self.ckpt_path}_stage{j}.pt"):
                        os.remove(f"{self.ckpt_path}_stage{j}.pt")
            
            # Save timing info
            self.save_timing_info()
            
            # Calculate metrics
            self.calculate_metrics()

    
    
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name_or_path', type=str, default='data/Anthropic', help='dataset name')
    parser.add_argument('--model_name_or_path', type=str, help='model name or path')
    parser.add_argument('--model_name', type=str, default='dummy', help='model name')
    parser.add_argument('--memory_threshold', type=float, default=0.5, 
                        help='threshold for maximum memory allocation in each GPU device')
    parser.add_argument('--access_token', type=str, default=None, help='access token')
    parser.add_argument('--num_nodes', type=int, default=2)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--save_length', action='store_true', help='save the length of each task')
    parser.add_argument('--setting', type=str, default='active', choices=['active','interval','isolated'], 
                        help='training setting')
    parser.add_argument('--isolated_split', type=float, default=None, 
                        help='split ratio for isolated test & train nodes. If not provided, the retraining rate is used.')
    parser.add_argument('--priority', type=str, default='FIFO', help='scheduling priority, default: FIFO')
    parser.add_argument('--task_assignment', type=str, default='random', choices=['rr', 'random', 'workload', 'util', 'rr+', 'random+', 'util+'], 
                        help='node level scheduling policy')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--retraining_rate', type=float, default=0.1, help='proportion of training tasks')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--rate_lambda', type=int, default=10, help='Average number of inference requests per second')
    parser.add_argument('--training_lambda', type=int, default=30, help='Average number of training requests per second')
    parser.add_argument('--alpha', type=float, default=1, help='response time coefficient')
    parser.add_argument('--beta', type=float, default=1, help='length heterogeneity coefficient')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='small value to avoid division by zero')
    parser.add_argument('--k', type=float, default=0.5, help='weight for balancing the loss and length consistency for adaptive training')
    parser.add_argument('--workload', type=str, default='poisson', help='workload arrival pattern')
    parser.add_argument('--length_distribution', type=str, default='random', choices=['random', 'ascending', 'descending', 'bursty'], 
                        help='distribution of input sequence length')
    parser.add_argument('--length_heterogeneity', type=int, default=None, 
                        help='standard deviation of the length distribution of the sampled subset')
    parser.add_argument('--active_selection', type=str, default=None,
                        help='active selection ratio for training tasks')
    parser.add_argument('--profile_dir', type=str, default='profile', help='directory to save profiling results')
    parser.add_argument('--output_dir', type=str, default='prof')
    parser.add_argument('--experiments', type=int, default=1, help='number of experiments')
    parser.add_argument('--run_mode', type=str, default='online', choices=['online', 'offline'], help='Whether to use RECORD MODEL for offline profiling')
    parser.add_argument('--no_prior_profile', action='store_true', help='Whether to use offline profiling results as prior')
    args = parser.parse_args()
    
    for i in range(args.experiments):
        run_experiment(args, i)
    
