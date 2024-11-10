import sys 
sys.dont_write_bytecode = True
import copy
from collections import defaultdict
from collections.abc import Mapping
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
from .llama import (
    LlamaStartingStage,
    LlamaIntermediateStage,
    LlamaEndingStage,
)
from .dialogpt import (
    GPTStartingStage,
    GPTIntermediateStage,
    GPTEndingStage,
    GPTOneStage,
)
from transformers import (
    LlamaConfig, 
    AutoTokenizer, 
    AutoConfig, 
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache


def _prepare_input(
    data: Union[torch.Tensor, Any],
    device: torch.device = 'cuda',
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: _prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(_prepare_input(v, device) for v in data)
    elif isinstance(data, DynamicCache):
        data.key_cache = _prepare_input(data.key_cache, device)
        data.value_cache = _prepare_input(data.value_cache, device)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    return data
    

def _prepare_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
    device: torch.device = 'cuda',
) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    new_inputs = _prepare_input(inputs, device=device)
    if new_inputs is None or len(new_inputs) == 0:
        raise ValueError(
            "The batch received was empty, your model won't be able to train on it."
        )
    return new_inputs


def get_stages(
    config: LlamaConfig,
    token: str,
    # tokenizer: Union[LlamaTokenizer, GPT2Tokenizer],
    model_name_or_path: str,
    num_stages: int,
    hidden_layers_assignments: List[int] = None,
    init_device: int = 0,
    timing_info: dict = None,
) -> List[Union[GPTEndingStage, LlamaEndingStage]]:
    """
    start_stage (stage_id == 0): nn.Embeddings + [LlamaDecoderLayer] * N_s
    intermediate_stage (0 < stage_id < num_stages - 1): [LlamaDecoderLayer] * N_i
    end_stage (stage_id == num_stages - 1): [LlamaDecoderLayer] * N_e + BertPreTrainingHeads

    N_s, N_i, N_e: the number of hidden layers (LlamaDecoderLayers) for each stage
    """
    assert num_stages > 0, 'At least 1 stage is required.'
    
    if hidden_layers_assignments is None:
        """
        Assign the number of hidden layers (BertLayers) so that
        the following are satisfied: 
            N_e <= N_s <= N_i
        """
        hidden_layers_assignments = [0] * num_stages
        for i in range(config.num_hidden_layers):
            hidden_layers_assignments[-((i + 2) % num_stages)] += 1
    assert len(hidden_layers_assignments) == num_stages
    
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # low_cpu_mem_usage=True,
        token=token,
        config=config,
        # torch_dtype="float32",
    )
    # if 'llama' in model_name_or_path.lower():
    #     pretrained_model.resize_token_embeddings(len(tokenizer))
    #     pretrained_model = pretrained_model.half()
    #     pretrained_model.gradient_checkpointing_enable()
    print("Model hidden dim {}, num layers {}, num heads {}, num parameters {}".format(
        pretrained_model.config.hidden_size, 
        pretrained_model.config.num_hidden_layers, 
        pretrained_model.config.num_attention_heads, 
        pretrained_model.num_parameters()
    ))
    
    pipeline_stages = []
    totoal_params = 0
    start_stage_class = GPTStartingStage if 'gpt' in model_name_or_path.lower() else LlamaStartingStage
    intermediate_stage_class = GPTIntermediateStage if 'gpt' in model_name_or_path.lower() else LlamaIntermediateStage
    end_stage_class = GPTEndingStage if 'gpt' in model_name_or_path.lower() else LlamaEndingStage
    # gradient_checkpointing_kwargs={"use_reentrant":False}
    
    for stage_id in range(num_stages):
        # Overwrite num_hidden_layers with the number for this stage
        config = copy.deepcopy(config)
        config.pad_token_id = config.eos_token_id
        config.num_hidden_layers = hidden_layers_assignments[stage_id]
        device = init_device + stage_id
        layer_ids = list(range(stage_id * config.num_hidden_layers, (stage_id + 1) * config.num_hidden_layers))
        if num_stages == 1:
            stage = GPTOneStage(
                config=config,
                device=device, 
                layer_ids=layer_ids,
                pretrained_model=pretrained_model,
                timing_info=timing_info,
            )
            # stage.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            # Set pad_token_id to eos_token_id because Llama / GPT does not have a PAD token
            stage.generation_config.pad_token_id = stage.generation_config.eos_token_id
        else:
            if stage_id == num_stages - 1:
                # Ending stage
                stage = end_stage_class(
                    config=config,
                    device=device, 
                    layer_ids=layer_ids,
                    pretrained_model=pretrained_model,
                    timing_info=timing_info,
                )
                # stage.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                # Set pad_token_id to eos_token_id because Llama / GPT does not have a PAD token
                stage.generation_config.pad_token_id = stage.generation_config.eos_token_id
            elif stage_id == 0:
                # Starting stage
                stage = start_stage_class(
                    config=config,
                    device=device, 
                    layer_ids=layer_ids,
                    pretrained_model=pretrained_model,
                    timing_info=timing_info,
                )
                # stage.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                # Intermediate stage
                stage = intermediate_stage_class(
                    config=config,
                    device=device, 
                    layer_ids=layer_ids,
                    pretrained_model=pretrained_model,
                    timing_info=timing_info,
                )
                # stage.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            
        print(f"Put stage {stage.__class__.__name__} ({stage.num_parameters()} parameters) on device {device}")
        totoal_params += stage.num_parameters()
        pipeline_stages.append(stage)
            
    return pipeline_stages


def stages_forward(
    stages: List[Union[GPTEndingStage, LlamaEndingStage]], 
    inputs: Dict[str, Union[torch.Tensor, Any]],
    verbose: bool = False,
):
    labels = inputs.get('labels', None)
    labels = labels.to(stages[-1]._device) if labels is not None else None
    for i, stage in enumerate(stages):
        # print(f'current stage {i}: {stage}')
        if verbose:
            print(f"Forward pass for stage {i} on device {stage._device}")
        if i == 0:
            batch_inputs = _prepare_inputs(inputs, device=stage._device)
            # print(f"inputs: {batch_inputs['input_ids'].shape} (min: {batch_inputs['input_ids'].min()}, max: {batch_inputs['input_ids'].max()})")
            outputs = stage(**batch_inputs)
        else:
            batch_inputs = _prepare_inputs(outputs, device=stage._device) 
            if "llama" in stage.__class__.__name__.lower():
                outputs = stage(
                    hidden_states=batch_inputs[0],
                    past_key_values=batch_inputs[1],
                    all_hidden_states=batch_inputs[2],
                    all_self_attns=batch_inputs[3],
                    position_ids=batch_inputs[4],
                    attention_mask=batch_inputs[5],
                    labels=labels,
                )
            elif "gpt" in stage.__class__.__name__.lower():
                outputs = stage(
                    hidden_states=batch_inputs[0],
                    attention_mask=batch_inputs[1],
                    head_mask=batch_inputs[2],
                    encoder_hidden_states=batch_inputs[3],
                    encoder_attention_mask=batch_inputs[4],
                    all_hidden_states=batch_inputs[5],
                    all_self_attentions=batch_inputs[6],
                    all_cross_attentions=batch_inputs[7],
                    output_shape=batch_inputs[8],
                    labels=labels,
                )
            else:
                raise ValueError(f"Unknown model type: {stage.__class__.__name__}")
        
    return outputs


def _prepare_decoding_inputs(
    inputs: Dict[str, Union[torch.Tensor, Any]],
):
    # Create a new dictionary to avoid modifying the original inputs
    new_inputs = {key: val.clone() if torch.is_tensor(val) else val for key, val in inputs.items()}
    labels = new_inputs.pop("labels", None)
    if labels is not None:
        labels_attention_mask = torch.ones_like(labels)
        new_inputs['input_ids'] = torch.cat((inputs['input_ids'], labels), dim=1)
        new_inputs['attention_mask'] = torch.cat((inputs['attention_mask'], labels_attention_mask), dim=1)
        new_labels = torch.cat(
            (-100 * torch.ones_like(inputs['input_ids']), labels), dim=1
        )
        new_inputs['labels'] = new_labels
    return new_inputs


def stages_decoding(
    stages: List[Union[GPTEndingStage, LlamaEndingStage]], 
    inputs: Dict[str, Union[torch.Tensor, Any]],
    verbose: bool = False,
):
    # For CLM, we need to mask the sentence (-100) and add to the labels
    new_inputs = _prepare_decoding_inputs(inputs)
    return stages_forward(stages, new_inputs, verbose=verbose)



if __name__ == '__main__':
    import time
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForSeq2Seq 
    
    datasets = load_dataset('data/Anthropic')
    access_token = "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"
    model_name_or_path = "meta-llama/Llama-2-7b-chat-hf"
    # model_name_or_path = "microsoft/DialoGPT-small"
    config = AutoConfig.from_pretrained(model_name_or_path, token=access_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        config=config, 
        token=access_token,
        # device_map='auto',
        # torch_dtype="auto"
    )
    
    num_stages = 8
    timing_info = defaultdict(list)
    stages = get_stages(
        config, 
        access_token,
        model_name_or_path,
        num_stages, 
        timing_info=timing_info,
    )
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['query'], 
            padding=False, 
            truncation=True,
        )
        labels = tokenizer(
            examples['reference'], 
            padding=False, 
            truncation=True,
            
        )
        tokenized_inputs['labels'] = labels['input_ids']
        return tokenized_inputs
    
    print("Dataset example (first 2 examples):", datasets['test'][:2])
    train_dataset = datasets['test'].map(
        tokenize_and_align_labels,
        batched=True,
    ).remove_columns(datasets['test'].column_names)
    print("Dataset tokenized example (first 2 examples):", train_dataset[:2])
    print(f" ** pretrain name {model_name_or_path} tokenizer size: {len(tokenizer)} tokenized dataset IDs range: ({min(min(train_dataset['input_ids']))}, {max(max(train_dataset['input_ids']))}) ** ")
    
    label_pad_token_id = tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        # model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
    )
    inputs = next(iter(train_dataloader))
    
    # # Test Causal Language Modeling (CLM)
    # start = time.time()
    # outputs = model(**inputs)
    # loss = outputs.loss
    # logits = outputs.logits
    # end = time.time()
    # print(f'loss: {loss}, logits: {logits.shape}')
    # print(f'forward overhead {end - start}')

    # start = time.time()
    # outputs = stages_forward(stages, inputs)
    # loss = outputs[0]
    # logits = outputs[1]
    # end = time.time()
    # print(f'loss: {loss}, logits: {logits.shape}')
    # print(f'forward overhead {end - start}')
    # loss.backward()
    # timing_info['0_end'].append((time.time(), 'backward'))
    # print(timing_info)
    
    # Test decoding nll loss
    # start = time.time()
    # # loss = compute_nll_loss(model, inputs, labels)
    # new_inputs = _prepare_decoding_inputs(inputs)
    # print({k: v.shape for k, v in new_inputs.items()})
    # outputs = model(**new_inputs)
    # loss = outputs.loss
    # end = time.time()
    # print(f'nll loss: {loss}, overhead: {end - start}')
    
    start = time.time()
    outputs = stages_decoding(stages, inputs)
    loss = outputs[0]
    end = time.time()
    print(f'nll loss: {loss}, overhead: {end - start}')
    loss.backward()
    timing_info['0_end'].append((time.time(), 'backward'))
    print(timing_info)