
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Dict, List, Any, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch import Tensor, LongTensor, FloatTensor
from transformers import (
    LlamaConfig, 
    LlamaForCausalLM,
    LlamaPreTrainedModel,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


@dataclass
class CustomizedLlamaOut(ModelOutput):
    hidden_states: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_self_attns: Optional[Tuple[torch.FloatTensor]] = None
    position_ids: Optional[torch.LongTensor] = None
    attention_mask: Optional[torch.Tensor] = None
    


class LlamaStartingStage(LlamaPreTrainedModel):
    def __init__(
        self, 
        config: LlamaConfig, 
        device: int, 
        layer_ids: List[int], 
        pretrained_model: LlamaForCausalLM,
        timing_info: dict = None,
    ):
        # Explicitly initialize LlamaModel with its expected arguments
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = pretrained_model.get_input_embeddings()
        
        self.layers = nn.ModuleList([
            pretrained_model.get_decoder().layers[layer_id] for layer_id in layer_ids
        ])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.gradient_checkpointing = False
        
        self._device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))
        
        self.to(device)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[Tensor], 
        grad_output: Tuple[Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self._device+1}_start" in timing_info:
            timing_info[f"{self._device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self._device}_start"].append((start_time, "backward"))
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> CustomizedLlamaOut:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        # return CustomizedLlamaOut(
        #     hidden_states=hidden_states,
        #     past_key_values=past_key_values,
        #     all_hidden_states=all_hidden_states,
        #     all_self_attns=all_self_attns,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        # )
        return (hidden_states, past_key_values, all_hidden_states, all_self_attns, position_ids, attention_mask)
        
    
    
class LlamaIntermediateStage(LlamaPreTrainedModel):
    def __init__(
        self, 
        config: LlamaConfig, 
        device: int, 
        layer_ids: List[int], 
        pretrained_model: LlamaForCausalLM,
        timing_info: dict = None,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.layers = nn.ModuleList([
            pretrained_model.get_decoder().layers[layer_id] for layer_id in layer_ids
        ])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        
        self._device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))
        self.to(device)

    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[Tensor], 
        grad_output: Tuple[Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self._device+1}_start" in timing_info:
            timing_info[f"{self._device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self._device}_start"].append((start_time, "backward"))
        
    def forward(
        self,
        hidden_states: FloatTensor,
        past_key_values: Optional[List[FloatTensor]] = None,
        all_hidden_states: Optional[Tuple[FloatTensor]] = None,
        all_self_attns: Optional[Tuple[FloatTensor]] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> CustomizedLlamaOut:
        
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        # return CustomizedLlamaOut(
        #     hidden_states=hidden_states,
        #     past_key_values=past_key_values,
        #     all_hidden_states=all_hidden_states,
        #     all_self_attns=all_self_attns,
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        # )
        return (hidden_states, past_key_values, all_hidden_states, all_self_attns, position_ids, attention_mask)
        
    

class LlamaEndingStage(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(
        self, 
        config: LlamaConfig, 
        device: int, 
        layer_ids: List[int], 
        pretrained_model: LlamaForCausalLM,
        timing_info: dict = None,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList([
            pretrained_model.get_decoder().layers[layer_id] for layer_id in layer_ids
        ])
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = pretrained_model.get_decoder().norm
        self.lm_head = pretrained_model.get_output_embeddings()
        self.gradient_checkpointing = False
        
        self._device = device
        self.register_full_backward_hook(partial(self.backward_hook, timing_info=timing_info))
        self.to(device)
        
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    # Add a method to profile the backward pass
    def backward_hook(
        self, 
        module: nn.Module, 
        grad_input: Tuple[Tensor], 
        grad_output: Tuple[Tensor], 
        timing_info: dict = None,
    ):
        # print(f"Backward pass started for {module}")
        start_time = time.time()
        if f"{self._device+1}_start" in timing_info:
            timing_info[f"{self._device+1}_end"].append((start_time, "backward"))
        timing_info[f"{self._device}_start"].append((start_time, "backward"))
        
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        inputs_embeds=None, 
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
        
    def forward(
        self,
        hidden_states: FloatTensor,
        past_key_values: Optional[List[FloatTensor]] = None,
        all_hidden_states: Optional[Tuple[FloatTensor]] = None,
        all_self_attns: Optional[Tuple[FloatTensor]] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        next_decoder_cache = None
        
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=next_decoder_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return (loss, logits, next_decoder_cache, all_hidden_states, all_self_attns)

    