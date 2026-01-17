# Copyright 2024 ZeroModel Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
DeepSeek-style MLA Model with LRU Integration.

This module provides a transformer model that uses Multi-head Latent Attention
(MLA) with optional Latent Reasoning Unit (LRU) for iterative refinement.

Can be used as a drop-in replacement for Qwen2 models by converting
attention layers to MLA.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache

from .config import MLAConfig, LRUConfig, DeepSeekMLAConfig
from .mla_attention import MLAAttention
from .lru import LatentReasoningUnit, SimpleLRU, LRUOutput


class DeepSeekMLAPretrainedConfig(PretrainedConfig):
    """Configuration class for DeepSeek MLA models."""

    model_type = "deepseek_mla"

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        intermediate_size: int = 4864,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        attention_dropout: float = 0.0,
        # MLA specific
        kv_latent_dim: int = 256,
        q_latent_dim: int = 512,
        rope_head_dim: int = 64,
        # LRU specific
        use_lru: bool = True,
        lru_max_iterations: int = 8,
        lru_halt_threshold: float = 0.99,
        lru_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout

        # MLA config
        self.kv_latent_dim = kv_latent_dim
        self.q_latent_dim = q_latent_dim
        self.rope_head_dim = rope_head_dim

        # LRU config
        self.use_lru = use_lru
        self.lru_max_iterations = lru_max_iterations
        self.lru_halt_threshold = lru_halt_threshold
        self.lru_layers = lru_layers

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepSeekMLAMLP(nn.Module):
    """MLP module (same as standard transformer)."""

    def __init__(self, config: DeepSeekMLAPretrainedConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMLADecoderLayer(nn.Module):
    """Transformer decoder layer with MLA attention and optional LRU."""

    def __init__(
        self,
        config: DeepSeekMLAPretrainedConfig,
        layer_idx: int,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Create MLA config from model config
        mla_config = MLAConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            kv_latent_dim=config.kv_latent_dim,
            q_latent_dim=config.q_latent_dim,
            rope_head_dim=config.rope_head_dim,
            head_dim=config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            attention_dropout=config.attention_dropout,
        )

        # MLA attention
        self.self_attn = MLAAttention(mla_config, layer_idx=layer_idx)

        # Optional LRU
        self.lru = None
        if config.use_lru:
            should_use_lru = (
                config.lru_layers is None or layer_idx in config.lru_layers
            )
            if should_use_lru:
                lru_config = LRUConfig(
                    latent_dim=config.kv_latent_dim,
                    max_iterations=config.lru_max_iterations,
                    halt_threshold=config.lru_halt_threshold,
                )
                self.lru = LatentReasoningUnit(lru_config)
                self.self_attn.set_lru(self.lru)

        # MLP
        self.mlp = DeepSeekMLAMLP(config)

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention with MLA
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        # Extract LRU info if present
        lru_output = None
        if isinstance(attn_output, tuple):
            attn_output, lru_output = attn_output

        hidden_states = residual + attn_output

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if lru_output is not None:
            outputs += (lru_output,)

        return outputs


class DeepSeekMLAModel(PreTrainedModel):
    """DeepSeek MLA Transformer Model (base, without LM head)."""

    config_class = DeepSeekMLAPretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepSeekMLADecoderLayer"]

    def __init__(self, config: DeepSeekMLAPretrainedConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            DeepSeekMLADecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Position IDs
        if position_ids is None:
            past_length = past_key_values[0][0].shape[1] if past_key_values is not None else 0
            position_ids = torch.arange(
                past_length, seq_length + past_length,
                dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0)

        # Attention mask
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(attention_mask, (batch_size, seq_length))

        hidden_states = inputs_embeds

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        all_lru_outputs = []

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            # Collect LRU outputs if present
            if len(layer_outputs) > (3 if output_attentions else 2):
                lru_output = layer_outputs[-1]
                if isinstance(lru_output, LRUOutput):
                    all_lru_outputs.append(lru_output)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        # Attach LRU outputs for loss computation
        if all_lru_outputs:
            output.lru_outputs = all_lru_outputs

        return output

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Prepare the attention mask for attention computation."""
        batch_size, seq_length = input_shape

        # Expand mask to [B, 1, S, S] for attention
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min

        return extended_attention_mask


class DeepSeekMLAForCausalLM(PreTrainedModel):
    """DeepSeek MLA Model with a language modeling head."""

    config_class = DeepSeekMLAPretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: DeepSeekMLAPretrainedConfig):
        super().__init__(config)
        self.model = DeepSeekMLAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # LRU loss module for auxiliary losses (stability, sparsity, ponder)
        if config.use_lru:
            from verl.trainer.lru.losses import LRULossModule
            self.lru_loss_module = LRULossModule(
                stability_weight=getattr(config, 'lru_stability_weight', 0.1),
                sparsity_weight=getattr(config, 'lru_sparsity_weight', 0.01),
                ponder_weight=getattr(config, 'lru_ponder_weight', 0.001),
                max_iterations=config.lru_max_iterations,
            )
        else:
            self.lru_loss_module = None

        # Initialize weights
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Model forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        lru_loss_output = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            ce_loss = loss_fct(shift_logits, shift_labels)

            # FIXED: Integrate LRU auxiliary losses
            if self.lru_loss_module is not None and hasattr(outputs, 'lru_outputs') and outputs.lru_outputs:
                # Aggregate LRU losses from all layers
                total_lru_loss = torch.tensor(0.0, device=ce_loss.device, dtype=ce_loss.dtype)
                all_metrics = {}

                for layer_idx, lru_output in enumerate(outputs.lru_outputs):
                    layer_loss = self.lru_loss_module(lru_output, attention_mask=attention_mask)
                    total_lru_loss = total_lru_loss + layer_loss.total_loss

                    # Collect metrics from first layer for logging
                    if layer_idx == 0:
                        all_metrics = layer_loss.metrics

                # Average across layers
                num_lru_layers = len(outputs.lru_outputs)
                total_lru_loss = total_lru_loss / num_lru_layers

                loss = ce_loss + total_lru_loss
                lru_loss_output = {
                    'ce_loss': ce_loss,
                    'lru_loss': total_lru_loss,
                    'lru_metrics': all_metrics,
                }
            else:
                loss = ce_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        # Pass through LRU outputs and loss details for logging/analysis
        if hasattr(outputs, 'lru_outputs'):
            output.lru_outputs = outputs.lru_outputs
        if lru_loss_output is not None:
            output.lru_loss_details = lru_loss_output

        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            # Only use last token for generation
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

        return model_inputs
