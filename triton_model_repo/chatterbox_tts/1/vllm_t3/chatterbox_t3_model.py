"""Standalone vanilla-vLLM T3 model, adapted from the unmerged vllm-omni PR
(https://github.com/vllm-project/vllm-omni/pull/1517, lishunyang12/vllm-omni
branch chatterbox-turbo-tts), stripped of vllm-omni's multi-stage/streaming
plumbing (OmniOutput, preprocess/postprocess hooks, stage connectors) since
this is a T3-only profiling comparison, not a production serving pipeline.

Conditioning (t3_cond -> prompt_embeds) is precomputed separately using the
proven chatterbox package (see precompute_prompt.py) and fed in directly via
vLLM's native EmbedsPrompt -- so this model only needs to implement the GPT2
decode loop + speech-token sampling, not conditioning/tokenization.

Fix vs. the original PR (confirmed via our actual checkpoint's safetensors
keys): cond_enc weight is under `cond_enc.spkr_enc.*`, not
`cond_enc.speaker_proj.*` -- the PR's mapper would have silently left that
module randomly initialized. Not that it matters here (conditioning is
precomputed outside this model), but the mapper is fixed for correctness in
case cond_enc weights are ever loaded through this path too.
"""
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.gpt2 import GPT2Model
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from configuration_chatterbox import ChatterboxTurboConfig

SPEECH_VOCAB_SIZE = 6561  # excludes SOS/EOS
EOS_TOKEN = 6562


class ChatterboxTurboT3ForGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, ChatterboxTurboConfig):
            config = ChatterboxTurboConfig()
        self.config = config
        hidden_size = config.hidden_size

        from transformers import GPT2Config as HFGPT2Config
        gpt2_config = HFGPT2Config(
            vocab_size=config.vocab_size,
            n_embd=hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_inner=config.intermediate_size,
            n_positions=config.max_position_embeddings,
            add_cross_attention=False,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
        )
        orig_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = gpt2_config
        self.model = GPT2Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "tfmr"))
        vllm_config.model_config.hf_config = orig_hf_config

        self.text_emb = nn.Embedding(config.vocab_size, hidden_size)
        self.speech_emb = nn.Embedding(config.speech_vocab_size, hidden_size)
        self.speech_head = ParallelLMHead(config.speech_vocab_size, hidden_size)
        self.logits_processor = LogitsProcessor(config.speech_vocab_size)

        self.cond_enc_spkr_enc = nn.Linear(config.speaker_embed_size, hidden_size)

        speech_mask = torch.zeros((config.speech_vocab_size,), dtype=torch.bool)
        speech_mask[:SPEECH_VOCAB_SIZE] = True
        speech_mask[EOS_TOKEN] = True
        self.register_buffer("_speech_allowed_mask", speech_mask, persistent=False)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # Decode-step tokens are sampled SPEECH tokens -- must route through
        # speech_emb, NOT GPT2's own (unused/deleted in the reference impl) wte.
        return self.speech_emb(input_ids.clamp(0, self.config.speech_vocab_size - 1))

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.speech_head, hidden_states)
        if logits is None:
            return None
        return logits.masked_fill(~self._speech_allowed_mask, float("-inf"))

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "tfmr.": "model.",
            "text_emb.": "text_emb.",
            "speech_emb.": "speech_emb.",
            "speech_head.": "speech_head.",
            "cond_enc.spkr_enc.": "cond_enc_spkr_enc.",  # FIX: real checkpoint key, not speaker_proj
        }
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if ".attn.bias" in name or ".attn.masked_bias" in name:
                continue
            mapped_name = name
            for old_prefix, new_prefix in self.hf_to_vllm_mapper.orig_to_new_prefix.items():
                if name.startswith(old_prefix):
                    mapped_name = new_prefix + name[len(old_prefix):]
                    break
            if mapped_name not in params_dict:
                continue
            param = params_dict[mapped_name]
            for conv1d_name in ["c_attn", "c_proj", "c_fc"]:
                if conv1d_name in mapped_name and mapped_name.endswith(".weight"):
                    loaded_weight = loaded_weight.t()
                    break
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                param.data.copy_(loaded_weight)
            loaded_params.add(mapped_name)
        return loaded_params
