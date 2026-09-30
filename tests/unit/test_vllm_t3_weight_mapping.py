"""Unit tests for the vLLM T3 checkpoint-key mapping.

No vLLM, no GPU, no model download -- weight_mapping.py is deliberately free
of vLLM imports so this can run in CI. That matters more here than usual: a
wrong mapping in this model does not raise. `load_weights()` skips any key it
cannot place, so an unmapped prefix leaves a module randomly initialized and
the model then loads, runs, and emits confident nonsense.

That has already happened twice:

  - the upstream PR mapped `cond_enc.speaker_proj.*`; the real checkpoint has
    `cond_enc.spkr_enc.*`.
  - `ParallelLMHead` defaults to `bias=False`, so `speech_head.bias` -- which
    the real checkpoint carries -- had no parameter to land in.

Both silent. These tests assert against CHECKPOINT_TENSOR_NAMES, taken from
the actual t3_turbo_v1.safetensors header in ResembleAI/chatterbox-turbo.
"""

import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "triton_model_repo", "chatterbox_tts", "1", "vllm_t3",
    ),
)

import pytest  # noqa: E402

from weight_mapping import (  # noqa: E402
    CHECKPOINT_SHAPES,
    CHECKPOINT_TENSOR_NAMES,
    EXPECTED_UNUSED_PREFIXES,
    NUM_HIDDEN_LAYERS,
    ORIG_TO_NEW_PREFIX,
    has_mapping,
    map_checkpoint_key,
    needs_transpose,
)


def test_checkpoint_has_the_expected_tensor_count():
    """299 tensors: 24 blocks x 12, plus ln_f x2, wpe, wte, and 7 non-tfmr."""
    assert len(CHECKPOINT_TENSOR_NAMES) == 299
    assert len(CHECKPOINT_TENSOR_NAMES) == NUM_HIDDEN_LAYERS * 12 + 4 + 7


def test_every_checkpoint_key_is_mapped_or_explicitly_unused():
    """The core guarantee: nothing gets silently dropped."""
    unmapped = [
        name
        for name in CHECKPOINT_TENSOR_NAMES
        if not has_mapping(name)
        and not name.startswith(EXPECTED_UNUSED_PREFIXES)
    ]
    assert not unmapped, f"these would be silently skipped by load_weights(): {unmapped}"


def test_cond_enc_uses_the_real_key_not_the_upstream_prs():
    """Regression: the PR's `speaker_proj` left this module random."""
    assert "cond_enc.spkr_enc." in ORIG_TO_NEW_PREFIX
    assert "cond_enc.speaker_proj." not in ORIG_TO_NEW_PREFIX
    assert map_checkpoint_key("cond_enc.spkr_enc.weight") == "cond_enc_spkr_enc.weight"


def test_speech_head_bias_exists_in_the_checkpoint_and_is_mapped():
    """Regression: ParallelLMHead(bias=False) had nowhere to put this."""
    assert "speech_head.bias" in CHECKPOINT_TENSOR_NAMES
    assert CHECKPOINT_SHAPES["speech_head.bias"] == [6563]
    assert map_checkpoint_key("speech_head.bias") == "speech_head.bias"


def test_text_head_is_the_only_expected_unused_key():
    """If this list ever grows, it should be a deliberate, argued change."""
    assert EXPECTED_UNUSED_PREFIXES == ("text_head.",)
    unused = [n for n in CHECKPOINT_TENSOR_NAMES if n.startswith(EXPECTED_UNUSED_PREFIXES)]
    assert unused == ["text_head.weight"]


def test_transformer_weights_land_under_the_vllm_gpt2_prefix():
    assert map_checkpoint_key("tfmr.h.0.attn.c_attn.weight") == "model.h.0.attn.c_attn.weight"
    assert map_checkpoint_key("tfmr.wte.weight") == "model.wte.weight"
    assert map_checkpoint_key("tfmr.ln_f.bias") == "model.ln_f.bias"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("model.h.0.attn.c_attn.weight", True),
        ("model.h.0.attn.c_proj.weight", True),
        ("model.h.0.mlp.c_fc.weight", True),
        ("model.h.0.mlp.c_proj.weight", True),
        ("model.h.0.attn.c_attn.bias", False),   # biases are 1-D, never transposed
        ("model.ln_f.weight", False),
        ("speech_emb.weight", False),
    ],
)
def test_only_conv1d_weights_are_transposed(name, expected):
    """GPT2's Conv1D stores [in, out]; vLLM's linears want [out, in]."""
    assert needs_transpose(name) is expected


def test_conv1d_shapes_confirm_the_transpose_is_needed():
    """c_attn is [hidden, 3*hidden] in the checkpoint -- i.e. [in, out]."""
    assert CHECKPOINT_SHAPES["tfmr.h.0.attn.c_attn.weight"] == [1024, 3072]
    assert CHECKPOINT_SHAPES["tfmr.h.0.mlp.c_fc.weight"] == [1024, 4096]


def test_checkpoint_shapes_pin_the_config_defaults():
    """ChatterboxTurboConfig's defaults were inferred; these make them facts.

    scripts/export_chatterbox_t3_for_vllm.py writes a config.json from those
    defaults, and a mismatch there would silently misshape the model.
    """
    hidden = 1024
    assert CHECKPOINT_SHAPES["speech_emb.weight"] == [6563, hidden]   # speech_vocab_size
    assert CHECKPOINT_SHAPES["text_emb.weight"] == [50276, hidden]    # vocab_size
    assert CHECKPOINT_SHAPES["tfmr.wpe.weight"] == [8196, hidden]     # max_position_embeddings
    assert CHECKPOINT_SHAPES["cond_enc.spkr_enc.weight"] == [hidden, 256]  # speaker_embed_size
    assert CHECKPOINT_SHAPES["tfmr.h.0.mlp.c_fc.weight"][1] == 4096   # intermediate_size
    assert CHECKPOINT_SHAPES["tfmr.h.0.attn.c_attn.weight"][1] == 3 * hidden
    assert NUM_HIDDEN_LAYERS == 24


def test_only_gpt2s_unused_token_embedding_may_go_unfilled():
    """wte is the one parameter allowed to receive no weight.

    vLLM's GPT2Model always builds `wte`, but T3 never reads it: text goes
    through text_emb, sampled speech tokens through speech_emb, and prompts
    arrive as embeddings. The reference implementation deletes t3.tfmr.wte
    outright, which is why a t3 state_dict export has 298 tensors against the
    checkpoint's 299.

    Confirmed by an actual export+load, not by reading code: exactly this one
    parameter came back unfilled. Widening this list is how a real
    randomly-initialized module would get waved through, so it should not grow
    without the same kind of evidence.
    """
    from weight_mapping import EXPECTED_UNFILLED_PARAMS

    assert EXPECTED_UNFILLED_PARAMS == ("model.wte.weight",)


def test_the_export_is_expected_to_be_one_tensor_short():
    """299 in the checkpoint, 298 exported -- the missing one is wte."""
    exported_expected = [
        n for n in CHECKPOINT_TENSOR_NAMES if n != "tfmr.wte.weight"
    ]
    assert len(CHECKPOINT_TENSOR_NAMES) == 299
    assert len(exported_expected) == 298
