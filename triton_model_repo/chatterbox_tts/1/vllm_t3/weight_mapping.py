"""Checkpoint-key mapping for the vLLM T3 model, kept free of vLLM imports.

Split out of chatterbox_t3_model.py purely so it can be tested: that module
imports vllm at module scope, which needs a GPU-capable install, so the
mapping could otherwise only be exercised by actually standing up the vLLM
backend. Given that a wrong mapping here produces a model that loads, runs,
and emits confident nonsense -- rather than raising -- it is the last thing
that should be untestable.

The mapping has already been wrong twice:

  - the upstream PR this was adapted from mapped `cond_enc.speaker_proj.*`,
    while the real checkpoint has `cond_enc.spkr_enc.*`.
  - ParallelLMHead defaults to bias=False, so `speech_head.bias` (which the
    real checkpoint carries) had no parameter to land in.

Both were silent. Hence CHECKPOINT_TENSOR_NAMES below, taken from the actual
t3_turbo_v1.safetensors header in ResembleAI/chatterbox-turbo, and the test
in tests/unit/test_vllm_t3_weight_mapping.py that asserts every one of them
is either mapped or explicitly expected-unused.
"""

# tfmr.* -> vLLM's GPT2Model, registered under self.model; the rest keep their
# names except cond_enc, whose submodule is flattened into one attribute.
ORIG_TO_NEW_PREFIX = {
    "tfmr.": "model.",
    "text_emb.": "text_emb.",
    "speech_emb.": "speech_emb.",
    "speech_head.": "speech_head.",
    "cond_enc.spkr_enc.": "cond_enc_spkr_enc.",
}

# Checkpoint keys this model deliberately does not use. text_head predicts
# TEXT tokens; the TTS path only ever samples speech tokens, so there is no
# parameter to receive it. Anything else unmapped is a bug.
EXPECTED_UNUSED_PREFIXES = ("text_head.",)

# Parameters that legitimately receive no weight, in the other direction.
#
# vLLM's GPT2Model always constructs `wte`, GPT2's own token-embedding table,
# but T3 never uses it: text tokens go through text_emb and sampled speech
# tokens through speech_emb (see embed_input_ids), and prompts arrive as
# embeddings rather than ids. The reference implementation makes the same
# judgement more bluntly -- it DELETES t3.tfmr.wte -- which is why the
# checkpoint has 299 tensors but a t3 state_dict export has 298.
#
# Confirmed empirically rather than assumed: exporting and loading for real
# leaves exactly this one parameter unfilled and nothing else.
#
# This list is not a place to silence inconvenient failures. Every entry must
# be a parameter that is provably never read; anything else left random
# produces a model that runs and sounds wrong.
EXPECTED_UNFILLED_PARAMS = ("model.wte.weight",)

# HF GPT2 causal-mask buffers, not weights.
IGNORED_SUBSTRINGS = (".attn.bias", ".attn.masked_bias")

# Conv1D stores [in, out]; vLLM's linear layers want [out, in].
CONV1D_NAMES = ("c_attn", "c_proj", "c_fc")


def has_mapping(name: str) -> bool:
    """Whether any prefix rule matches this checkpoint key.

    Note this is NOT the same as `map_checkpoint_key(name) != name`: three of
    the rules map a prefix to itself (`text_emb.` -> `text_emb.`), so an
    identity result is a successful match, not a miss. Confusing the two is
    easy and gives a test that reports real keys as unmapped.
    """
    return any(name.startswith(p) for p in ORIG_TO_NEW_PREFIX)


def map_checkpoint_key(name: str) -> str:
    """Apply the prefix rewrite. Returns the name unchanged if nothing matches."""
    for old_prefix, new_prefix in ORIG_TO_NEW_PREFIX.items():
        if name.startswith(old_prefix):
            return new_prefix + name[len(old_prefix):]
    return name


def needs_transpose(mapped_name: str) -> bool:
    return mapped_name.endswith(".weight") and any(
        c in mapped_name for c in CONV1D_NAMES
    )


# Every tensor in ResembleAI/chatterbox-turbo's t3_turbo_v1.safetensors, read
# from its header (299 tensors). The 24 transformer blocks are collapsed to
# block 0 plus a count, since every block is structurally identical.
_TFMR_BLOCK_KEYS = [
    "tfmr.h.{i}.attn.c_attn.bias",
    "tfmr.h.{i}.attn.c_attn.weight",
    "tfmr.h.{i}.attn.c_proj.bias",
    "tfmr.h.{i}.attn.c_proj.weight",
    "tfmr.h.{i}.ln_1.bias",
    "tfmr.h.{i}.ln_1.weight",
    "tfmr.h.{i}.ln_2.bias",
    "tfmr.h.{i}.ln_2.weight",
    "tfmr.h.{i}.mlp.c_fc.bias",
    "tfmr.h.{i}.mlp.c_fc.weight",
    "tfmr.h.{i}.mlp.c_proj.bias",
    "tfmr.h.{i}.mlp.c_proj.weight",
]

NUM_HIDDEN_LAYERS = 24

CHECKPOINT_TENSOR_NAMES = (
    [k.format(i=i) for i in range(NUM_HIDDEN_LAYERS) for k in _TFMR_BLOCK_KEYS]
    + [
        "tfmr.ln_f.bias",
        "tfmr.ln_f.weight",
        "tfmr.wpe.weight",
        "tfmr.wte.weight",
        "cond_enc.spkr_enc.bias",
        "cond_enc.spkr_enc.weight",
        "speech_emb.weight",
        "speech_head.bias",
        "speech_head.weight",
        "text_emb.weight",
        "text_head.weight",
    ]
)

# Shapes of the non-tfmr tensors, from the same header. These are what pin the
# ChatterboxTurboConfig defaults to the real checkpoint rather than to a guess.
CHECKPOINT_SHAPES = {
    "cond_enc.spkr_enc.bias": [1024],
    "cond_enc.spkr_enc.weight": [1024, 256],
    "speech_emb.weight": [6563, 1024],
    "speech_head.bias": [6563],
    "speech_head.weight": [6563, 1024],
    "text_emb.weight": [50276, 1024],
    "text_head.weight": [50276, 1024],
    "tfmr.wpe.weight": [8196, 1024],
    "tfmr.wte.weight": [50276, 1024],
    "tfmr.h.0.attn.c_attn.weight": [1024, 3072],
    "tfmr.h.0.mlp.c_fc.weight": [1024, 4096],
}
