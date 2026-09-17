"""Export chatterbox-tts's T3 GPT2 backbone into a standalone HF-loadable
directory so vLLM's LLM(model=<dir>, ...) can load it directly, for
triton_model_repo/chatterbox_tts/1/chatterbox_worker_vllm.py's
CHATTERBOX_BACKEND=vllm path.

RECONSTRUCTED, NOT VERIFIED AGAINST A REAL RUN: this is genuinely missing
from the repo. chatterbox_worker_vllm.py's docstring says "see deploy
notes" for how vllm_t3_model_dir gets built, but no such notes exist here
-- docs/tts-replacement-investigation.md, precompute_prompt.py, and
chatterbox_profiling/ are all referenced by comments elsewhere in this
codebase and none of them exist in this repo either. This script was
written purely by reading vllm_t3/chatterbox_t3_model.py's load_weights()
(which tells us exactly what input key names it expects: tfmr.*,
text_emb.*, speech_emb.*, speech_head.*, cond_enc.spkr_enc.*) and
vllm_t3/configuration_chatterbox.py's ChatterboxTurboConfig (which ships
full working defaults matching GPT2-medium -- no dimensions had to be
guessed). Treat the first real run of this script, and the first real
`CHATTERBOX_BACKEND=vllm` load against its output, as the actual
verification -- same posture this repo's own REBUILD.md and docker/
README.md take toward their own unverified steps.

What this assumes, and why:
  - model.t3 (ChatterboxTurboTTS.from_pretrained()'s T3 submodule) has
    tfmr/text_emb/speech_emb/speech_head/cond_enc as direct attributes,
    so state_dict() naturally produces keys with those exact prefixes --
    inferred from chatterbox_worker_vllm.py's own `del model.t3.tfmr`
    line and chatterbox_t3_model.py's WeightsMapper choosing those exact
    prefixes as its *input* side (the mapper wouldn't make sense against
    any other naming).
  - ChatterboxTurboConfig()'s defaults (hidden_size=1024, num_hidden_
    layers=24, num_attention_heads=16, intermediate_size=4096, ...) match
    the actual checkpoint architecture -- both chatterbox_t3_model.py
    (`if not isinstance(config, ChatterboxTurboConfig): config =
    ChatterboxTurboConfig()`, a bare fallback with no override) and this
    config class's own docstring ("GPT-2-medium backbone") support this,
    but it's still an inference, not a confirmed fact about the specific
    checkpoint chatterbox-tts==0.1.7 ships.

Usage (inside an environment with chatterbox-tts installed, e.g.
/venv/chatterbox or a fresh venv with chatterbox-tts's own deps --
crucially NOT /venv/vllm itself, since the point is producing vllm's
input, and chatterbox-tts's own torch pin would conflict with vllm's if
run there; use --no-deps chatterbox-tts install elsewhere, or run this
from the existing /venv/chatterbox which already has full chatterbox-tts
+ its torch):

    /venv/chatterbox/bin/python3 scripts/export_chatterbox_t3_for_vllm.py \\
        --output-dir /workspace/speech-cascade-inference/vllm_t3_model_dir
"""

import argparse
import json

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/workspace/speech-cascade-inference/vllm_t3_model_dir",
        help="Directory to write config.json + model.safetensors into.",
    )
    args = parser.parse_args()

    from chatterbox.tts_turbo import ChatterboxTurboTTS

    print("Loading ChatterboxTurboTTS (this downloads/loads the full model, "
          "including S3Gen and the reference-voice conditioning machinery "
          "we don't need here -- there's no lighter-weight load path in "
          "the package's public API)...")
    model = ChatterboxTurboTTS.from_pretrained(device="cpu")

    t3_state_dict = model.t3.state_dict()
    print(f"Extracted model.t3.state_dict(): {len(t3_state_dict)} tensors")

    # safetensors requires contiguous, non-shared-storage tensors.
    export_state_dict = {k: v.contiguous().clone() for k, v in t3_state_dict.items()}

    import os

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(export_state_dict, weights_path)
    print(f"Wrote {weights_path}")

    # Defaults straight from vllm_t3/configuration_chatterbox.py's
    # ChatterboxTurboConfig -- NOT re-derived from the loaded model here,
    # since T3's plain nn.Module doesn't carry a HF-style config object to
    # read them back off of. If a future chatterbox-tts release changes
    # these dimensions, this config.json needs updating by hand to match
    # (the safetensors shapes below would silently mismatch otherwise).
    config = {
        "model_type": "chatterbox_turbo",
        "architectures": ["ChatterboxTurboT3ForGeneration"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "vocab_size": 50276,
        "speech_vocab_size": 6563,
        "max_position_embeddings": 8196,
        "start_speech_token": 6561,
        "stop_speech_token": 6562,
        "speaker_embed_size": 256,
        "speech_cond_prompt_len": 375,
        "use_perceiver_resampler": False,
        "emotion_adv": False,
        "s3gen_sample_rate": 24000,
        "s3_token_rate": 25,
        "torch_dtype": "float32",
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {config_path}")

    print(
        "\nDone. UNVERIFIED past this point -- next step is a real "
        "CHATTERBOX_BACKEND=vllm load against this directory (which needs "
        "/venv/vllm built first, see docker/triton/Dockerfile) and "
        "checking chatterbox_t3_model.py's load_weights() actually maps "
        "every one of these tensors (it silently skips anything it "
        "doesn't recognize -- a real run's loaded_params return value, or "
        "a manual diff against params_dict, is the only way to confirm "
        "nothing was silently dropped)."
    )


if __name__ == "__main__":
    main()
