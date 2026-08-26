"""Reconstructs the NVFP4 post-training quantization step for
Llama-3.1-Nemotron-Nano-4B-v1.1 -- the checkpoint `nemotron_llm` currently
serves (`triton_model_repo/nemotron_llm/config.pbtxt`'s `engine_dir`, loaded
directly by TensorRT-LLM's `LLM` API, no `trtllm-build` step -- see
`scripts/quantize_fp8.py` + `scripts/build_engine.sh` for the older, AOT-
compiled FP8 path this replaced).

WHY THIS SCRIPT DIFFERS IN SHAPE FROM quantize_fp8.py
-------------------------------------------------------
quantize_fp8.py calls `modelopt.torch.quantization`/`modelopt.torch.export`
directly in-process -- a small, self-contained calibration loop. The NVFP4
checkpoint was NOT produced that way: per
`speech-cascade-inference/reports/session-report.md` ("1. Quantization") and
`deploy/REBUILD.md` ("4b"), it was produced by running NVIDIA's own
`examples/hf_ptq/hf_ptq.py` CLI script out of a checkout of the
`NVIDIA/TensorRT-Model-Optimizer` GitHub repo -- not the bare Python API.
This script mirrors that shape instead: clone/checkout the pinned repo,
install its (trimmed) example requirements, then invoke `hf_ptq.py` as a
subprocess with the documented flags. That is the most faithful "capture the
interactive run as code" available here, short of reimplementing internals
of a 1800-line upstream script by hand.

PROVENANCE -- what's verified vs. reconstructed
------------------------------------------------
The session report is explicit that "Exact hf_ptq.py flags were run
interactively and NOT preserved as a script" (REBUILD.md, "4b") -- so no
transcript of the literal command line survives. Everything below was
reconstructed from the prose in session-report.md + REBUILD.md, and then
CROSS-CHECKED against the actual `hf_ptq.py` source, argparse definitions,
and dataset registry at the pinned tag, fetched directly from
github.com/NVIDIA/TensorRT-Model-Optimizer during this reconstruction (via
`gh api repos/NVIDIA/TensorRT-Model-Optimizer/contents/...?ref=0.46.0`).
That verification is stronger than "best guess" but still short of "this is
exactly what was run" -- treat it as a high-confidence reconstruction, not a
transcript:

  - Tag: session-report.md/REBUILD.md say "v0.46.0"; the actual git tag on
    NVIDIA's repo is `0.46.0` (no "v" prefix) -- confirmed via `gh api
    .../tags`. `v0.46.0` does not exist. Corrected here.
  - `examples/hf_ptq/hf_ptq.py` (REBUILD.md separately says
    `examples/llm_ptq/hf_ptq.py`) -- confirmed BOTH paths are correct:
    `examples/llm_ptq` is a symlink to `examples/hf_ptq` in this repo at
    this tag. No actual discrepancy.
  - `--qformat nvfp4`, `--kv_cache_qformat none`: confirmed as literal,
    valid flag values against `hf_ptq.py`'s own `parse_args()` (default
    `--qformat` is `fp8`, default `--kv_cache_qformat` is `fp8_cast` --
    both had to be passed explicitly, matching "KV cache left unquantized"
    and "every linear layer... quantized" in the session report) and against
    `modelopt_recipes/configs/ptq/presets/model/nvfp4.yaml` (exists,
    `algorithm: max` -- plain PTQ, not AWQ-calibrated, consistent with the
    report never mentioning an AWQ/SmoothQuant step) and
    `.../presets/kv/` (KV_CACHE_NONE == "none" is a hardcoded sentinel, not
    a preset file).
  - `--dataset cnn_nemotron_v2_mix --calib_size 512`: the report says "512
    samples from cnn_dailymail + nvidia/Nemotron-Post-Training-Dataset-v2".
    `modelopt/torch/utils/dataset_utils.py` defines exactly this combo --
    `DATASET_COMBOS["cnn_nemotron_v2_mix"] = ["cnn_dailymail",
    "nemotron-post-training-dataset-v2"]`, the latter mapping to HF dataset
    `nvidia/Nemotron-Post-Training-Dataset-v2` -- and `hf_ptq.py` itself
    defaults to exactly this combo with the comment "Defaulting to the
    'cnn_nemotron_v2_mix' combo (cnn_dailymail + nemotron-post-training-
    dataset-v2)" in its speculative-decoding branch, i.e. this is a
    first-class, NVIDIA-recognized combo name, not a guess. The 512/2=256
    split confirmed exact (get_dataset_dataloader does
    `divmod(n, len(members))`, evenly splits 512 with no remainder).
    NOT verified: that `cnn_nemotron_v2_mix` (rather than the two dataset
    names passed separately) is literally the string that was typed, vs.
    two `--dataset` values with `--calib_size 256,256`. Both produce an
    identical calibration set per the code above, so it doesn't change the
    output either way.
  - Requires an `HF_TOKEN` (env var, read automatically by `datasets`/`huggingface_hub`)
    from an account that has accepted `nvidia/Nemotron-Post-Training-Dataset-v2`'s
    gated terms -- stated directly in the session report.
  - Export format: no `--export_fmt` is passed (that flag is documented
    upstream as deprecated). `hf_ptq.py`'s default (`export_fmt="hf"`)
    produces a safetensors + config.json HF-shaped checkpoint carrying
    `quant_method: "modelopt"` metadata -- matches the report's own
    description ("current transformers doesn't recognize the exported
    quant_method") and matches how `nemotron_llm/1/model.py` loads it
    (`tensorrt_llm.llmapi.llm._TrtLLM(model=engine_dir, ...)` on the
    checkpoint dir directly, no separate `trtllm-build` step).
  - flash-attn is deliberately excluded from the pip install: the upstream
    `examples/hf_ptq/requirements.txt` lists `flash-attn>=2.6.0` by default;
    the session report says it "was deliberately stripped from the example
    script's install requirements -- a source build for this torch/CUDA/
    Python combination risked 30-60 minutes and only affects calibration
    speed, not correctness." Mirrored below by filtering it out before
    `pip install -r requirements.txt`.

NOT verified / genuinely unrecorded (left at upstream defaults, not guessed):
  - `--calib_seq` (max calibration sequence length) -- not mentioned in
    either source document. Left unset (upstream default: 512).
  - `--batch_size` -- not mentioned. Left unset (upstream default: 0, i.e.
    auto-computed).
  - `--trust_remote_code` -- not mentioned either way. Left off (upstream
    default: False). If loading `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1`
    requires it, hf_ptq.py will say so; pass --trust-remote-code here to add it.
  - Whether GPU memory / device_map flags (`--use_seq_device_map`,
    `--gpu_max_mem_percentage`) were needed. This model is small (~8.5GB
    BF16) and the report's peak-memory number (13.69GB during calibration)
    fits on this instance's single 16GB GPU without special handling, so
    none are passed by default.

SAFETY: this script's default --export-dir is a scratch path, NOT
`speech-cascade-inference/models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4` --
the directory `nemotron_llm/config.pbtxt` actually points Triton at. Running
this script with defaults cannot clobber the live checkpoint. Pass
--export-dir explicitly (and reload the Triton model yourself, deliberately)
if you actually intend to replace what's deployed.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

MODELOPT_REPO_URL = "https://github.com/NVIDIA/TensorRT-Model-Optimizer"
MODELOPT_TAG = "0.46.0"  # NOT "v0.46.0" -- see docstring "PROVENANCE"
MODELOPT_CLONE_DIR = "/tmp/modelopt"  # matches the path used in the recorded session

SOURCE_CHECKPOINT = "/workspace/speech-cascade-inference/models/Llama-3.1-Nemotron-Nano-4B-v1.1"
# Deliberately NOT the live-served path -- see docstring "SAFETY".
EXPORT_DIR = "/workspace/nvfp4-quantize-scratch/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4"

QFORMAT = "nvfp4"  # the "full" variant actually deployed; "nvfp4_mlp_only" is the other
# benchmarked-but-not-shipped variant (session report: full NVFP4 outscored
# MLP-only on both engines tested, the opposite of NVIDIA's general guidance).
KV_CACHE_QFORMAT = "none"
DATASET = "cnn_nemotron_v2_mix"
CALIB_SIZE = 512


def run(cmd, **kwargs):
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, **kwargs)


def ensure_modelopt_checkout(clone_dir: str) -> str:
    """Clone TensorRT-Model-Optimizer at the pinned tag if not already present.

    Sparse-checkout is NOT used here (unlike what the session report implies
    was safe for `examples/hf_ptq/` specifically) because `hf_ptq.py` imports
    sibling modules (`example_utils`, `cast_mxfp4_to_nvfp4`, ...) from the
    same `examples/hf_ptq/` directory but those in turn may reach into the
    installed `modelopt` package -- a full checkout avoids re-litigating that
    every run. Disk cost is modest (source checkout, not weights).
    """
    hf_ptq_path = os.path.join(clone_dir, "examples", "hf_ptq", "hf_ptq.py")
    if os.path.exists(hf_ptq_path):
        print(f"Reusing existing checkout at {clone_dir}", flush=True)
        return hf_ptq_path

    if os.path.exists(clone_dir):
        raise RuntimeError(
            f"{clone_dir} exists but doesn't look like a TensorRT-Model-Optimizer "
            f"checkout (missing {hf_ptq_path}). Remove it or pass --modelopt-dir "
            "pointing somewhere else."
        )

    run(["git", "clone", MODELOPT_REPO_URL, clone_dir])
    run(["git", "checkout", MODELOPT_TAG], cwd=clone_dir)
    return hf_ptq_path


def install_deps(modelopt_dir: str, modelopt_version: str):
    """pip install nvidia-modelopt[hf]==<pinned> plus the example script's own
    requirements.txt, with flash-attn filtered out (see docstring)."""
    run([sys.executable, "-m", "pip", "install", f"nvidia-modelopt[hf]=={modelopt_version}"])

    req_path = os.path.join(modelopt_dir, "examples", "hf_ptq", "requirements.txt")
    with open(req_path) as f:
        reqs = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("flash-attn")
        ]
    if reqs:
        run([sys.executable, "-m", "pip", "install", *reqs])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-checkpoint",
        default=SOURCE_CHECKPOINT,
        help="HF BF16 checkpoint to quantize (default: %(default)s)",
    )
    parser.add_argument(
        "--export-dir",
        default=EXPORT_DIR,
        help=(
            "Output dir for the quantized checkpoint. Defaults to a scratch path -- "
            "NOT the live-served models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4 (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--qformat",
        default=QFORMAT,
        help="modelopt --qformat value; 'nvfp4' (full, deployed) or 'nvfp4_mlp_only' (default: %(default)s)",
    )
    parser.add_argument("--kv-cache-qformat", default=KV_CACHE_QFORMAT, help="modelopt --kv_cache_qformat value (default: %(default)s)")
    parser.add_argument("--dataset", default=DATASET, help="modelopt --dataset value (default: %(default)s)")
    parser.add_argument("--calib-size", type=int, default=CALIB_SIZE, help="modelopt --calib_size value (default: %(default)s)")
    parser.add_argument(
        "--modelopt-dir",
        default=MODELOPT_CLONE_DIR,
        help="Where to clone/reuse the TensorRT-Model-Optimizer checkout (default: %(default)s)",
    )
    parser.add_argument(
        "--modelopt-version",
        default=MODELOPT_TAG,
        help="pip version pin for nvidia-modelopt[hf] AND the git tag to check out (default: %(default)s)",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass --trust_remote_code through to hf_ptq.py (not used in the recorded run; see docstring)")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install steps (assume deps already present)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="HF token with access to the gated Nemotron-Post-Training-Dataset-v2 (default: $HF_TOKEN)")
    args = parser.parse_args()

    if not args.hf_token:
        print(
            "WARNING: no --hf-token / $HF_TOKEN set. nvidia/Nemotron-Post-Training-Dataset-v2 "
            "is gated -- the calibration dataset download will fail without a token from an "
            "account that has accepted its terms.",
            file=sys.stderr,
        )

    if os.path.abspath(args.export_dir) == os.path.abspath(
        "/workspace/speech-cascade-inference/models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4"
    ):
        raise SystemExit(
            "Refusing to export directly onto the live-served checkpoint path. "
            "Quantize to a scratch --export-dir and swap it in deliberately."
        )

    t0 = time.time()
    hf_ptq_path = ensure_modelopt_checkout(args.modelopt_dir)
    print(f"Checkout ready in {time.time()-t0:.1f}s", flush=True)

    if not args.skip_install:
        t0 = time.time()
        install_deps(args.modelopt_dir, args.modelopt_version)
        print(f"Installed deps in {time.time()-t0:.1f}s", flush=True)

    os.makedirs(args.export_dir, exist_ok=True)

    cmd = [
        sys.executable,
        hf_ptq_path,
        "--pyt_ckpt_path", args.source_checkpoint,
        "--qformat", args.qformat,
        "--kv_cache_qformat", args.kv_cache_qformat,
        "--dataset", args.dataset,
        "--calib_size", str(args.calib_size),
        "--export_path", args.export_dir,
    ]
    if args.trust_remote_code:
        cmd.append("--trust_remote_code")

    env = dict(os.environ)
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token

    t0 = time.time()
    run(cmd, cwd=os.path.join(args.modelopt_dir, "examples", "hf_ptq"), env=env)
    print(f"Quantized + exported in {time.time()-t0:.1f}s to {args.export_dir}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
