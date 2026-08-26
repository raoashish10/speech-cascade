# Rebuild runbook: fresh instance -> working deployment

This is the executable version of "if this box died right now, how would
someone rebuild it." It was written by walking the actually-running
instance (not from memory): the live `tritonserver` process's real command
line and environment, the real `/etc/portal.yaml`, the real supervisor
configs, and `speech-cascade-inference/reports/session-report.md` (the
detailed account of how the currently-deployed NVFP4 LLM checkpoint was
produced). Where a step is scripted and tested, this points at the script.
Where it isn't (see "Known gaps" at the bottom), it says so explicitly
instead of inventing flags nobody has actually verified.

This project deliberately runs bare-metal (no Docker -- this container
can't do Docker-in-Docker), which is *why* this runbook is long: none of
this is needed inside NVIDIA's NGC containers, which ship a matched
toolchain. See the main `README.md`'s "Environment quirks fixed to make
this run bare-metal" for the full list this runbook builds on.

## 0. Layout recap

Two directories, two different persistence stories:

- `speech-cascade` (this repo, git-tracked): `model.py`, `config.pbtxt`,
  `scripts/`, `tests/`, `deploy/`. Survives anything, as long as it's
  pushed.
- `speech-cascade-inference` (NOT git-tracked, plain directory): model
  weights, compiled TensorRT engines, the extracted Triton server binary.
  Backed up to S3 (`s3://ashish-s3-coding-bucket/speech-cascade-inference/`,
  mirrors the directory 1:1). **This instance's `/workspace` is NOT a
  persistent volume** -- everything outside git is gone on recycle/destroy.
  This has already happened once (see
  `speech-cascade-inference/reports/session-report.md`, "The runtime was
  gone").

## 1. Fresh instance: system packages

All of this is already-validated, copied verbatim from the main
`README.md`'s "Environment quirks" section (numbered to match):

```bash
# 1. MPI (TensorRT-LLM's executor uses MPI-based worker spawning even for
#    one GPU)
apt-get install -y openmpi-bin libopenmpi-dev

# 2. CUDA 13 runtime libraries (the tensorrt-llm wheel pulls CUDA 13
#    bindings but not the runtime libs) -- install a toolkit-only package,
#    never the `cuda` metapackage (see the top-level agent guide's CUDA
#    section: never let apt touch the driver)
apt-get install -y cuda-libraries-13-2 cuda-toolkit-13-2

# 5. Ubuntu 24.04 doesn't ship libssl.so.1.1, which the Triton server
#    binary links against. Extract just the .so files from the Ubuntu
#    20.04 libssl1.1 .deb -- do NOT install system-wide (avoids touching
#    the system's OpenSSL 3):
mkdir -p /workspace/speech-cascade-inference/triton_server/compat_libs
cd /workspace/speech-cascade-inference/triton_server/compat_libs
wget http://security.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
dpkg-deb -x libssl1.1_1.1.1f-1ubuntu2_amd64.deb extracted/

# 6. NVIDIA DCGM (Triton links against libdcgm.so.4 for GPU metrics --
#    this is a hard ELF dependency, not something --allow-metrics=false
#    routes around)
apt-get install -y datacenter-gpu-manager-4-cuda13
```

Steps 3, 4, 7, 8, 9, 10 in the main README are runtime/environment fixes
(PYTHONHOME, PATH, LD_LIBRARY_PATH, specific package choices), not one-time
package installs -- they're already baked into
`deploy/supervisor/speech-cascade-triton.sh` (see step 6 below).

## 2. Restore weights, engines, and the Triton binary from S3

```bash
aws configure   # needs credentials with access to the bucket below
aws s3 sync s3://ashish-s3-coding-bucket/speech-cascade-inference/ \
    /workspace/speech-cascade-inference/
```

This restores `models/` (all checkpoints and compiled engines),
`triton_server/extracted/` (the redistributable no-Docker Triton server
build) and `compat_libs/`, and `deprecated/`. It does NOT restore this git
repo (`speech-cascade`) -- clone that separately in the normal way.

## 3. Python environments

Two isolated venvs. Restore exact versions from this repo rather than
letting pip/uv re-resolve latest (that's the whole point of capturing
these):

```bash
# Main serving venv -- tensorrt_llm, torch, onnxruntime-gpu, kokoro-onnx.
# FRAGILE: numpy is pinned <2 (TensorRT-LLM's compiled bindings are built
# against numpy 1.x; several packages in this list silently upgrade numpy
# if installed carelessly -- see main README quirk #10). Installing from
# this frozen list preserves the pin; do not add new packages to this venv
# without re-checking `python -c "import numpy; print(numpy.__version__)"`
# afterward.
python3.12 -m venv /venv/main
/venv/main/bin/pip install -r deploy/requirements-main.txt

# Gateway venv -- fastapi/uvicorn/websockets/silero-vad/tritonclient,
# deliberately isolated from /venv/main so its deps can never touch the
# numpy pin above.
python3.12 -m venv /venv/gateway
/venv/gateway/bin/pip install -r deploy/requirements-gateway.txt
```

`deploy/requirements-main.txt` and `deploy/requirements-gateway.txt` are
`pip freeze` snapshots of the venvs actually running this deployment,
taken while writing this runbook -- exact, not "whatever resolves today".
Regenerate them after any intentional dependency change:

```bash
/venv/main/bin/python -m pip freeze > deploy/requirements-main.txt
uv pip freeze --python /venv/gateway/bin/python > deploy/requirements-gateway.txt
```

Getting TensorRT-LLM itself importable in `/venv/main` in the first place
(vs. just `pip install`-ing the frozen list, which assumes it already
works) is the single hardest part of this whole rebuild -- see
`speech-cascade-inference/reports/session-report.md`, section "4. Deploying
into speech-cascade-inference" for the exact three-way CPython ABI
mismatch that was hit and how it was diagnosed (`ldd`/`nm`/`readelf` on the
actual binaries) and fixed (consolidate into one venv, force
`LD_LIBRARY_PATH` to win over the stub's own `RUNPATH`). That fix is what
`deploy/supervisor/speech-cascade-triton.sh` encodes.

## 4. Compiled artifacts: what's scripted vs. what isn't

`models/`, once restored from S3 in step 2, already contains every
compiled engine this deployment needs -- **you do not need to rebuild
anything to bring the service back up.** Rebuild only if a checkpoint or
engine is missing, corrupted, or being intentionally regenerated (new GPU
architecture, new TensorRT-LLM version, new quantization approach).

### 4a. LLM, FP8 (fully scripted, reproducible)

The original AOT-compiled path, still valid on its own hardware/TensorRT-LLM-version pair:

```bash
python3 scripts/quantize_fp8.py            # HF BF16 -> FP8 TensorRT-LLM checkpoint
scripts/build_engine.sh                    # checkpoint -> compiled .engine (trtllm-build)
```

Both scripts live in this repo (`scripts/`) and are the exact commands the
original FP8 engine was built with. `build_engine.sh`'s own header explains
why the output isn't portable across GPU architectures or TensorRT-LLM
versions -- rerun it on the target machine, don't copy the `.engine` file.

### 4b. LLM, NVFP4 (currently deployed -- reconstructed script, not yet run end-to-end)

`triton_model_repo/nemotron_llm/config.pbtxt` currently points at
`models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4` (the "full" NVFP4 variant,
loaded dynamically via TensorRT-LLM's `LLM` API -- no `trtllm-build` AOT
step for this path, unlike 4a). This checkpoint already exists once S3 is
restored (step 2); the steps below are for regenerating it from scratch.

`scripts/quantize_nvfp4.py` now captures this as code (previously this
section was the only record of the step -- see "Known gaps" below for what
changed). Usage:

```bash
python3 scripts/quantize_nvfp4.py --hf-token $HF_TOKEN
# outputs to a scratch dir by default, NOT the live-served checkpoint path --
# pass --export-dir explicitly to produce a real replacement
```

Recorded in `speech-cascade-inference/reports/session-report.md`
("1. Quantization"), summarized here (see the script's own docstring for
the full breakdown of what's confirmed against upstream source vs.
reconstructed from prose):

```bash
pip install nvidia-modelopt[hf]==0.46.0

# NVIDIA's Model-Optimizer GitHub repo -- checkout the 0.46.0 tag (no "v"
# prefix -- session-report.md and an earlier version of this doc said
# "v0.46.0", which doesn't exist as a tag; confirmed via the GitHub API).
# Not main: main's hf_ptq.py example imports a module that doesn't exist in
# the 0.46.0 release.
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer /tmp/modelopt
cd /tmp/modelopt && git checkout 0.46.0

# Calibration: 512 samples from cnn_dailymail + a Nemotron post-training
# dataset (nvidia/Nemotron-Post-Training-Dataset-v2, gated -- needs an
# HF_TOKEN from an account that accepted its terms) -- this is NVIDIA's own
# "cnn_nemotron_v2_mix" combo, confirmed in modelopt's dataset registry, an
# even 256/256 split of 512 samples. Full NVFP4 (every linear layer, not
# MLP-only) is what's actually deployed -- the session report's cross-engine
# benchmark found it outscored MLP-only on this model, the opposite of
# NVIDIA's general guidance.
#
# Exact hf_ptq.py flags were run interactively and NOT preserved as a
# script. scripts/quantize_nvfp4.py reconstructs and runs:
#   --pyt_ckpt_path <source BF16 checkpoint>
#   --qformat nvfp4            # not the mlp_only variant; confirmed valid
#                               # against hf_ptq.py's own --qformat choices
#   --kv_cache_qformat none    # KV cache left unquantized; confirmed valid
#   --dataset cnn_nemotron_v2_mix --calib_size 512
#   --export_path <output, under models/Llama-3.1-Nemotron-Nano-4B-v1.1-NVFP4 to deploy>
# confirmed against the actual hf_ptq.py source + modelopt dataset/preset
# registries at tag 0.46.0 -- see the script's docstring for what remains
# genuinely unverified (calib_seq, batch_size, trust_remote_code).
```

`nemotron_llm/1/model.py` loads this checkpoint directly via
`tensorrt_llm.llmapi.llm._TrtLLM(model=engine_dir, tokenizer=tokenizer_dir, ...)`
-- no separate `trtllm-build` step, unlike 4a.

### 4c. Whisper ASR, TensorRT-LLM engines (documented here, not previously written down anywhere)

`triton_model_repo/whisper_asr/config.pbtxt` points `engine_dir` at
`models/whisper-base-trtllm` (compiled encoder + decoder engines) and
`assets_dir` at `models/whisper-base-trtllm/assets` (mel filterbank +
tokenizer vocab). `triton_model_repo/whisper_asr/1/trtllm_whisper/` vendors
the runtime pieces (`whisper_model.py`, `whisper_utils.py`, `tokenizer.py`)
from TensorRT-LLM's own Whisper example
(`tensorrt_llm/examples/models/core/whisper/` in the TensorRT-LLM source
tree, e.g. already present on this instance at
`/workspace/tensorrt_llm_repo/examples/models/core/whisper/`) -- that
example's `README.md` is the authoritative build reference; the values
below (`max_seq_len 114`, etc.) are copied from it directly and confirmed
to match this deployment's actual decoder config (see commit
"Cap whisper_asr's TensorRT-LLM KV cache to 5% of free GPU memory", which
references this exact `max_seq_len 114`):

```bash
cd /workspace/tensorrt_llm_repo/examples/models/core/whisper

# whisper-base weights (not large-v3 -- this deployment uses the base
# model, see main README's model choice)
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt

INFERENCE_PRECISION=float16
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_base_weights
output_dir=/workspace/speech-cascade-inference/models/whisper-base-trtllm

python3 convert_checkpoint.py --output_dir $checkpoint_dir --model_name base

trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
             --output_dir ${output_dir}/encoder \
             --moe_plugin disable \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --gemm_plugin disable \
             --bert_attention_plugin ${INFERENCE_PRECISION} \
             --max_input_len 3000 --max_seq_len=3000

trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
             --output_dir ${output_dir}/decoder \
             --moe_plugin disable \
             --max_beam_width 4 \
             --max_batch_size ${MAX_BATCH_SIZE} \
             --max_seq_len 114 \
             --max_input_len 14 \
             --max_encoder_input_len 3000 \
             --gemm_plugin ${INFERENCE_PRECISION} \
             --bert_attention_plugin ${INFERENCE_PRECISION} \
             --gpt_attention_plugin ${INFERENCE_PRECISION}

mkdir -p ${output_dir}/assets
cp assets/multilingual.tiktoken assets/mel_filters.npz ${output_dir}/assets/
```

This was NOT run to produce this repo's runbook -- it's reconstructed from
the TensorRT-LLM example's own README plus the deployed config values,
which match. **Verify against `tests/integration/test_whisper_asr.py`
before trusting a from-scratch rebuild.**

## 5. Point config.pbtxt at the right paths

Already done in this repo (`triton_model_repo/*/config.pbtxt` `parameters`
blocks) -- if you rebuilt an engine to a different path, update the
matching `engine_dir`/`tokenizer_dir`/`assets_dir`/`model_path` there and
reload just that model (see main README's "Managing the service" for the
reload API), not the whole server.

## 6. Install and start supervisor services

```bash
cp deploy/supervisor/speech-cascade-gateway.sh /opt/supervisor-scripts/
cp deploy/supervisor/speech-cascade-gateway.conf /etc/supervisor/conf.d/
cp deploy/supervisor/speech-cascade-triton.sh /opt/supervisor-scripts/
cp deploy/supervisor/speech-cascade-triton.conf /etc/supervisor/conf.d/
chmod +x /opt/supervisor-scripts/speech-cascade-gateway.sh /opt/supervisor-scripts/speech-cascade-triton.sh
supervisorctl reread && supervisorctl update
```

`speech-cascade-triton` is new as of this PR -- as of this writing, Triton
had been running as a manually-launched process with no supervisor entry
at all (confirmed via `supervisorctl status` and `ps aux`), meaning it
would not restart on crash and was not part of any documented boot
sequence. See `deploy/supervisor/speech-cascade-triton.sh`'s own header for
why it's captured but was deliberately **not installed live** as part of
this PR (avoiding a Triton restart while other agents have models loaded
and VRAM is tight).

`speech-cascade-triton` takes ~5-6 minutes to become ready (TensorRT-LLM
import + engine load dominate). `speech-cascade-gateway` needs Triton
already up and READY on `localhost:18001` before it's useful, though it
will start regardless.

## 7. Verify

```bash
curl -s -X POST http://localhost:18000/v2/repository/index    # all 4 models READY?
python3 -m pytest tests/integration -v                          # run from /venv/main
```

See `tests/README.md` for which Python environment each test tier needs.

## Known gaps (be upfront about these, don't paper over them)

- **4b (NVFP4 LLM quantization) now has `scripts/quantize_nvfp4.py`, but it
  is reconstructed and NOT verified end-to-end on this instance.** The
  exact `hf_ptq.py` invocation was run interactively during the original
  quantization session and was never preserved as a script; the survives-as-
  prose gap this note used to describe is closed, but the closing was done
  without re-running the quantization here (this instance's GPU had ~1.6GB
  free at reconstruction time -- comfortably below what a ~8.5GB BF16 model
  plus calibration needs, and taking that memory would have risked the
  live-serving models sharing this GPU). The script's flags and dataset
  handling are cross-checked directly against the `hf_ptq.py` source,
  argparse definitions, and dataset registry at the pinned `0.46.0` tag
  (fetched from `NVIDIA/TensorRT-Model-Optimizer` on GitHub), which is
  stronger grounding than the prose alone gave -- but "the flags are valid
  and match the documented calibration recipe" is not the same claim as
  "this exact invocation was run and reproduces the deployed checkpoint."
  Next person with real GPU headroom should run it end-to-end and diff the
  resulting checkpoint's tensor count/dtypes/size against the deployed one
  (963 tensors, 3.5GB, per the session report) before trusting it as a
  faithful rebuild.
- **4c (Whisper TensorRT-LLM engines) was reconstructed from the upstream
  example + the deployed config, not re-run end-to-end** during this PR
  (would require a multi-GB weight download and several minutes of
  `trtllm-build`, and risks GPU contention with the other agents' live
  work). The values are cross-checked against the actually-deployed
  decoder config (`max_seq_len 114` etc.) and against
  `tests/integration/test_whisper_asr.py` passing against the live
  server, but the *build steps themselves* are unverified on this
  instance.
- **`speech-cascade-triton`'s supervisor files are captured but not
  installed live** (see step 6) -- next person to restart Triton on this
  instance should install them then, rather than falling back to a manual
  launch again.
