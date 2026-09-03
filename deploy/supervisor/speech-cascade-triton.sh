#!/bin/bash
# Runs the project's Triton Inference Server (the 4-model voice pipeline:
# qwen_llm, whisper_asr, chatterbox_tts, voice_pipeline) as a managed
# supervisor service.
#
# Installed and running live as a supervisor service (this file's history:
# it started out captured-but-not-installed, documenting a manually-launched
# process with no supervisor entry, autorestart, or place in the boot
# sequence -- since fixed by actually installing it).
#
# Models are loaded ONE AT A TIME via the explicit-mode repository control
# API below, not with --load-model flags at startup. Measured directly on
# this box: qwen_llm's own JIT engine build transiently peaks host RAM at
# ~14.5GB (settling back to ~8-9GB once loaded) against this container's
# ~15GB cgroup memory.max. Loading all four models concurrently at startup
# (the previous --load-model=* approach) overlapped that peak with
# whisper_asr/chatterbox_tts's own concurrent startup RAM use, exceeding the
# container's memory ceiling and getting qwen_llm's python-backend stub
# OOM-killed -- surfaced as "Stub process 'qwen_llm_0_0' is not healthy"
# with no Python traceback (consistent with SIGKILL, not a catchable
# exception). Triton's /v2/repository/models/{name}/load call blocks until
# that model's load finishes (success or failure), so looping over it
# sequentially guarantees no two models' loading-time peaks overlap.
# qwen_llm loads first, while baseline RAM is lowest.
#
# To install on a live instance:
#   cp deploy/supervisor/speech-cascade-triton.sh /opt/supervisor-scripts/
#   cp deploy/supervisor/speech-cascade-triton.conf /etc/supervisor/conf.d/
#   chmod +x /opt/supervisor-scripts/speech-cascade-triton.sh
#   supervisorctl reread && supervisorctl update
#
# CAUTION: do not install/restart this on an instance where Triton is
# already running with models another process or agent depends on --
# starting this service binds ports 18000-18002 and loads all four models
# onto the GPU again, colliding with (or duplicating VRAM alongside) a
# manually-launched server. Coordinate before restarting a live one; see
# README.md "Managing the service" for the ~5-6 minute restart cost
# (dominated by the LLM's TensorRT-LLM import + engine load).
#
# Binds to 127.0.0.1 only (matches the currently-running process) -- no
# portal.yaml entry needed here. External exposure of anything Triton-
# adjacent (if wanted) is a separate concern, being handled by the
# external-access/Caddy work in a parallel PR.

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

# --- Environment quirks from README.md's "Environment quirks fixed to
# make this run bare-metal" (#1-#7) -- required for the Python-backend
# stub's CPython ABI to resolve consistently across the system Python, the
# venv's conda-forge Python, and Triton's own bundled libpython, and for
# onnxruntime's CUDA EP to find cuBLAS/cuDNN 12.x (shipped inside the
# venv's own nvidia-* pip packages) instead of the system's cuBLAS 13.x. ---
export PYTHONHOME=/venv/main
export PYTHONPATH=/venv/main/lib/python3.12/site-packages
export PATH="/venv/main/bin:${PATH}"
export LD_LIBRARY_PATH="/venv/main/lib:/workspace/speech-cascade-inference/triton_server/compat_libs/extracted/usr/lib/x86_64-linux-gnu:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/gateway/lib/python3.12/site-packages/nvidia/cu13/lib:/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/lib64:${LD_LIBRARY_PATH:-}"

TRITON_BIN=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/bin/tritonserver
MODEL_REPO=/workspace/speech-cascade-inference/triton_model_repo
BACKEND_DIR=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/backends

cd /workspace

# Not run through pty/unbuffer here (unlike other supervisor scripts) --
# it needs to background cleanly under a plain $!/wait pair below, and
# tritonserver's own log output already flushes promptly without it.
"${TRITON_BIN}" \
  --model-repository="${MODEL_REPO}" \
  --backend-directory="${BACKEND_DIR}" \
  --http-port=18000 \
  --grpc-port=18001 \
  --metrics-port=18002 \
  --http-address=127.0.0.1 \
  --grpc-address=127.0.0.1 \
  --model-control-mode=explicit \
  --exit-on-error=false \
  --log-verbose=0 2>&1 &
TRITON_PID=$!

until curl -sf -o /dev/null http://127.0.0.1:18000/v2/health/live; do
  sleep 1
done

# qwen_llm first (baseline RAM lowest here), then the lighter models --
# see the header comment for why this must stay sequential, not parallel.
for model in qwen_llm whisper_asr chatterbox_tts voice_pipeline; do
  echo "loading ${model}..."
  if curl -sf -X POST "http://127.0.0.1:18000/v2/repository/models/${model}/load"; then
    echo "${model} load request succeeded"
  else
    echo "${model} load request FAILED -- see the server log above for the reason"
  fi
done

wait "${TRITON_PID}"
