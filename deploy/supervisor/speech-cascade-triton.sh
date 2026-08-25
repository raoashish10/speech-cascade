#!/bin/bash
# Runs the project's Triton Inference Server (the 4-model voice pipeline:
# nemotron_llm, whisper_asr, kokoro_tts, voice_pipeline) as a managed
# supervisor service.
#
# STATUS AS OF THIS PR: captured but NOT installed on the live instance.
# The server is currently started by hand (confirmed via
# `ps aux | grep tritonserver` -- no supervisor entry existed for it, only
# for the streaming gateway). This file records the exact command line and
# environment that manual invocation actually needs (cross-checked against
# /proc/<pid>/environ of the running process and against README.md's
# "Environment quirks fixed to make this run bare-metal"), so a fresh
# instance -- or this one, the next time it needs a restart -- runs Triton
# as a real supervisor service (auto-restart on crash, logs in
# /var/log/portal/, part of the boot sequence) instead of a manually
# launched process nobody but a terminal scrollback remembers how to
# reproduce.
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
export LD_LIBRARY_PATH="/venv/main/lib:/workspace/speech-cascade-inference/triton_server/compat_libs/extracted/usr/lib/x86_64-linux-gnu:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/lib64:${LD_LIBRARY_PATH:-}"

TRITON_BIN=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/bin/tritonserver
MODEL_REPO=/workspace/speech-cascade-inference/triton_model_repo
BACKEND_DIR=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/backends

cd /workspace

pty "${TRITON_BIN}" \
  --model-repository="${MODEL_REPO}" \
  --backend-directory="${BACKEND_DIR}" \
  --http-port=18000 \
  --grpc-port=18001 \
  --metrics-port=18002 \
  --http-address=127.0.0.1 \
  --grpc-address=127.0.0.1 \
  --model-control-mode=explicit \
  --load-model=nemotron_llm \
  --load-model=whisper_asr \
  --load-model=kokoro_tts \
  --load-model=voice_pipeline \
  --log-verbose=0 2>&1
