#!/bin/bash
# Wraps this project's Triton launch (deploy/supervisor/speech-cascade-triton.sh)
# under `nsys profile`, so the resulting .nsys-rep captures GPU activity
# across all four backends -- TensorRT-LLM's own compiled engines/MPI worker
# for qwen_llm and whisper_asr, and the separate /venv/chatterbox subprocess
# chatterbox_tts spawns -- on one system-wide timeline.
#
# See deploy/PROFILING.md for the full writeup, including why nsys and
# torch.profiler aren't interchangeable here: torch.profiler only sees
# Python-visible ATen ops, so it's useful for chatterbox_tts (plain
# PyTorch -- see CHATTERBOX_TORCH_PROFILE_STEPS in chatterbox_worker.py) but
# blind to qwen_llm/whisper_asr, whose actual generation runs inside a
# TensorRT-LLM engine's compiled CUDA graph replay. nsys operates below the
# framework, at the CUDA driver level, so it's the only tool here that shows
# real per-kernel timing for those two stages, and the only one that puts
# all three GPU-bound stages on one shared timeline.
#
# The NVTX ranges duplicated across each triton_model_repo/*/1/model.py
# (and chatterbox_worker.py) label which pipeline stage produced which
# kernel in that timeline -- without them nsys shows an unlabeled wall of
# CUDA activity. Set NSYS_NVTX=0 (exported before running this script) to
# disable them if their overhead is ever a concern; it's normally
# negligible (a push/pop into nvToolsExt per call).
#
# IMPORTANT -- child-process tracing: chatterbox_tts's real GPU work runs in
# a separate /venv/chatterbox/bin/python3 subprocess spawned by
# chatterbox_tts/1/model.py, not as a plain thread inside tritonserver.
# --trace-fork-before-exec=true below tells nsys to follow processes that
# fork-then-exec (exactly what subprocess.Popen does), but this has not been
# verified end-to-end on this project's actual nsys/driver version. If
# chatterbox_tts's CUDA activity is missing from the report, attach nsys to
# that worker's PID directly instead, in a separate terminal:
#   ps aux | grep chatterbox_worker.py
#   nsys profile -o chatterbox_only --trace=cuda,nvtx -p <pid>
#
# This starts a REAL server bound to ports 18000-18002 and loads all four
# models onto the GPU, same as the real supervisor service -- do not run
# this against an instance where the real speech-cascade-triton service (or
# another profiling session) is already using the GPU/those ports. Stop the
# real service first:
#   supervisorctl stop speech-cascade-triton
#
# Usage:
#   scripts/profile_nsys.sh
#   NSYS_OUTPUT=/tmp/my_run scripts/profile_nsys.sh
#
# Then, once "all four models loading/loaded" prints below, drive real
# traffic against the server in another terminal, e.g.:
#   python3 scripts/load_test.py --concurrency 4 --total-requests 40
# Ctrl-C *this* script when done -- nsys stops the wrapped tritonserver
# process and finalizes the report on SIGINT, same as a normal server
# shutdown. View the result with `nsys-ui <report>.nsys-rep` (GUI) or
# `nsys stats <report>.nsys-rep` (text summary) on a machine with Nsight
# Systems installed (the report can be copied off this instance for that).

set -euo pipefail

command -v nsys >/dev/null || {
  echo "nsys not found on PATH -- install NVIDIA Nsight Systems first." >&2
  exit 1
}

NSYS_OUTPUT="${NSYS_OUTPUT:-./nsys_reports/speech_cascade_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "${NSYS_OUTPUT}")"

# Same environment quirks as deploy/supervisor/speech-cascade-triton.sh --
# see README.md's "Environment quirks fixed to make this run bare-metal"
# (#1-#7). Kept in sync with that script by hand; if that script's exports
# change, update this one too.
export PYTHONHOME=/venv/main
export PYTHONPATH=/venv/main/lib/python3.12/site-packages
export PATH="/venv/main/bin:${PATH}"
export LD_LIBRARY_PATH="/venv/main/lib:/workspace/speech-cascade-inference/triton_server/compat_libs/extracted/usr/lib/x86_64-linux-gnu:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/gateway/lib/python3.12/site-packages/nvidia/cu13/lib:/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/lib64:${LD_LIBRARY_PATH:-}"

TRITON_BIN=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/bin/tritonserver
MODEL_REPO=/workspace/speech-cascade-inference/triton_model_repo
BACKEND_DIR=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/backends

cd /workspace

nsys profile \
  --output="${NSYS_OUTPUT}" \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --force-overwrite=true \
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
    --log-verbose=0 &
NSYS_PID=$!

until curl -sf -o /dev/null http://127.0.0.1:18000/v2/health/live; do
  sleep 1
done

# Sequential load, same order/reasoning as the real supervisor script --
# see its header comment for why this must not be concurrent.
for model in qwen_llm whisper_asr chatterbox_tts voice_pipeline; do
  echo "loading ${model}..."
  if curl -sf -X POST "http://127.0.0.1:18000/v2/repository/models/${model}/load"; then
    echo "${model} load request succeeded"
  else
    echo "${model} load request FAILED -- see the server log above for the reason"
  fi
done

echo ""
echo "All four models loading/loaded, running under nsys."
echo "Drive real traffic now (e.g. python3 scripts/load_test.py ...)."
echo "Ctrl-C this script to stop the server and write the report to:"
echo "  ${NSYS_OUTPUT}.nsys-rep"

wait "${NSYS_PID}"
