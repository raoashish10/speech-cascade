#!/usr/bin/env bash
# Container entrypoint for the triton service. Adapted from
# deploy/supervisor/speech-cascade-triton.sh, minus the
# PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH stitching that script needs on bare
# metal (see docker/triton/Dockerfile's header for why that shouldn't be
# needed inside a matched-toolchain NGC image) and minus the libssl.so.1.1
# compat-lib wiring (Ubuntu-24.04-specific, not this image's base OS).
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/speech-cascade-inference/models}"
MODEL_REPO="${MODEL_REPO:-/workspace/speech-cascade-inference/triton_model_repo}"

# Bootstrap weights/engines when models/ is empty. This runs on a plain
# `docker run`, a compose volume's first start, or a Runpod pod created
# straight from this image -- all three arrive here with nothing in
# MODELS_DIR unless something was mounted.
#
# Two sources, in priority order:
#
#   S3_BUCKET_URI set -> restore a prebuilt tree (deploy/REBUILD.md step 2's
#     `aws s3 sync`). Fastest, but needs somewhere to have built it first.
#
#   otherwise -> build from scratch: pull the checkpoints from the Hugging
#     Face Hub and compile the whisper TensorRT-LLM engines HERE. That is
#     not a fallback so much as the right default for this image: the
#     engines are GPU-architecture-specific (see docker/README.md's "GPU
#     portability caveat"), so compiling them at first start on the card
#     that will actually serve them is what makes one image work across
#     different GPUs. Nothing is baked in, and no S3 bucket is required.
#
# Cost of the from-scratch path, measured on an RTX PRO 4500: ~6GB of Hub
# download for Qwen3-8B-NVFP4 plus a few seconds of trtllm-build for
# whisper's encoder/decoder. It writes into MODELS_DIR, so mounting a
# volume there makes it a one-time cost per volume rather than per start.
if [ ! -d "${MODELS_DIR}/Qwen3-8B-NVFP4" ]; then
  if [ -n "${S3_BUCKET_URI:-}" ]; then
    echo "models/ is empty -- restoring from ${S3_BUCKET_URI}"
    aws s3 sync "${S3_BUCKET_URI}" /workspace/speech-cascade-inference/
  else
    echo "models/ is empty and S3_BUCKET_URI is unset -- building from"
    echo "scratch: Hugging Face checkpoints + whisper engines for this GPU."
    bash "${BOOTSTRAP_SCRIPT:-/opt/speech-cascade/scripts/build_models_from_scratch.sh}"
  fi
fi

# --http-address/--grpc-address bind 0.0.0.0 (not 127.0.0.1, unlike the
# bare-metal supervisor script) so the gateway container can reach this one
# over the compose network by service name -- docker-compose.yml doesn't
# publish these ports to the host, so they stay unreachable from outside
# the Docker network, the same "internal-only" property README.md
# describes for the bare-metal deployment.
tritonserver \
  --model-repository="${MODEL_REPO}" \
  --http-port=18000 \
  --grpc-port=18001 \
  --metrics-port=18002 \
  --http-address=0.0.0.0 \
  --grpc-address=0.0.0.0 \
  --model-control-mode=explicit \
  --exit-on-error=false \
  --log-verbose=0 \
  --metrics-config summary_latencies=true &
TRITON_PID=$!

until curl -sf -o /dev/null http://127.0.0.1:18000/v2/health/live; do
  sleep 1
done

# Sequential load, one model at a time, qwen_llm first -- not a stylistic
# choice, see deploy/supervisor/speech-cascade-triton.sh's header: loading
# all four concurrently overlapped qwen_llm's transient ~14.5GB host-RAM
# JIT-build peak with whisper_asr/chatterbox_tts's own startup RAM use and
# got qwen_llm's python-backend stub OOM-killed under that instance's 15GB
# cgroup limit. This container's own memory limit hasn't been re-verified
# against that number -- keeping the same conservative order is the safe
# default until it is.
for model in qwen_llm whisper_asr chatterbox_tts voice_pipeline; do
  echo "loading ${model}..."
  if curl -sf -X POST "http://127.0.0.1:18000/v2/repository/models/${model}/load" -d '{}'; then
    echo "${model} load request succeeded"
  else
    echo "${model} load request FAILED -- see the server log above for the reason"
  fi
done

wait "${TRITON_PID}"
