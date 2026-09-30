#!/bin/bash
# Populate models/ from scratch -- no S3, no pre-built artifacts.
#
# Run this INSIDE the Triton container (docker/triton/Dockerfile's image),
# on the GPU the engines will actually serve on. The output lands in
# /workspace/speech-cascade-inference/models/, which docker-compose.yml
# mounts as a named volume -- engines are deliberately NOT baked into the
# image: they're GPU-architecture-specific (see docker/README.md's "GPU
# portability caveat") and multi-GB, so baking them in would both bloat the
# image and tie it to one card.
#
# VERIFIED end-to-end on an RTX PRO 4500 Blackwell (sm_120) Runpod pod
# against nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3: both models
# below reached Triton state READY afterwards. That's a real run, not a
# transcription of REBUILD.md -- though the whisper half follows
# deploy/REBUILD.md section 4c's commands exactly, with two additions that
# section doesn't mention because it was written for the bare-metal box:
#   - OPAL_PREFIX (see below)
#   - the TensorRT-LLM whisper example lives at /app/examples/... in this
#     image, not /workspace/tensorrt_llm_repo/examples/... as on bare metal
#
# What this does NOT cover:
#   - chatterbox_tts's reference voice clip (ref_audio_path in its
#     config.pbtxt, a ~41s recording). That's a deployment artifact, not
#     something reproducible from public sources -- supply your own.
#   - the vLLM T3 export (CHATTERBOX_BACKEND=vllm). See
#     scripts/export_chatterbox_t3_for_vllm.py, still unverified.
set -euo pipefail

# OpenMPI in this NGC image was built with prefix
# /build-result/hpcx-v2.25.1-.../ompi but installed at /usr/local/mpi, so
# without this OPAL can't find its own runtime data and MPI_Init_thread
# aborts with "opal_init:startup:internal-failure" -- which takes
# tensorrt_llm (and therefore trtllm-build) down with it.
export OPAL_PREFIX="${OPAL_PREFIX:-/usr/local/mpi}"

INFERENCE_DIR="${INFERENCE_DIR:-/workspace/speech-cascade-inference}"
MODELS_DIR="${INFERENCE_DIR}/models"
WHISPER_EXAMPLE_DIR="${WHISPER_EXAMPLE_DIR:-/app/examples/models/core/whisper}"

mkdir -p "${MODELS_DIR}"

# --- qwen_llm -------------------------------------------------------------
# No build step: the live config points at a checkpoint that's already
# quantized and published, pulled straight from the Hub (see
# deploy/REBUILD.md 4b -- the quantize_nvfp4.py path there is superseded and
# describes the older Nemotron checkpoint, not this one).
echo "==> qwen_llm: downloading Qwen3-8B-NVFP4 (~6GB)"
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
dest = os.path.join(
    os.environ.get("INFERENCE_DIR", "/workspace/speech-cascade-inference"),
    "models", "Qwen3-8B-NVFP4",
)
snapshot_download("raoashish10/Qwen3-8B-NVFP4", local_dir=dest)
print("qwen_llm checkpoint at", dest)
PY

# --- whisper_asr ----------------------------------------------------------
# deploy/REBUILD.md section 4c, run verbatim apart from the example path.
# max_seq_len 114 etc. are that section's values, which match this
# deployment's actual decoder config -- don't "tidy" them.
echo "==> whisper_asr: building TensorRT-LLM encoder/decoder engines"
cd "${WHISPER_EXAMPLE_DIR}"
mkdir -p assets
curl -sSL -o assets/multilingual.tiktoken \
  https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
curl -sSL -o assets/mel_filters.npz \
  https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
curl -sSL -o assets/base.pt \
  https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt

INFERENCE_PRECISION=float16
MAX_BATCH_SIZE=8
checkpoint_dir=whisper_base_weights
output_dir="${MODELS_DIR}/whisper-base-trtllm"

python3 convert_checkpoint.py --output_dir "${checkpoint_dir}" --model_name base

trtllm-build --checkpoint_dir "${checkpoint_dir}/encoder" \
             --output_dir "${output_dir}/encoder" \
             --moe_plugin disable \
             --max_batch_size "${MAX_BATCH_SIZE}" \
             --gemm_plugin disable \
             --bert_attention_plugin "${INFERENCE_PRECISION}" \
             --max_input_len 3000 --max_seq_len=3000

trtllm-build --checkpoint_dir "${checkpoint_dir}/decoder" \
             --output_dir "${output_dir}/decoder" \
             --moe_plugin disable \
             --max_beam_width 4 \
             --max_batch_size "${MAX_BATCH_SIZE}" \
             --max_seq_len 114 \
             --max_input_len 14 \
             --max_encoder_input_len 3000 \
             --gemm_plugin "${INFERENCE_PRECISION}" \
             --bert_attention_plugin "${INFERENCE_PRECISION}" \
             --gpt_attention_plugin "${INFERENCE_PRECISION}"

mkdir -p "${output_dir}/assets"
cp assets/multilingual.tiktoken assets/mel_filters.npz "${output_dir}/assets/"

# --- chatterbox_tts (CHATTERBOX_BACKEND=vllm only) -------------------------
# vLLM cannot load chatterbox's T3 GPT2 backbone from the package directly;
# it needs it re-exported as a standalone HF-loadable directory for the
# out-of-tree loader (triton_model_repo/chatterbox_tts/1/vllm_t3/). That
# export is this step. Skipped when /venv/vllm is absent, i.e. when the image
# was built without the vLLM backend, and skipped when the directory already
# exists -- the weights come from the Hub and do not change.
#
# ~22s and 1.7GB, both measured. The pytorch backend needs none of this.
VLLM_PYTHON="${VLLM_PYTHON:-/venv/vllm/bin/python3}"
VLLM_T3_DIR="${VLLM_T3_DIR:-${INFERENCE_DIR}/vllm_t3_model_dir}"
EXPORT_SCRIPT="${EXPORT_SCRIPT:-/opt/speech-cascade/scripts/export_chatterbox_t3_for_vllm.py}"

if [ ! -x "${VLLM_PYTHON}" ]; then
  echo "==> chatterbox_tts: /venv/vllm not present, skipping the T3 export"
  echo "    (CHATTERBOX_BACKEND=pytorch needs no export)"
elif [ -f "${VLLM_T3_DIR}/model.safetensors" ]; then
  echo "==> chatterbox_tts: T3 export already present at ${VLLM_T3_DIR}"
else
  echo "==> chatterbox_tts: exporting T3 backbone for vLLM (~1.7GB)"
  "${VLLM_PYTHON}" "${EXPORT_SCRIPT}" --output-dir "${VLLM_T3_DIR}"
fi

echo "==> done. models/ now contains:"
ls -1 "${MODELS_DIR}"
