#!/bin/bash
# Compiles the FP8 TensorRT-LLM checkpoint produced by quantize_fp8.py into
# an ahead-of-time TensorRT .engine -- the second and final step of the
# HF BF16 -> FP8 checkpoint -> compiled engine pipeline documented in
# README.md ("Why the LLM is Llama-3.1-Nemotron-Nano-4B").
#
# This engine is NOT portable: it's compiled for one specific GPU
# architecture and one specific TensorRT-LLM version (this project's
# reference build: RTX 5070 Ti / sm_120, TensorRT-LLM 1.2.1, driver 595.84,
# CUDA 13.2 toolkit -- see README.md "Environment quirks"). Do not copy the
# output directory to different hardware or a different TensorRT-LLM
# install; rerun this script there instead.
#
# Usage:
#   scripts/build_engine.sh
#   CHECKPOINT_DIR=/path/to/ckpt ENGINE_DIR=/path/to/out scripts/build_engine.sh

set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/speech-cascade-inference/models/llama_nemotron_fp8_ckpt}"
ENGINE_DIR="${ENGINE_DIR:-/workspace/speech-cascade-inference/models/llama_nemotron_engine}"
TIMING_CACHE="${TIMING_CACHE:-/workspace/speech-cascade-inference/models/trtllm_build_timing_cache.bin}"

# Matches nemotron_llm/config.pbtxt's dynamic_batching (preferred_batch_size
# up to 8) and the max_seq_len the NVFP4 deployment report confirmed this
# engine was built with (see "An unconstrained 128K context length" -- the
# FP8 engine never hit that OOM because this limit was already baked in
# here at build time).
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "error: checkpoint dir not found: $CHECKPOINT_DIR" >&2
  echo "run scripts/quantize_fp8.py first, or point CHECKPOINT_DIR at an existing checkpoint" >&2
  exit 1
fi

trtllm-build \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --output_dir "$ENGINE_DIR" \
  --max_batch_size "$MAX_BATCH_SIZE" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --timing_cache "$TIMING_CACHE"
  # reduce_fusion, multiple_profiles, and use_fp8_context_fmha were left at
  # their (disabled) defaults in the reference build -- see README.md
  # "Not yet done" for why, and worth revisiting together as a rebuild.

echo "engine written to $ENGINE_DIR"
echo "point nemotron_llm/config.pbtxt's engine_dir parameter at this path"
