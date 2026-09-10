"""Standalone ASR validation script -- loads WhisperTRTLLM directly (the
same TensorRT-LLM runtime triton_model_repo/whisper_asr/1/model.py uses in
production) against a real audio file, outside Triton's stub-process
environment. Useful for isolating an ASR problem without needing the whole
server up.

Rewritten from the pre-migration version, which loaded Whisper through
optimum.onnxruntime.ORTModelForSpeechSeq2Seq -- that path was replaced by
this TensorRT-LLM engine (see deploy/REBUILD.md "4c" and README.md's model
table) and no longer reflects what's actually served; onnxruntime-gpu was
removed from deploy/requirements-main.txt separately since this was its
only remaining caller. This version mirrors model.py's real load/feature-
extraction/inference calls line for line, so it stays a faithful standalone
reproduction of production rather than a second, drifting implementation.

NOT run against a live engine in this session -- this environment has no
GPU or model weights (models/ doesn't exist here; restored from S3 on a
real instance per deploy/REBUILD.md). Verify against a real
models/whisper-base-trtllm checkpoint before trusting this blindly, same
caveat this project's other "written without access to the live instance"
scripts carry (see scripts/archive_docs_to_s3.sh's own header for the
precedent).

Usage:
  python3 scripts/test_asr.py
  WHISPER_TEST_AUDIO=/path/to/your.wav python3 scripts/test_asr.py
"""
import os
import sys
import time

import librosa
import numpy as np
import torch

# trtllm_whisper is vendored inside the Triton model directory (not a pip
# package) -- same runtime whisper_asr/1/model.py imports, added to
# sys.path the same way that file does.
WHISPER_ASR_MODEL_DIR = os.environ.get(
    "WHISPER_ASR_MODEL_DIR",
    "/workspace/speech-cascade-inference/triton_model_repo/whisper_asr/1",
)
sys.path.insert(0, WHISPER_ASR_MODEL_DIR)
from trtllm_whisper.whisper_model import WhisperTRTLLM  # noqa: E402
from trtllm_whisper.whisper_utils import log_mel_spectrogram  # noqa: E402

ENGINE_DIR = os.environ.get(
    "WHISPER_ENGINE_DIR", "/workspace/speech-cascade-inference/models/whisper-base-trtllm"
)
ASSETS_DIR = os.environ.get("WHISPER_ASSETS_DIR", f"{ENGINE_DIR}/assets")
# tests/fixtures/vad_sample_16k.wav is this repo's own committed real-speech
# fixture (used throughout tests/integration and several investigation
# docs) -- a real, checked-in file, unlike the pre-migration script's
# hardcoded path into a different session's scratch directory.
AUDIO_PATH = os.environ.get(
    "WHISPER_TEST_AUDIO",
    "/workspace/speech-cascade-inference/tests/fixtures/vad_sample_16k.wav",
)

# Same constants whisper_asr/1/model.py uses.
WHISPER_SAMPLE_RATE = 16000
N_SAMPLES_30S = WHISPER_SAMPLE_RATE * 30
TEXT_PREFIX = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"

t0 = time.time()
model = WhisperTRTLLM(
    ENGINE_DIR,
    assets_dir=ASSETS_DIR,
    batch_size=1,
    use_py_session=False,
    num_beams=1,
    # Matches model.py's production value -- see that file's own comment on
    # why the example's 0.9 default OOMs a GPU shared with qwen_llm/chatterbox_tts.
    kv_cache_free_gpu_memory_fraction=0.05,
)
print(f"Whisper (TensorRT-LLM) load took {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
audio, sr = librosa.load(AUDIO_PATH, sr=WHISPER_SAMPLE_RATE)
audio = audio.astype(np.float32)
print(f"Loaded audio: {len(audio)} samples at {sr} Hz ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
# Full 30s-window padding, same as model.py -- see its own comment on why
# "longest" padding breaks this engine's remove_input_padding build config.
features = [
    log_mel_spectrogram(
        audio,
        model.n_mels,
        padding=N_SAMPLES_30S - audio.shape[-1],
        device="cuda",
        mel_filters_dir=ASSETS_DIR,
    ).unsqueeze(0)
]
mel_input_lengths = torch.tensor([f.shape[2] for f in features], dtype=torch.int32, device="cuda")

transcripts = model.process_batch(features, mel_input_lengths, TEXT_PREFIX)
print(f"ASR generation took {time.time()-t0:.1f}s", flush=True)
print("TRANSCRIPT:", transcripts[0], flush=True)
print("DONE", flush=True)
