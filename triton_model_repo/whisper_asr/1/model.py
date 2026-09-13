"""Triton Python backend for Whisper-base ASR, served via TensorRT-LLM's
native Whisper runtime: separate encoder/decoder TensorRT engines, with the
autoregressive decode loop (including the prefill/decode branch that broke
ONNX Runtime's TensorRT execution provider's static-graph parser) handled by
TensorRT-LLM's own C++ executor rather than baked into a single ONNX graph.
"""

import json
import os
import re
import sys
from contextlib import contextmanager

# Deployment-specific fix -- see qwen_llm/1/model.py's identical block for
# the full root-cause writeup: CPython's `_strptime` lazily builds a
# locale-format cache on first use, and that lazy init is not thread-safe.
# Triton's python-backend stub is multi-threaded from the moment it starts,
# so the first transitive call into `_strptime` (pandas, deep in
# transformers' import chain, imported below via trtllm_whisper) can race
# and corrupt the cache -- symptom: a misleading "cannot import name
# 'GenerationMixin' from 'transformers.generation'" several frames removed
# from the real AttributeError inside stdlib calendar.py. Priming it here,
# single-threaded, before any of those imports, avoids the race entirely.
#
# A single priming call isn't enough: something else in Triton's own stub
# hits this cold-start path CONTINUOUSLY, not just once (confirmed
# empirically against qwen_llm's identical fix -- a bare tight retry loop
# with no delay lost every attempt). Most likely culprit: Triton's own C++
# logging emits a timestamp on every log line from multiple internal
# threads for the whole process lifetime, so a real gap has to open up
# between two of its calls for a Python-side attempt to land cleanly. A
# short sleep between retries plus a broad except (the corrupted-state
# failure mode isn't guaranteed to always surface as AttributeError) is
# what actually makes this land in practice.
import random
import time
for _attempt in range(200):
    try:
        time.strptime("2000-01-01", "%Y-%m-%d")
        break
    except Exception:
        time.sleep(0.05 + random.random() * 0.05)

import librosa
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

# Backend switch for the soak-test comparison (see qwen_llm/1/model.py's
# identical QWEN_BACKEND comment for the full rationale): WHISPER_BACKEND=
# trtllm (default) is the existing TensorRT-LLM engine path below.
# WHISPER_BACKEND=pytorch runs the same openai/whisper-base checkpoint
# through plain HF `transformers` (WhisperForConditionalGeneration.generate())
# instead -- no compiled TensorRT engines, no separate encoder/decoder
# engine build step. Checked BEFORE the trtllm_whisper import below: that
# package transitively imports tensorrt_llm -> mpi4py.MPI, which spawns an
# `orted` MPI singleton daemon as an import side effect (see qwen_llm/1/
# model.py's PYTHONHOME comment for the full mechanism) -- the pytorch
# backend has no use for any of that and skips the import entirely rather
# than pay its cost/risk for nothing.
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "trtllm")

sys.path.insert(0, os.path.dirname(__file__))
if WHISPER_BACKEND != "pytorch":
    # See qwen_llm/1/model.py's identical os.environ.pop("PYTHONHOME", ...)
    # comment: must happen before tensorrt_llm is imported (transitively, via
    # trtllm_whisper below), not just before WhisperTRTLLM(...) is constructed.
    os.environ.pop("PYTHONHOME", None)
    from trtllm_whisper.whisper_model import WhisperTRTLLM
    from trtllm_whisper.whisper_utils import log_mel_spectrogram

# Whisper's encoder was trained on 16kHz mono audio; anything else must be
# resampled before feature extraction or the model mishears speed/pitch.
WHISPER_SAMPLE_RATE = 16000
# Whisper always processes a fixed 30s window internally.
N_SAMPLES_30S = WHISPER_SAMPLE_RATE * 30
TEXT_PREFIX = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
SPECIAL_TOKEN_RE = re.compile(r'<\|.*?\|>')

# See qwen_llm/1/model.py's own copy of this helper for the full rationale
# (deploy/PROFILING.md): labels this stage's GPU work in nsys's timeline.
# torch.profiler can't see inside process_batch() -- it runs inside
# TensorRT-LLM's own compiled encoder/decoder engines, not as Python-visible
# ATen ops. Set NSYS_NVTX=0 to disable.
_NVTX_ENABLED = os.environ.get("NSYS_NVTX", "1") != "0"


@contextmanager
def nvtx_range(name):
    if _NVTX_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        if WHISPER_BACKEND == "pytorch":
            self._init_pytorch(params)
        else:
            self._init_trtllm(params, model_config)

    def _init_pytorch(self, params):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_dir = params.get("pytorch_model_dir", {}).get(
            "string_value", "/workspace/speech-cascade-inference/models/whisper-base-hf"
        )
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.pt_model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.float16
        ).to("cuda")
        self.pt_model.eval()

    def _init_trtllm(self, params, model_config):
        engine_dir = params["engine_dir"]["string_value"]
        assets_dir = params["assets_dir"]["string_value"]
        max_batch_size = model_config.get("max_batch_size", 8) or 8

        self.assets_dir = assets_dir
        self.model = WhisperTRTLLM(
            engine_dir,
            assets_dir=assets_dir,
            batch_size=max_batch_size,
            use_py_session=False,
            num_beams=1,
            # The example's own default (0.9) claims ~90% of whatever VRAM is
            # free at load time for KV cache paging -- reasonable if Whisper
            # owns the whole GPU, but this one is shared with qwen_llm
            # and kokoro_tts. A max_seq_len of 114 needs only a handful of KV
            # cache blocks per sequence; measured OOM-ing kokoro_tts under
            # concurrent voice_pipeline load at 0.9, fine at 0.05.
            kv_cache_free_gpu_memory_fraction=0.05,
        )

    def execute(self, requests):
        # Batch every request bundled into this execute() call into a single
        # process_batch() call, same reasoning as the ONNX Runtime version
        # this replaces: dynamic_batching in config.pbtxt only pays off if
        # this loop doesn't call the model once per request.
        audios = []
        for request in requests:
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SAMPLES")
            audio = audio_tensor.as_numpy().flatten().astype(np.float32)

            sr_tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
            input_sample_rate = (
                int(sr_tensor.as_numpy().flatten()[0]) if sr_tensor is not None else WHISPER_SAMPLE_RATE
            )
            if input_sample_rate != WHISPER_SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=input_sample_rate, target_sr=WHISPER_SAMPLE_RATE)
            audios.append(audio)

        if WHISPER_BACKEND == "pytorch":
            with nvtx_range("whisper_asr.generate"):
                transcripts = self._transcribe_pytorch(audios)
            responses = []
            for transcript in transcripts:
                out_tensor = pb_utils.Tensor(
                    "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=np.object_)
                )
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
            return responses

        # Pad every sample to the full 30s window Whisper was trained/built
        # for ("max" padding strategy -- the TensorRT-LLM example's own
        # default). Padding only to the longest-in-batch instead (its
        # "longest" strategy, cheaper on paper) was tried first and measured
        # broken: with remove_input_padding enabled (this engine's build
        # config), the ragged/list encoder-input-features path it requires
        # degenerates into repeating the first decoded phrase until the
        # token budget runs out -- reproduced even for a single-item batch,
        # so it wasn't specific to concurrent/duplicate requests. Full 30s
        # padding for every sample avoids that path and decodes cleanly.
        with nvtx_range("whisper_asr.feature_extraction"):
            features = [
                log_mel_spectrogram(
                    audio,
                    self.model.n_mels,
                    padding=N_SAMPLES_30S - audio.shape[-1],
                    device='cuda',
                    mel_filters_dir=self.assets_dir,
                ).unsqueeze(0)
                for audio in audios
            ]

            mel_input_lengths = torch.tensor(
                [f.shape[2] for f in features], dtype=torch.int32, device='cuda'
            )

        with nvtx_range("whisper_asr.process_batch"):
            transcripts = self.model.process_batch(features, mel_input_lengths, TEXT_PREFIX)

        responses = []
        for transcript in transcripts:
            transcript = SPECIAL_TOKEN_RE.sub('', transcript).strip()
            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def _transcribe_pytorch(self, audios):
        # Same 16kHz-mono-in / batched-generate contract as the trtllm path
        # above, but through plain HF transformers -- WhisperFeatureExtractor
        # (inside self.processor) handles the fixed 30s-window padding
        # itself, no need to replicate log_mel_spectrogram's manual padding.
        inputs = self.processor(audios, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            generated_ids = self.pt_model.generate(
                input_features, language="english", task="transcribe"
            )
        transcripts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [SPECIAL_TOKEN_RE.sub('', t).strip() for t in transcripts]

    def finalize(self):
        pass
