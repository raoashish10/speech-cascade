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

import librosa
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

sys.path.insert(0, os.path.dirname(__file__))
from trtllm_whisper.whisper_model import WhisperTRTLLM
from trtllm_whisper.whisper_utils import log_mel_spectrogram

# Whisper's encoder was trained on 16kHz mono audio; anything else must be
# resampled before feature extraction or the model mishears speed/pitch.
WHISPER_SAMPLE_RATE = 16000
# Whisper always processes a fixed 30s window internally.
N_SAMPLES_30S = WHISPER_SAMPLE_RATE * 30
TEXT_PREFIX = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
SPECIAL_TOKEN_RE = re.compile(r'<\|.*?\|>')


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
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
            # owns the whole GPU, but this one is shared with nemotron_llm
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

        transcripts = self.model.process_batch(features, mel_input_lengths, TEXT_PREFIX)

        responses = []
        for transcript in transcripts:
            transcript = SPECIAL_TOKEN_RE.sub('', transcript).strip()
            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        pass
