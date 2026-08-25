"""Triton Python backend for Kokoro-82M TTS, served via kokoro-onnx (GPU CUDA
execution provider).

kokoro_onnx's default Kokoro(model_path, voices_path) constructor builds its
onnxruntime session with no session/provider options at all
(kokoro_onnx.session.create_session() -> rt.InferenceSession(model_path,
providers=providers), nothing else). That leaves the CUDA EP on its
memory-hungry defaults: arena_extend_strategy=kNextPowerOfTwo (doubles the
arena on every growth event instead of growing by the amount actually
requested) and cudnn_conv_algo_search=EXHAUSTIVE (benchmarks every candidate
conv algorithm -- allocating scratch workspace for each candidate -- the
first time a given input shape is seen). Kokoro's conv layers see a new
shape on close to every call, since synthesis input length varies with the
request text, so EXHAUSTIVE re-triggers constantly instead of being a
one-time warmup cost. Both are plausible drivers of the ~2.4GB/instance
arena growth this model was measured hitting under sustained load (see
config.pbtxt and docs/kokoro-tts-capacity-fix.md for the before/after
measurement). Building our own session with tuned CUDA EP options and
handing it to Kokoro.from_session() -- an escape hatch kokoro_onnx exposes
for exactly this -- keeps every other default (including the existing
Tensorrt->CUDA->CPU provider fallback order) unchanged.
"""

import json

import numpy as np
import onnxruntime as rt
import triton_python_backend_utils as pb_utils
from kokoro_onnx.session import resolve_providers


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        model_path = params["model_path"]["string_value"]
        voices_path = params["voices_path"]["string_value"]
        self.default_voice = params["default_voice"]["string_value"]

        from kokoro_onnx import Kokoro

        cuda_options = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "HEURISTIC",
        }
        providers = [
            (name, cuda_options) if name == "CUDAExecutionProvider" else name
            for name in resolve_providers()
        ]
        session = rt.InferenceSession(model_path, providers=providers)
        self.kokoro = Kokoro.from_session(session, voices_path)

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text = text_tensor.as_numpy().flatten()[0]
            if isinstance(text, bytes):
                text = text.decode("utf-8")

            voice = self.default_voice
            voice_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE")
            if voice_tensor is not None:
                v = voice_tensor.as_numpy().flatten()[0]
                voice = v.decode("utf-8") if isinstance(v, bytes) else v

            samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")

            audio_out = pb_utils.Tensor("AUDIO_SAMPLES", samples.astype(np.float32))
            sr_out = pb_utils.Tensor("SAMPLE_RATE", np.array([sample_rate], dtype=np.int32))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[audio_out, sr_out])
            )
        return responses

    def finalize(self):
        self.kokoro = None
