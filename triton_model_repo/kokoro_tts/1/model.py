"""Triton Python backend for Kokoro-82M TTS, served via kokoro-onnx (GPU CUDA
execution provider)."""

import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        model_path = params["model_path"]["string_value"]
        voices_path = params["voices_path"]["string_value"]
        self.default_voice = params["default_voice"]["string_value"]

        from kokoro_onnx import Kokoro

        self.kokoro = Kokoro(model_path, voices_path)

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
