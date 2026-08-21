"""Triton Python backend for Whisper-base ASR, served via optimum's ONNX Runtime
wrapper (GPU CUDA execution provider)."""

import json

import librosa
import numpy as np
import triton_python_backend_utils as pb_utils

# Whisper's encoder was trained on 16kHz mono audio; anything else must be
# resampled before feature extraction or the model mishears speed/pitch.
WHISPER_SAMPLE_RATE = 16000


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        model_dir = params["model_dir"]["string_value"]

        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            use_merged=True,
            # IOBinding reuses pre-allocated output buffers across generate()
            # calls on this session. Under concurrent load Triton bundles
            # multiple queued requests into one execute() call, and back-to-
            # back generate() calls on the same session then hit stale
            # buffer shapes from the previous call's last decode step
            # (RuntimeError: computed output shape doesn't match the
            # pre-allocated one). Disabling it trades a small copy overhead
            # for correctness when the session is reused this way.
            use_io_binding=False,
            provider="CUDAExecutionProvider",
        ).to("cuda")

    def execute(self, requests):
        responses = []
        for request in requests:
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SAMPLES")
            audio = audio_tensor.as_numpy().flatten().astype(np.float32)

            sr_tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
            input_sample_rate = (
                int(sr_tensor.as_numpy().flatten()[0]) if sr_tensor is not None else WHISPER_SAMPLE_RATE
            )
            if input_sample_rate != WHISPER_SAMPLE_RATE:
                audio = librosa.resample(
                    audio, orig_sr=input_sample_rate, target_sr=WHISPER_SAMPLE_RATE
                )

            inputs = self.processor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")
            inputs["input_features"] = inputs["input_features"].to("cuda")

            generated_ids = self.model.generate(inputs["input_features"])
            transcript = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            out_tensor = pb_utils.Tensor(
                "TRANSCRIPT", np.array([transcript.encode("utf-8")], dtype=np.object_)
            )
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses

    def finalize(self):
        self.model = None
