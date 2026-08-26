"""Triton Python backend for NVIDIA Magpie-TTS-Multilingual-357M, served via
NeMo's own PyTorch runtime (nemo.collections.tts.models.MagpieTTSModel).

Replaces kokoro_tts. Unlike Kokoro (a single feed-forward ONNX graph),
Magpie-TTS is a multi-stage autoregressive system -- a transformer decoder
generating discrete audio-codec tokens (cross-attending to text and a baked
speaker embedding), a neural audio codec reconstructing the waveform, and a
local transformer expanding codebook predictions per frame. There is no
ONNX/TensorRT export path for this yet (see deploy/REBUILD.md); this is the
plain-PyTorch baseline, run through NeMo's own do_tts() convenience method.

Interface kept identical to kokoro_tts so voice_pipeline and the streaming
gateway don't need to change: TEXT (+ optional VOICE) in, AUDIO_SAMPLES (+
SAMPLE_RATE) out. VOICE now selects one of Magpie's 5 baked English speakers
by name (see SPEAKER_MAP) instead of a Kokoro voice id.
"""

import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

SPEAKER_MAP = {
    "Aria": 0,
    "Jason": 1,
    "John": 2,
    "Leo": 3,
    "Sofia": 4,
}

# Fixed by the model (22.05kHz mono, per the model card) -- not something
# do_tts() reports back per-call, so it's hardcoded here same as Kokoro's
# was effectively fixed by its own onnx graph.
SAMPLE_RATE = 22050


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        model_path = params["model_path"]["string_value"]
        self.default_voice = params["default_voice"]["string_value"]
        self.default_language = params["default_language"]["string_value"]

        from nemo.collections.tts.models import MagpieTTSModel

        self.model = MagpieTTSModel.restore_from(model_path, map_location="cuda")
        self.model = self.model.cuda().eval()

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
            speaker_index = SPEAKER_MAP.get(voice, SPEAKER_MAP[self.default_voice])

            with torch.no_grad():
                audio, audio_len = self.model.do_tts(
                    transcript=text,
                    language=self.default_language,
                    apply_TN=False,
                    use_cfg=True,
                    speaker_index=speaker_index,
                )

            samples = audio[0, : audio_len[0]].float().cpu().numpy().astype(np.float32)

            audio_out = pb_utils.Tensor("AUDIO_SAMPLES", samples)
            sr_out = pb_utils.Tensor("SAMPLE_RATE", np.array([SAMPLE_RATE], dtype=np.int32))
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[audio_out, sr_out])
            )
        return responses

    def finalize(self):
        self.model = None
