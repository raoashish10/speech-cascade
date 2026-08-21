"""Triton BLS orchestrator: chains whisper_asr -> nemotron_llm -> kokoro_tts
into a single request/response, matching the diagram's audio-in/audio-out
voice pipeline. Pure orchestration -- no model weights of its own, so it
needs no GPU instance; the three models it calls each manage their own.

voice_pipeline itself has max_batch_size 0 (no implicit batch dim), but the
three models it calls all have max_batch_size > 0 -- Triton reads their
first tensor dimension as the batch size. Every tensor forwarded downstream
must be reshaped to add a leading batch dim of 1, and every tensor read back
must have that dim stripped again.
"""

import numpy as np
import triton_python_backend_utils as pb_utils


def _batched(tensor):
    """Add a leading batch dim of 1 before forwarding to a batched model."""
    arr = tensor.as_numpy()
    return pb_utils.Tensor(tensor.name(), arr.reshape(1, *arr.shape))


def _run(model_name, inputs, output_names):
    request = pb_utils.InferenceRequest(
        model_name=model_name,
        requested_output_names=output_names,
        inputs=[_batched(t) for t in inputs],
    )
    response = request.exec()
    if response.has_error():
        raise pb_utils.TritonModelException(
            f"{model_name} failed: {response.error().message()}"
        )
    # Unlike inputs, BLS responses come back exactly as the callee's own
    # execute() constructed them per-request -- no wire-level batch dim to strip.
    return {
        name: pb_utils.get_output_tensor_by_name(response, name).as_numpy()
        for name in output_names
    }


def _decode_str(arr):
    value = arr.flatten()[0] if hasattr(arr, "flatten") else arr
    return value.decode("utf-8") if isinstance(value, bytes) else value


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SAMPLES")

            asr_inputs = [audio_tensor]
            sr_tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
            if sr_tensor is not None:
                asr_inputs.append(sr_tensor)

            asr_out = _run("whisper_asr", asr_inputs, ["TRANSCRIPT"])
            transcript = _decode_str(asr_out["TRANSCRIPT"])

            prompt_tensor = pb_utils.Tensor(
                "PROMPT", np.array([transcript.encode("utf-8")], dtype=np.object_)
            )
            llm_out = _run("nemotron_llm", [prompt_tensor], ["GENERATED_TEXT"])
            generated_text = _decode_str(llm_out["GENERATED_TEXT"])

            text_tensor = pb_utils.Tensor(
                "TEXT", np.array([generated_text.encode("utf-8")], dtype=np.object_)
            )
            tts_inputs = [text_tensor]
            voice_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE")
            if voice_tensor is not None:
                tts_inputs.append(voice_tensor)

            tts_out = _run("kokoro_tts", tts_inputs, ["AUDIO_SAMPLES", "SAMPLE_RATE"])

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "TRANSCRIPT",
                            np.array([transcript.encode("utf-8")], dtype=np.object_),
                        ),
                        pb_utils.Tensor(
                            "GENERATED_TEXT",
                            np.array([generated_text.encode("utf-8")], dtype=np.object_),
                        ),
                        pb_utils.Tensor("AUDIO_SAMPLES", tts_out["AUDIO_SAMPLES"]),
                        pb_utils.Tensor("SAMPLE_RATE", tts_out["SAMPLE_RATE"]),
                    ]
                )
            )
        return responses
