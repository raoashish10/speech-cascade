"""Triton BLS orchestrator: chains whisper_asr -> qwen_llm -> chatterbox_tts
into a single request/response, matching the diagram's audio-in/audio-out
voice pipeline. Pure orchestration -- no model weights of its own, so it
needs no GPU instance; the three models it calls each manage their own.

voice_pipeline itself has max_batch_size 0 (no implicit batch dim), but the
three models it calls all have max_batch_size > 0 -- Triton reads their
first tensor dimension as the batch size. Every tensor forwarded downstream
must be reshaped to add a leading batch dim of 1, and every tensor read back
must have that dim stripped again.
"""

import os
from contextlib import contextmanager

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

# See qwen_llm/1/model.py's own copy of this helper for the full rationale
# (deploy/PROFILING.md). Here it labels each downstream BLS call's wall-clock
# span (including the callee's own queueing) in nsys's timeline, so the
# three GPU-bound stages -- and voice_pipeline's own orchestration overhead
# between them -- are visible as one sequence, not just three isolated
# per-model traces. Set NSYS_NVTX=0 to disable.
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


def _run_llm_decoupled(prompt_tensor):
    """qwen_llm is decoupled/streaming now (max_batch_size: 0, no batch
    dim -- unlike _run()'s callees, don't _batched() it) so a plain exec()
    doesn't work against it; exec(decoupled=True) returns an iterator of
    responses instead of one. Drain and concatenate GENERATED_TEXT chunks
    into the single string voice_pipeline's own non-streaming contract
    still promises callers."""
    request = pb_utils.InferenceRequest(
        model_name="qwen_llm",
        requested_output_names=["GENERATED_TEXT"],
        inputs=[prompt_tensor],
    )
    chunks = []
    for response in request.exec(decoupled=True):
        if response.has_error():
            raise pb_utils.TritonModelException(
                f"qwen_llm failed: {response.error().message()}"
            )
        out = pb_utils.get_output_tensor_by_name(response, "GENERATED_TEXT")
        if out is None:
            continue
        chunks.append(_decode_str(out.as_numpy()))
    return "".join(chunks)


def _decode_str(arr):
    value = arr.flatten()[0] if hasattr(arr, "flatten") else arr
    return value.decode("utf-8") if isinstance(value, bytes) else value


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            # Any of the three downstream calls below can legitimately fail
            # under load now that whisper_asr/chatterbox_tts/qwen_llm all
            # have admission control (dynamic_batching default_queue_policy
            # REJECT / the in-flight-request counter) -- a downstream
            # rejection is an expected, everyday response under overload, not
            # a bug. _run()/_run_llm_decoupled() raise
            # pb_utils.TritonModelException on any such error; left
            # unhandled, an exception raised out of execute() gets wrapped by
            # Triton's python backend into a generic INTERNAL error carrying
            # a full Python stack trace -- technically not a hang, but not
            # the "clean, fast rejection" callers should see either.
            # Catching it here and returning a plain InferenceResponse(error=...)
            # for just this request gives callers the original short message
            # (e.g. "qwen_llm is overloaded: ...") with no traceback
            # noise, while any other request in the same execute() batch is
            # unaffected.
            try:
                responses.append(self._run_one(request))
            except pb_utils.TritonModelException as e:
                responses.append(
                    pb_utils.InferenceResponse(error=pb_utils.TritonError(str(e)))
                )
        return responses

    def _run_one(self, request):
        audio_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_SAMPLES")

        asr_inputs = [audio_tensor]
        sr_tensor = pb_utils.get_input_tensor_by_name(request, "SAMPLE_RATE")
        if sr_tensor is not None:
            asr_inputs.append(sr_tensor)

        with nvtx_range("voice_pipeline.whisper_asr"):
            asr_out = _run("whisper_asr", asr_inputs, ["TRANSCRIPT"])
        transcript = _decode_str(asr_out["TRANSCRIPT"])

        prompt_tensor = pb_utils.Tensor(
            "PROMPT", np.array([transcript.encode("utf-8")], dtype=np.object_)
        )
        with nvtx_range("voice_pipeline.qwen_llm"):
            generated_text = _run_llm_decoupled(prompt_tensor)

        text_tensor = pb_utils.Tensor(
            "TEXT", np.array([generated_text.encode("utf-8")], dtype=np.object_)
        )
        tts_inputs = [text_tensor]
        voice_tensor = pb_utils.get_input_tensor_by_name(request, "VOICE")
        if voice_tensor is not None:
            tts_inputs.append(voice_tensor)

        with nvtx_range("voice_pipeline.chatterbox_tts"):
            tts_out = _run("chatterbox_tts", tts_inputs, ["AUDIO_SAMPLES", "SAMPLE_RATE"])

        return pb_utils.InferenceResponse(
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
