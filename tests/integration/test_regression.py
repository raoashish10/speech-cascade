"""A small, fixed set of "known good" sample requests per model -- a
lightweight regression guard against the live server, not exhaustive
correctness testing of model quality. Each of these was manually spot-
checked during development (see reports/session-report.md in
speech-cascade-inference for the deployment history); this formalizes that
into something that fails loudly if a future change breaks basic sanity
(model errors out, returns empty output, etc.) without pinning down exact
text/audio content, which would be too brittle across model/decoding
changes.

NOTE for whoever picks this up after the capacity/backpressure PR lands:
these assertions check for success + non-empty output, not specific
latency or instance-count behavior. If that PR adds admission-control
rejection (e.g. 429-equivalent errors under load), these single-request,
non-concurrent calls should still pass unchanged -- but worth a quick
recheck once that PR is in.
"""

import numpy as np
import pytest
import tritonclient.grpc.aio as grpcclient

CLIENT_TIMEOUT_S = 60.0

# Fixed prompts, chosen to be short, unambiguous, and non-empty-completion
# under greedy decoding -- not testing model "intelligence", just that the
# serving stack still produces *something* sane end to end.
KNOWN_GOOD_LLM_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Two plus two equals",
]

KNOWN_GOOD_TTS_TEXTS = [
    "Hello there, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
]


@pytest.mark.parametrize("prompt", KNOWN_GOOD_LLM_PROMPTS)
async def test_nemotron_llm_known_good_prompts(grpc_client, require_ready, stream_infer_collect, prompt):
    await require_ready("nemotron_llm")
    arr = np.array([prompt.encode("utf-8")], dtype=object)
    inp = grpcclient.InferInput("PROMPT", arr.shape, "BYTES")
    inp.set_data_from_numpy(arr)
    text = await stream_infer_collect(
        grpc_client, "nemotron_llm", [inp], [grpcclient.InferRequestedOutput("GENERATED_TEXT")]
    )
    assert text.strip() != "", f"empty completion for known-good prompt {prompt!r}"


@pytest.mark.parametrize("text", KNOWN_GOOD_TTS_TEXTS)
async def test_kokoro_tts_known_good_texts(grpc_client, require_ready, text):
    await require_ready("kokoro_tts")
    arr = np.array([[text]], dtype=object)
    inp = grpcclient.InferInput("TEXT", arr.shape, "BYTES")
    inp.set_data_from_numpy(arr)
    result = await grpc_client.infer(
        model_name="kokoro_tts",
        inputs=[inp],
        outputs=[
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ],
        client_timeout=CLIENT_TIMEOUT_S,
    )
    audio = result.as_numpy("AUDIO_SAMPLES").astype(np.float32).flatten()
    sample_rate = int(result.as_numpy("SAMPLE_RATE").flatten()[0])
    assert audio.size > 0, f"empty audio for known-good text {text!r}"
    assert sample_rate > 0


async def test_known_good_speech_round_trip_through_whisper_asr(grpc_client, require_ready, synth_speech):
    """synth_speech is itself a known-good TTS sample ("This is a known
    good test sentence.") -- round-tripping it through ASR is the fixed
    regression case for the ASR stage, without committing a stale binary
    audio fixture to git."""
    await require_ready("whisper_asr")
    _text, audio, sample_rate = synth_speech
    audio2d = audio.reshape(1, -1)
    sr_arr = np.array([[sample_rate]], dtype=np.int32)
    audio_inp = grpcclient.InferInput("AUDIO_SAMPLES", audio2d.shape, "FP32")
    audio_inp.set_data_from_numpy(audio2d)
    sr_inp = grpcclient.InferInput("SAMPLE_RATE", sr_arr.shape, "INT32")
    sr_inp.set_data_from_numpy(sr_arr)
    result = await grpc_client.infer(
        model_name="whisper_asr",
        inputs=[audio_inp, sr_inp],
        outputs=[grpcclient.InferRequestedOutput("TRANSCRIPT")],
        client_timeout=CLIENT_TIMEOUT_S,
    )
    transcript = result.as_numpy("TRANSCRIPT").flatten()[0]
    transcript = transcript.decode("utf-8") if isinstance(transcript, bytes) else transcript
    assert transcript.strip() != "", "known-good speech sample produced an empty transcript"
