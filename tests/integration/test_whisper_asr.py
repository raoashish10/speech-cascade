"""Integration tests for whisper_asr in isolation, against the live Triton
server. See tests/integration/conftest.py and tests/README.md.

The `integration` marker is applied automatically to every test collected
under this directory (tests/integration/conftest.py's
pytest_collection_modifyitems) -- no need to mark individual tests here.
"""

import numpy as np
import tritonclient.grpc.aio as grpcclient

CLIENT_TIMEOUT_S = 60.0


async def _asr_infer(client, audio, sample_rate):
    audio = audio.astype(np.float32).reshape(1, -1)
    sr_arr = np.array([[sample_rate]], dtype=np.int32)
    audio_inp = grpcclient.InferInput("AUDIO_SAMPLES", audio.shape, "FP32")
    audio_inp.set_data_from_numpy(audio)
    sr_inp = grpcclient.InferInput("SAMPLE_RATE", sr_arr.shape, "INT32")
    sr_inp.set_data_from_numpy(sr_arr)
    result = await client.infer(
        model_name="whisper_asr",
        inputs=[audio_inp, sr_inp],
        outputs=[grpcclient.InferRequestedOutput("TRANSCRIPT")],
        client_timeout=CLIENT_TIMEOUT_S,
    )
    value = result.as_numpy("TRANSCRIPT").flatten()[0]
    return value.decode("utf-8") if isinstance(value, bytes) else value


async def test_whisper_asr_responds_to_real_speech(grpc_client, require_ready, synth_speech):
    await require_ready("whisper_asr")
    _text, audio, sample_rate = synth_speech
    transcript = await _asr_infer(grpc_client, audio, sample_rate)

    assert isinstance(transcript, str)
    assert transcript.strip() != "", "transcript should be non-empty for real speech input"


async def test_whisper_asr_handles_silence_without_error(grpc_client, require_ready):
    """Not a quality check (silence -> empty/garbage transcript is fine) --
    just confirms the model doesn't error out on a degenerate input."""
    await require_ready("whisper_asr")
    silence = np.zeros(16000, dtype=np.float32)  # 1s @ 16kHz
    transcript = await _asr_infer(grpc_client, silence, 16000)
    assert isinstance(transcript, str)


async def test_whisper_asr_resamples_non_native_sample_rate(grpc_client, require_ready, synth_speech):
    """whisper_asr resamples internally when SAMPLE_RATE != 16000 (see
    triton_model_repo/whisper_asr/1/model.py) -- confirm audio tagged at
    kokoro_tts's native rate (usually 24kHz, not Whisper's native 16kHz)
    still returns a sane, non-erroring response, exercising that resample
    code path rather than the SAMPLE_RATE==16000 fast path."""
    await require_ready("whisper_asr")
    _text, audio, sample_rate = synth_speech
    transcript = await _asr_infer(grpc_client, audio, sample_rate)
    assert isinstance(transcript, str)
