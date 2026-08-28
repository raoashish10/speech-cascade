"""Integration tests for chatterbox_tts in isolation, against the live
Triton server. See tests/integration/conftest.py."""

import numpy as np
import tritonclient.grpc.aio as grpcclient

CLIENT_TIMEOUT_S = 60.0


async def _tts_infer(client, text, voice=None):
    arr = np.array([[text]], dtype=object)
    inp = grpcclient.InferInput("TEXT", arr.shape, "BYTES")
    inp.set_data_from_numpy(arr)
    inputs = [inp]
    if voice is not None:
        v_arr = np.array([[voice]], dtype=object)
        v_inp = grpcclient.InferInput("VOICE", v_arr.shape, "BYTES")
        v_inp.set_data_from_numpy(v_arr)
        inputs.append(v_inp)
    result = await client.infer(
        model_name="chatterbox_tts",
        inputs=inputs,
        outputs=[
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ],
        client_timeout=CLIENT_TIMEOUT_S,
    )
    audio = result.as_numpy("AUDIO_SAMPLES").astype(np.float32).flatten()
    sample_rate = int(result.as_numpy("SAMPLE_RATE").flatten()[0])
    return audio, sample_rate


async def test_chatterbox_tts_responds_to_real_text(grpc_client, require_ready):
    await require_ready("chatterbox_tts")
    audio, sample_rate = await _tts_infer(grpc_client, "Hello there, this is a test.")

    assert audio.size > 0, "audio output should be non-empty"
    assert sample_rate > 0
    # Sanity floor: even a short sentence at any plausible TTS sample rate
    # should produce well over a tenth of a second of audio.
    assert audio.size / sample_rate > 0.1


async def test_chatterbox_tts_works_when_voice_unspecified(grpc_client, require_ready):
    await require_ready("chatterbox_tts")
    audio, sample_rate = await _tts_infer(grpc_client, "Testing with no VOICE field at all.")
    assert audio.size > 0
    assert sample_rate > 0


async def test_chatterbox_tts_ignores_voice_field(grpc_client, require_ready):
    """chatterbox_tts serves a single reference voice embedded at model
    load time (see triton_model_repo/chatterbox_tts/config.pbtxt's
    ref_audio_path) -- VOICE is accepted for interface compatibility with
    the prior kokoro_tts/magpie_tts backends but not acted on. Passing an
    arbitrary value must not error, and should produce audio just like
    omitting it -- this is documenting current (limited) behavior, not
    asserting the two outputs are identical, since generation isn't
    deterministic call-to-call."""
    audio, sample_rate = await _tts_infer(
        grpc_client, "Testing an explicit but currently-ignored voice field.", voice="Sofia"
    )
    assert audio.size > 0
    assert sample_rate > 0
