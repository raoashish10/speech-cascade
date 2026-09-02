"""Integration test for voice_pipeline end-to-end (audio in -> audio out,
one call), against the live Triton server. See tests/integration/conftest.py.

voice_pipeline has max_batch_size: 0 -- unlike the three models it wraps,
tensors here carry NO leading batch dimension (see
triton_model_repo/voice_pipeline/1/model.py's docstring on the
batch-dim-add-on-the-way-in/strip-on-the-way-out asymmetry, and
scripts/load_test.py's build_request("voice_pipeline") for the reference
shapes this test matches).
"""

import numpy as np
import tritonclient.grpc.aio as grpcclient

CLIENT_TIMEOUT_S = 60.0


async def test_voice_pipeline_end_to_end(grpc_client, require_ready, synth_speech):
    for model in ("whisper_asr", "qwen_llm", "chatterbox_tts", "voice_pipeline"):
        await require_ready(model)

    _text, audio, sample_rate = synth_speech
    audio = audio.astype(np.float32)  # no leading batch dim -- max_batch_size: 0
    sr_arr = np.array([sample_rate], dtype=np.int32)

    audio_inp = grpcclient.InferInput("AUDIO_SAMPLES", audio.shape, "FP32")
    audio_inp.set_data_from_numpy(audio)
    sr_inp = grpcclient.InferInput("SAMPLE_RATE", sr_arr.shape, "INT32")
    sr_inp.set_data_from_numpy(sr_arr)

    result = await grpc_client.infer(
        model_name="voice_pipeline",
        inputs=[audio_inp, sr_inp],
        outputs=[
            grpcclient.InferRequestedOutput("TRANSCRIPT"),
            grpcclient.InferRequestedOutput("GENERATED_TEXT"),
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ],
        client_timeout=CLIENT_TIMEOUT_S,
    )

    transcript = result.as_numpy("TRANSCRIPT").flatten()[0]
    transcript = transcript.decode("utf-8") if isinstance(transcript, bytes) else transcript
    generated = result.as_numpy("GENERATED_TEXT").flatten()[0]
    generated = generated.decode("utf-8") if isinstance(generated, bytes) else generated
    out_audio = result.as_numpy("AUDIO_SAMPLES").astype(np.float32).flatten()
    out_sr = int(result.as_numpy("SAMPLE_RATE").flatten()[0])

    assert isinstance(transcript, str) and transcript.strip() != "", "ASR stage: transcript non-empty"
    assert isinstance(generated, str) and generated.strip() != "", "LLM stage: generated text non-empty"
    assert out_audio.size > 0, "TTS stage: audio output non-empty"
    assert out_sr > 0
