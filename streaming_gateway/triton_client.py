"""Async Triton gRPC client wrapper for the streaming gateway.

whisper_asr and chatterbox_tts keep max_batch_size > 0, so their wire shapes
need an explicit leading batch dim. qwen_llm is now max_batch_size: 0
and decoupled (streaming), so it takes no batch dim and uses stream_infer()
instead of a unary infer() call.
"""

import numpy as np
import tritonclient.grpc.aio as grpcclient

TRITON_URL = "localhost:18001"

_client: grpcclient.InferenceServerClient | None = None


def get_client() -> grpcclient.InferenceServerClient:
    global _client
    if _client is None:
        _client = grpcclient.InferenceServerClient(url=TRITON_URL)
    return _client


async def transcribe(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Unary call to whisper_asr. audio: 1D float32 array, one full utterance."""
    audio = audio.reshape(1, -1).astype(np.float32)
    sr = np.array([[sample_rate]], dtype=np.int32)

    inp_audio = grpcclient.InferInput("AUDIO_SAMPLES", audio.shape, "FP32")
    inp_audio.set_data_from_numpy(audio)
    inp_sr = grpcclient.InferInput("SAMPLE_RATE", sr.shape, "INT32")
    inp_sr.set_data_from_numpy(sr)

    result = await get_client().infer(
        model_name="whisper_asr",
        inputs=[inp_audio, inp_sr],
        outputs=[grpcclient.InferRequestedOutput("TRANSCRIPT")],
    )
    text = result.as_numpy("TRANSCRIPT").flatten()[0]
    return text.decode("utf-8") if isinstance(text, bytes) else text


async def generate_stream(prompt: str):
    """Decoupled streaming call to qwen_llm. Yields text_diff strings
    as they're produced."""
    arr = np.array([prompt.encode("utf-8")], dtype=np.object_)  # max_batch_size: 0 -- no batch dim
    inp = grpcclient.InferInput("PROMPT", arr.shape, "BYTES")
    inp.set_data_from_numpy(arr)

    async def _one_request():
        yield {
            "model_name": "qwen_llm",
            "inputs": [inp],
            "outputs": [grpcclient.InferRequestedOutput("GENERATED_TEXT")],
        }

    async for result, error in get_client().stream_infer(_one_request()):
        if error is not None:
            raise RuntimeError(f"qwen_llm stream error: {error}")
        text = result.as_numpy("GENERATED_TEXT").flatten()[0]
        text = text.decode("utf-8") if isinstance(text, bytes) else text
        if text:
            yield text


async def synthesize(text: str, voice: str = "default"):
    """Unary call to chatterbox_tts for one already-complete sentence.
    Returns (audio: np.ndarray[float32], sample_rate: int).

    `voice` is accepted for interface compatibility with the prior
    kokoro_tts/magpie_tts backends but is currently ignored --
    chatterbox_tts serves a single reference voice embedded at model load
    time (see triton_model_repo/chatterbox_tts/config.pbtxt's
    ref_audio_path); multi-voice selection was not built for this backend.
    """
    text_arr = np.array([[text.encode("utf-8")]], dtype=np.object_)
    voice_arr = np.array([[voice.encode("utf-8")]], dtype=np.object_)

    inp_text = grpcclient.InferInput("TEXT", text_arr.shape, "BYTES")
    inp_text.set_data_from_numpy(text_arr)
    inp_voice = grpcclient.InferInput("VOICE", voice_arr.shape, "BYTES")
    inp_voice.set_data_from_numpy(voice_arr)

    result = await get_client().infer(
        model_name="chatterbox_tts",
        inputs=[inp_text, inp_voice],
        outputs=[
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ],
    )
    audio = result.as_numpy("AUDIO_SAMPLES")
    sample_rate = int(result.as_numpy("SAMPLE_RATE").flatten()[0])
    return audio, sample_rate
