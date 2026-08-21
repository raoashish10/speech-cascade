import time

from kokoro_onnx import Kokoro
import soundfile as sf

t0 = time.time()
kokoro = Kokoro(
    "/workspace/nemotron/kokoro-82m/onnx/model.onnx",
    "/workspace/nemotron/kokoro-82m/voices-v1.0.bin",
)
print(f"Kokoro load took {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
samples, sample_rate = kokoro.create(
    "Hello, this is a test of the text to speech pipeline.",
    voice="af_heart",
    speed=1.0,
    lang="en-us",
)
print(f"TTS generation took {time.time()-t0:.1f}s, sample_rate={sample_rate}, samples={len(samples)}", flush=True)

sf.write("/workspace/nemotron/test_tts_output.wav", samples, sample_rate)
print("Wrote /workspace/nemotron/test_tts_output.wav", flush=True)
print("DONE", flush=True)
