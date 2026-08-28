"""Standalone smoke test for chatterbox_tts, outside Triton. Run with
/venv/chatterbox/bin/python3 (this project's TensorRT-LLM venv, /venv/main,
does not have chatterbox-tts installed and is not compatible with its
torch/torchaudio pins -- see triton_model_repo/chatterbox_tts/1/model.py's
docstring for why the Triton backend runs this as a subprocess instead)."""

import time

import torchaudio as ta

from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_AUDIO = "/workspace/speech-cascade-inference/scripts/pipeline_output.wav"

t0 = time.time()
model = ChatterboxTurboTTS.from_pretrained(device="cuda")
model.prepare_conditionals(REF_AUDIO)
print(f"Chatterbox-Turbo load took {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
wav = model.generate("Hello, this is a test of the text to speech pipeline.")
print(f"TTS generation took {time.time()-t0:.1f}s, sample_rate={model.sr}, samples={wav.shape[-1]}", flush=True)

ta.save("/workspace/speech-cascade-inference/test_tts_output.wav", wav, model.sr)
print("Wrote /workspace/speech-cascade-inference/test_tts_output.wav", flush=True)
print("DONE", flush=True)
