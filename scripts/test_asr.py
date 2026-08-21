import time

import librosa
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor

MODEL_DIR = "/workspace/nemotron/whisper-base"
AUDIO_PATH = "/workspace/nemotron/test_tts_output.wav"

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = ORTModelForSpeechSeq2Seq.from_pretrained(
    MODEL_DIR,
    use_merged=True,
    provider="CUDAExecutionProvider",
)
model = model.to("cuda")
print(f"Whisper load took {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
audio, sr = librosa.load(AUDIO_PATH, sr=16000)
print(f"Loaded audio: {len(audio)} samples at {sr} Hz ({time.time()-t0:.2f}s)", flush=True)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
inputs["input_features"] = inputs["input_features"].to("cuda")

t0 = time.time()
generated_ids = model.generate(inputs["input_features"])
transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"ASR generation took {time.time()-t0:.1f}s", flush=True)
print("TRANSCRIPT:", transcript, flush=True)
print("DONE", flush=True)
