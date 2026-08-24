# Test fixtures

## `vad_sample_16k.wav`

A short (~3.2s), 16kHz mono, 16-bit PCM WAV: ~0.6s silence, then real
synthesized speech ("Testing one two three."), then ~1.2s silence. Used by
`tests/unit/test_vad.py` to exercise `UtteranceVAD` against a genuine
silence -> speech -> silence utterance shape.

Generated **offline**, directly against this project's own Kokoro-82M ONNX
weights (`/workspace/speech-cascade-inference/models/kokoro-82m/`) -- not
through the live Triton server, so regenerating it never touches the
running service or its loaded models. Regenerate with:

```bash
LD_LIBRARY_PATH="/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib" \
/venv/main/bin/python - <<'PY'
import librosa, numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

SR = 16000
kokoro = Kokoro(
    "/workspace/speech-cascade-inference/models/kokoro-82m/onnx/model.onnx",
    "/workspace/speech-cascade-inference/models/kokoro-82m/voices-v1.0.bin",
)
samples, sr = kokoro.create("Testing one two three.", voice="af_heart", speed=1.0, lang="en-us")
speech_16k = librosa.resample(samples.astype(np.float32), orig_sr=sr, target_sr=SR)
clip = np.concatenate([
    np.zeros(int(0.6 * SR), dtype=np.float32),
    speech_16k,
    np.zeros(int(1.2 * SR), dtype=np.float32),
]).astype(np.float32)
sf.write("tests/fixtures/vad_sample_16k.wav", clip, SR, subtype="PCM_16")
PY
```

The `LD_LIBRARY_PATH` override works around the cuBLAS 12-vs-13 mismatch
documented in the main README ("Environment quirks fixed to make this run
bare-metal", #7) -- `onnxruntime-gpu`'s CUDA EP needs the cu12 cuBLAS/cuDNN
shipped inside `/venv/main`'s own `nvidia-*` pip packages, not the system's
cu13 toolkit. The TensorRT EP will still fail to load and fall back to CUDA
EP with a warning printed to stderr -- expected, harmless, matches the "Not
yet done" note in the main README.
