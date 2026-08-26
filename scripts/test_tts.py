import time

import soundfile as sf
import torch

from nemo.collections.tts.models import MagpieTTSModel

t0 = time.time()
model = MagpieTTSModel.restore_from(
    "/workspace/speech-cascade-inference/models/magpie-tts-multilingual-357m/magpie_tts_multilingual_357m.nemo",
    map_location="cuda",
)
model = model.cuda().eval()
print(f"Magpie-TTS load took {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
with torch.no_grad():
    audio, audio_len = model.do_tts(
        transcript="Hello, this is a test of the text to speech pipeline.",
        language="en",
        apply_TN=False,
        use_cfg=True,
        speaker_index=4,  # Sofia
    )
samples = audio[0, : audio_len[0]].float().cpu().numpy()
sample_rate = 22050
print(f"TTS generation took {time.time()-t0:.1f}s, sample_rate={sample_rate}, samples={len(samples)}", flush=True)

sf.write("/workspace/speech-cascade-inference/test_tts_output.wav", samples, sample_rate)
print("Wrote /workspace/speech-cascade-inference/test_tts_output.wav", flush=True)
print("DONE", flush=True)
