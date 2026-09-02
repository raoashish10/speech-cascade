"""Persistent TTS worker: reads one line of text from stdin per turn,
synthesizes it, writes "OK <latency_s>" to stdout. Kept alive across turns
so model-load cost is paid once, matching how a real service would run.

Used as the TTS half of pipeline_stress_test.py's composed-pipeline soak
test harness -- see docs/tts-replacement-investigation.md for the results
this produced for Chatterbox-Turbo, F5-TTS, and XTTS-v2. Run under the
candidate's own venv (each of chatterbox/xtts/f5tts needs an isolated venv;
see deploy/requirements-chatterbox.txt and docs/tts-replacement-investigation.md
for how XTTS-v2's was built and the compatibility patches it needed).

Usage: python3 tts_worker.py <backend> <ref_audio_path> [out_wav_path]
  backend: chatterbox | xtts | f5tts
"""
import sys
import tempfile
import time

backend = sys.argv[1]
ref_audio = sys.argv[2]
OUT_WAV = sys.argv[3] if len(sys.argv) > 3 else tempfile.mktemp(suffix=".wav", prefix="tts_worker_")

if backend == "chatterbox":
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    import torchaudio as ta
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    model.prepare_conditionals(ref_audio)

    def synth(text):
        wav = model.generate(text)
        ta.save(OUT_WAV, wav, model.sr)

elif backend == "xtts":
    # torch>=2.6 flipped torch.load's weights_only default to True; Coqui's
    # checkpoint loader doesn't opt out, and the XTTS-v2 checkpoint pickles
    # config classes that aren't in torch's default safe-globals allowlist.
    # Registering them explicitly avoids needing weights_only=False (this
    # checkpoint is Coqui's own official release, not an untrusted source).
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

    def synth(text):
        tts.tts_to_file(text=text, speaker_wav=ref_audio, language="en", file_path=OUT_WAV)

elif backend == "f5tts":
    from f5_tts.api import F5TTS
    model = F5TTS()

    def synth(text):
        model.infer(ref_file=ref_audio, ref_text="Testing 1, 2, 3.", gen_text=text, file_wave=OUT_WAV)

else:
    raise ValueError(f"unknown backend {backend}")

# Protocol replies are tagged with a distinctive sentinel prefix that
# should never occur in ordinary output, and sent on stderr. This isn't
# just "use the other stream" -- library noise shows up on BOTH stdout
# (e.g. chatterbox-tts's own "loaded PerthNet...") and stderr (tqdm
# progress bars and the warnings module both default to stderr), so
# picking a stream alone doesn't isolate the channel. The orchestrator
# must filter for this exact prefix and discard everything else it reads,
# regardless of which stream it came from.
TAG = "@@PROTO@@"
print(f"{TAG} WORKER_READY", file=sys.stderr, flush=True)

for line in sys.stdin:
    text = line.strip()
    if not text or text == "QUIT":
        break
    try:
        t0 = time.time()
        synth(text)
        elapsed = time.time() - t0
        print(f"{TAG} OK {elapsed:.4f}", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"{TAG} ERR {e}", file=sys.stderr, flush=True)
