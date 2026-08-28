"""Persistent Chatterbox-Turbo TTS worker, run inside /venv/chatterbox (a
separate, self-contained venv from Triton's own /venv/main -- Chatterbox's
torch/torchaudio pins are not compatible with the TensorRT-LLM stack the
rest of this pipeline runs on). Spawned once by chatterbox_tts/1/model.py's
initialize() and kept alive for the life of that Triton model instance, so
the ~model-load cost is paid once, not per request -- same architecture
already built and validated (1039-turn, 0-error, 15-minute soak) in this
project's own TTS-replacement investigation
(docs/tts-replacement-investigation.md); this is that same worker adapted to
speak Triton's request/response shape instead of a plain stdin loop.

Protocol (line-based over stdin/stdout, one exchange per TTS call):
  in:  raw text to synthesize
  out: "OK <path-to-wav> <sample_rate> <latency_s>"  or  "ERR <message>"
Every reply is prefixed with a sentinel (see PROTO below) and sent on
stderr, exactly like the investigation's own tts_worker.py -- library noise
(warnings, progress output) lands unpredictably on both stdout and stderr,
so the caller must filter by sentinel prefix rather than by stream choice.

The reference voice is embedded ONCE at startup via prepare_conditionals()
(not per-call as the investigation's own throwaway script did) -- generate()
reuses the cached conditioning when called without audio_prompt_path, which
is both correct (same reference voice every call, matching a fixed-voice
production TTS model the way Kokoro/Magpie use a single default_voice) and
faster (skips re-embedding the reference clip on every single turn).
"""
import os
import sys
import tempfile
import time

import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_AUDIO = sys.argv[1]

PROTO = "@@PROTO@@"


def reply(msg):
    print(f"{PROTO} {msg}", file=sys.stderr, flush=True)


model = ChatterboxTurboTTS.from_pretrained(device="cuda")
# Embed the reference voice once; subsequent generate() calls with no
# audio_prompt_path reuse this cached conditioning (see module docstring).
model.prepare_conditionals(REF_AUDIO)
reply("WORKER_READY")

for line in sys.stdin:
    text = line.strip()
    if not text or text == "QUIT":
        break
    try:
        t0 = time.time()
        wav = model.generate(text)
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="chatterbox_")
        os.close(fd)
        ta.save(out_path, wav, model.sr)
        elapsed = time.time() - t0
        reply(f"OK {out_path} {model.sr} {elapsed:.4f}")
    except Exception as e:
        reply(f"ERR {e}")
