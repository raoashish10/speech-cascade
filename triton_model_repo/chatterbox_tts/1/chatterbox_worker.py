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
from contextlib import contextmanager

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

REF_AUDIO = sys.argv[1]

PROTO = "@@PROTO@@"


def reply(msg):
    print(f"{PROTO} {msg}", file=sys.stderr, flush=True)


# See qwen_llm/1/model.py's own copy of this helper for the full rationale
# (deploy/PROFILING.md): labels this stage's GPU work in nsys's timeline.
# Unlike qwen_llm/whisper_asr (TensorRT-LLM engines -- opaque to
# torch.profiler), this worker is plain PyTorch, so it also gets full
# torch.profiler support below for op-level (ATen/CUDA kernel) breakdown,
# which nsys alone doesn't label by op name.
_NVTX_ENABLED = os.environ.get("NSYS_NVTX", "1") != "0"


@contextmanager
def nvtx_range(name):
    if _NVTX_ENABLED and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


# Opt-in torch.profiler: set CHATTERBOX_TORCH_PROFILE_STEPS=N to profile the
# next N generate() calls after startup, then export a Chrome trace and stop
# (a bounded run, not "profile forever" -- torch.profiler's own bookkeeping
# overhead is real and this worker stays alive for the life of the Triton
# model instance). Default 0 = disabled, zero overhead. See deploy/PROFILING.md
# for how to view the resulting trace.
_PROFILE_STEPS = int(os.environ.get("CHATTERBOX_TORCH_PROFILE_STEPS", "0"))
_PROFILE_DIR = os.environ.get("CHATTERBOX_TORCH_PROFILE_DIR", "/tmp/chatterbox_torch_profile")
_profiler = None
_profiled_steps = 0
if _PROFILE_STEPS > 0:
    os.makedirs(_PROFILE_DIR, exist_ok=True)
    _profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    _profiler.start()
    print(
        f"[profiling] torch.profiler started, will export after {_PROFILE_STEPS} "
        f"generate() calls to {_PROFILE_DIR}",
        file=sys.stderr, flush=True,
    )

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
        # ChatterboxTurboTTS.generate() (chatterbox/tts_turbo.py) builds no
        # autograd graph on purpose -- it's inference-only -- but never
        # wraps itself in torch.no_grad()/inference_mode(), so PyTorch still
        # tracks every intermediate tensor's grad-fn through the T3 backbone
        # and S3Gen vocoder by default. Measured live (docs/
        # chatterbox-host-ram-leak.md): host RSS grows ~3.6MB/request,
        # unbounded, with no plateau across 200+ requests -- inside
        # generate() specifically (confirmed by isolating it from the
        # ta.save() call below). Wrapping just this call in
        # inference_mode() cuts steady-state growth to ~0.1MB/request and
        # it plateaus within ~100 requests, instead of growing forever.
        with torch.inference_mode(), nvtx_range("chatterbox.generate"):
            wav = model.generate(text)
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="chatterbox_")
        os.close(fd)
        ta.save(out_path, wav, model.sr)
        elapsed = time.time() - t0
        reply(f"OK {out_path} {model.sr} {elapsed:.4f}")

        if _profiler is not None:
            _profiled_steps += 1
            # Not sent via reply() -- that would corrupt the one-line-per-
            # request protocol the parent process parses (see module
            # docstring); this plain stderr line is filtered out the same
            # way library warning noise already is.
            if _profiled_steps >= _PROFILE_STEPS:
                _profiler.stop()
                trace_path = os.path.join(_PROFILE_DIR, f"chatterbox_trace_{os.getpid()}.json")
                _profiler.export_chrome_trace(trace_path)
                print(
                    f"[profiling] {_PROFILE_STEPS} steps captured, chrome trace "
                    f"saved to {trace_path}",
                    file=sys.stderr, flush=True,
                )
                _profiler = None
    except Exception as e:
        reply(f"ERR {e}")
