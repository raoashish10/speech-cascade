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

Protocol (line-based over stdin/stdout, one exchange per Triton execute()
call -- which may carry 1..max_batch_size requests once dynamic_batching is
enabled in config.pbtxt):
  in:  a JSON array of texts to synthesize, e.g. ["text one", "text two"]
  out: a JSON array of per-item results, same length/order as the input,
       each either {"ok": true, "path": "<wav path>", "sr": <int>} or
       {"ok": false, "error": "<message>"}
The whole batch shares one "elapsed" figure (see below) rather than a
fabricated per-item breakdown, since all items ran through one shared T3
decode pass -- there's no meaningful way to attribute wall-clock time to
one item over another within that pass.
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
import json
import os
import sys
import tempfile
import time
from contextlib import contextmanager

import soundfile as sf
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Optional. chatterbox-turbo ships a built-in voice (conds.pt, one of the
# files ChatterboxTurboTTS.from_pretrained() pulls from the Hub), and
# from_local() loads it into self.conds automatically when present. So a
# reference clip is only needed to override that default with a specific
# cloned voice -- see the startup logic below.
REF_AUDIO = sys.argv[1] if len(sys.argv) > 1 else ""

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


# Opt-in CUDA graph decode: set CHATTERBOX_USE_CUDA_GRAPH=1 to replay T3's
# per-step decode forward as a captured CUDA graph instead of calling it
# eagerly each step (see t3.py's inference_turbo docstring and
# chatterbox_profiling/ for why -- the decode loop profiled as kernel-
# launch-overhead-bound). Default off: falls back to the already-validated
# eager batched path. First real call after this is enabled pays a one-time
# graph-capture cost; every call after that reuses the captured graph.
_USE_CUDA_GRAPH = os.environ.get("CHATTERBOX_USE_CUDA_GRAPH", "0") == "1"

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
#
# Falling back to the built-in voice rather than requiring a clip is
# deliberate. prepare_conditionals() used to be called unconditionally on a
# path that is a DEPLOYMENT ARTIFACT -- not in this repo, not in the image --
# so a container started without one got as far as loading the whole model
# and then died, taking chatterbox_tts to UNAVAILABLE. Confirmed on a
# Runpod pod from the published image: three models READY, this one not.
# chatterbox-turbo ships conds.pt for exactly this case, so the default
# voice costs nothing and makes the image self-sufficient.
#
# A configured clip still wins, so deployments that set ref_audio_path keep
# the voice they had. Note prepare_conditionals() asserts the clip is longer
# than 5 seconds; a shorter one is a hard failure, not a fallback, because
# silently ignoring a clip someone deliberately configured would be worse
# than refusing to start.
if REF_AUDIO and os.path.exists(REF_AUDIO):
    model.prepare_conditionals(REF_AUDIO)
    print(f"[voice] cloned from reference clip: {REF_AUDIO}", file=sys.stderr, flush=True)
elif REF_AUDIO:
    raise SystemExit(
        f"ref_audio_path is set to {REF_AUDIO!r} but that file does not exist. "
        "Leave it empty to use chatterbox's built-in voice, or mount the clip."
    )
else:
    if model.conds is None:
        raise SystemExit(
            "No ref_audio_path configured and this checkpoint has no built-in "
            "conds.pt, so there is no voice to speak with. Supply a reference "
            "clip longer than 5 seconds via ref_audio_path."
        )
    print("[voice] using chatterbox's built-in voice (no ref_audio_path set)",
          file=sys.stderr, flush=True)

reply("WORKER_READY")

for line in sys.stdin:
    line = line.strip()
    if not line or line == "QUIT":
        break
    try:
        texts = json.loads(line)
        if not isinstance(texts, list) or not texts:
            raise ValueError(f"expected a non-empty JSON array of texts, got {line!r}")

        t0 = time.time()
        # The installed chatterbox-tts==0.1.7 (PyPI) release's ChatterboxTurboTTS
        # exposes only a single-item generate(text, ...) -- no generate_batch,
        # no use_cuda_graph -- unlike what an earlier revision of this worker
        # assumed (whatever venv that was validated against had a patched/
        # forked build with those extras; the public package doesn't). Loops
        # single-item generate() calls instead: strictly more "unaccelerated"
        # than a hypothetical batched-eager path anyway, which is the whole
        # point of this backend for the soak-test comparison (see
        # chatterbox_tts/1/model.py's CHATTERBOX_BACKEND comment).
        #
        # generate() builds no autograd graph on purpose -- it's inference-only --
        # but never wraps itself in torch.no_grad()/inference_mode(), so PyTorch
        # still tracks every intermediate tensor's grad-fn through the T3
        # backbone, S3Gen vocoder, AND the internal watermarker call by
        # default. Measured live (docs/chatterbox-host-ram-leak.md, and
        # independently rediscovered in chatterbox_worker_vllm.py's own fix):
        # host RSS grows unbounded per request without this wrapper. Wrapping
        # the whole loop in inference_mode() covers every one of those calls
        # (thread-local context, inherited by everything generate() calls
        # internally) and cuts steady-state growth to a small, plateauing
        # amount instead of growing forever.
        wavs = []
        with torch.inference_mode(), nvtx_range("chatterbox.generate"):
            for text in texts:
                try:
                    wav = model.generate(text)
                    wavs.append(wav.squeeze(0).detach().cpu().numpy())
                except Exception:
                    wavs.append(None)
        elapsed = time.time() - t0

        results = []
        for wav_np in wavs:
            if wav_np is None:
                results.append({"ok": False, "error": "generation failed for this item"})
                continue
            fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="chatterbox_")
            os.close(fd)
            sf.write(out_path, wav_np, model.sr)
            results.append({"ok": True, "path": out_path, "sr": model.sr})
        # "OK " + one JSON blob (never "OK <fields...>" split on spaces --
        # the JSON payload itself contains spaces) so model.py just does
        # status, payload = result.split(" ", 1).
        reply("OK " + json.dumps({"elapsed": elapsed, "results": results}))

        if _profiler is not None:
            _profiled_steps += len(texts)
            # Not sent via reply() -- that would corrupt the one-line-per-
            # request protocol the parent process parses (see module
            # docstring); this plain stderr line is filtered out the same
            # way library warning noise already is.
            if _profiled_steps >= _PROFILE_STEPS:
                _profiler.stop()
                # export_chrome_trace is opt-in (default off): with_stack=True
                # above makes it record a full Python call stack per event, and
                # over _PROFILE_STEPS=20 real generate() calls that produced a
                # 4.3GB trace -- and even 3 steps produced 1.07GB -- both well
                # past what chrome://tracing/Perfetto open, and large enough to
                # OOM-kill this worker during export on a loaded instance (see
                # chatterbox_profiling/accelerated/'s first attempt). The
                # key_averages table below is unaffected by this flag and is
                # the actually-comparable artifact; see deploy/PROFILING.md.
                trace_path = None
                if os.environ.get("CHATTERBOX_TORCH_PROFILE_EXPORT_TRACE") == "1":
                    trace_path = os.path.join(_PROFILE_DIR, f"chatterbox_trace_{os.getpid()}.json")
                    _profiler.export_chrome_trace(trace_path)
                # Printed directly to the log, not just exported to a file a
                # headless Vast.ai instance has no easy way to open: with
                # with_stack=True (set above), group_by_stack_n attributes
                # each op's time to its Python call stack, so T3
                # (t3.py: T3.inference_turbo) vs S3Gen (s3gen.py /
                # flow_matching.py: basic_euler/solve_euler) show up as
                # separate rows without needing to load the chrome trace
                # anywhere. See deploy/PROFILING.md.
                summary = _profiler.key_averages(group_by_stack_n=5).table(
                    sort_by="self_cuda_time_total", row_limit=30
                )
                trace_note = f"chrome trace saved to {trace_path}" if trace_path else "chrome trace export skipped (set CHATTERBOX_TORCH_PROFILE_EXPORT_TRACE=1 to enable)"
                print(
                    f"[profiling] {_PROFILE_STEPS} steps captured, {trace_note}"
                    f"\n[profiling] top ops by self CUDA time "
                    f"(grouped by call stack):\n{summary}",
                    file=sys.stderr, flush=True,
                )
                _profiler = None
    except Exception as e:
        reply(f"ERR {e}")
