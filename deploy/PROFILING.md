# Profiling: torch.profiler and Nsight Systems (nsys)

Two different tools for two different questions, added so a future
performance investigation (like the `nemotron-batch-size-scaling` or
`kv-cache-investigation` writeups referenced from `docs/README.md`, but with
real kernel-level evidence instead of hand-computed bandwidth math) doesn't
have to instrument the pipeline from scratch. This is operational reference
material, not a narrative investigation writeup, so — like `deploy/REBUILD.md`
and `deploy/README.md` — it stays here rather than under `docs/` (see
`docs/README.md` for why most of `docs/` moved to S3-only).

## Why two tools, not one

Of the four Triton models, only `chatterbox_tts` (via `chatterbox_worker.py`)
runs as plain PyTorch — its `model.generate(...)` call executes real,
Python-visible ATen ops that `torch.profiler` can break down op by op.

`qwen_llm` and `whisper_asr` both run through TensorRT-LLM: the actual
generation happens inside a compiled engine's CUDA graph replay (or, for
`qwen_llm`, a separate MPI worker process — see README.md's "Process
architecture" section), not as Python-visible PyTorch ops. `torch.profiler`
attached to the Triton stub process would show one big opaque
`generate_async`/`process_batch` call and nothing useful inside it.

**nsys operates below the framework**, at the CUDA driver level, so it sees
real kernel launches and timings regardless of which backend produced them
— TensorRT-LLM's engines, ONNX Runtime (if ever reintroduced), or plain
PyTorch. It's the only tool here that can put all three GPU-bound stages
(`whisper_asr`, `qwen_llm`, `chatterbox_tts`) on one shared timeline,
relative to each other.

| | `torch.profiler` | `nsys` |
|---|---|---|
| Sees inside TensorRT-LLM engines (`qwen_llm`, `whisper_asr`) | No — opaque | Yes |
| Sees inside plain PyTorch (`chatterbox_tts`) | Yes, op-by-op (ATen + CUDA kernel) | Yes, kernel-level only |
| Shows all pipeline stages on one timeline | No (per-process) | Yes |
| Output | Chrome trace JSON (`chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev)) | `.nsys-rep` (`nsys-ui` GUI or `nsys stats` text summary) |
| Where it's wired up | `chatterbox_worker.py` only | NVTX ranges in all four `triton_model_repo/*/1/model.py` + `chatterbox_worker.py` |

## torch.profiler: chatterbox_tts only

Set two env vars before the worker subprocess starts (i.e. before
`chatterbox_tts` loads in Triton — restart/reload the model to pick up a
change):

```bash
export CHATTERBOX_TORCH_PROFILE_STEPS=20   # profile the next 20 generate() calls, then stop
export CHATTERBOX_TORCH_PROFILE_DIR=/tmp/chatterbox_torch_profile  # default shown
```

`CHATTERBOX_TORCH_PROFILE_STEPS` defaults to `0` (disabled, zero overhead —
`torch.profiler.profile()` is never constructed). It's a bounded run, not
"profile forever": `torch.profiler`'s own bookkeeping (`record_shapes`,
`profile_memory`, `with_stack` are all enabled) has real overhead, and this
worker stays alive for the life of the Triton model instance, so nothing
here auto-stops on its own otherwise.

Once the Nth call completes, the worker exports a Chrome trace to
`${CHATTERBOX_TORCH_PROFILE_DIR}/chatterbox_trace_<pid>.json` and logs a
plain (non-protocol) line to stderr confirming it — check
`/var/log/portal/speech-cascade-triton.log` or wherever the supervisor
service's stderr lands. Open the trace at `chrome://tracing` (load the
file) or https://ui.perfetto.dev for a nicer UI — either shows the
CPU/CUDA op timeline, including which op(s) inside `model.generate()`
(T3 backbone vs. S3Gen vocoder) actually dominate a call.

**Restart cost**: reloading `chatterbox_tts` respawns the worker subprocess
(cheap — no TensorRT-LLM import, unlike `qwen_llm`), so toggling this on/off
doesn't cost the ~5-6 minute full-stack restart.

## nsys: the whole pipeline

`scripts/profile_nsys.sh` wraps a full Triton launch (same environment
setup and sequential model-load order as
`deploy/supervisor/speech-cascade-triton.sh`) under `nsys profile`. Read the
script's own header comment before running it — in particular:

- It starts a **real** server on the real ports (18000-18002) and loads all
  four models onto the GPU. Stop the actual supervisor-managed service
  first (`supervisorctl stop speech-cascade-triton`) — don't run both at
  once.
- `chatterbox_tts`'s real GPU work happens in a separate
  `/venv/chatterbox/bin/python3` subprocess (see README.md's "Process
  architecture" section), not inside `tritonserver` itself.
  `--trace-fork-before-exec=true` is intended to make nsys follow that
  child process, but **this has not been verified end-to-end** on this
  project's actual nsys version — if `chatterbox_tts`'s CUDA activity is
  missing from a report, attach nsys to that worker's PID directly instead
  (see the script's header for the exact command).

```bash
supervisorctl stop speech-cascade-triton
scripts/profile_nsys.sh
# in another terminal, once all four models report loaded:
python3 scripts/load_test.py --concurrency 4 --total-requests 40
# back in the first terminal: Ctrl-C to stop the server and finalize the report
```

Copy the resulting `<output>.nsys-rep` off the instance and open it with
`nsys-ui` (GUI) or run `nsys stats <output>.nsys-rep` for a text summary of
kernel/NVTX-range durations.

### Reading the timeline: NVTX ranges

Each pipeline stage pushes an NVTX range around its actual GPU-bound work,
so nsys's timeline shows labeled spans instead of an undifferentiated wall
of CUDA kernels:

| Range | Where |
|---|---|
| `qwen_llm.generate` | `qwen_llm/1/model.py` — the full streamed `generate_async` call |
| `whisper_asr.feature_extraction` | `whisper_asr/1/model.py` — mel-spectrogram extraction, before the engine call |
| `whisper_asr.process_batch` | `whisper_asr/1/model.py` — the TensorRT-LLM encoder+decoder call |
| `chatterbox.generate` | `chatterbox_worker.py` — the T3+S3Gen `model.generate()` call |
| `voice_pipeline.whisper_asr` / `.qwen_llm` / `.chatterbox_tts` | `voice_pipeline/1/model.py` — each downstream BLS call, including that callee's own queueing, as seen from the orchestrator's side |

The `voice_pipeline.*` ranges are the ones to check first for a
whole-pipeline latency question (e.g. "which stage actually dominates a
turn under concurrency" — the same question the `voice-pipeline-queueing`
writeup referenced from `docs/README.md` answered by diffing Prometheus
metrics; nsys now gives a direct visual answer instead) — they wrap the
same span Triton's own per-model queue/compute metrics measure, but on one
shared timeline instead of four separate per-model counters.

Set `NSYS_NVTX=0` (exported in the environment before Triton starts) to
disable every NVTX range if their overhead is ever a concern. It's normally
negligible — each is a single push/pop into `nvToolsExt`, a no-op cost when
no profiler is attached and cheap even when one is.

## What this doesn't do

- No automated regression tracking or CI wiring — this is a manual,
  point-in-time investigation tool, same as `scripts/load_test.py`.
- `chatterbox_tts/1/model.py` (the Triton stub, as opposed to
  `chatterbox_worker.py`) deliberately does **not** import `torch` or push
  NVTX ranges — it stays CUDA-free on purpose, since that stub process's
  whole reason for existing is to avoid importing Chatterbox's
  torch/torchaudio stack into `/venv/main` (see that file's own docstring).
  Its IPC wait time is visible indirectly as the gap around
  `voice_pipeline.chatterbox_tts`'s range in the orchestrator's own trace.
- The `--trace-fork-before-exec` child-process caveat above — not
  confirmed working on this project's actual instance/nsys version yet.
