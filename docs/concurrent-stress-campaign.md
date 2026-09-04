# Concurrent stress campaign against the current stack: chatterbox_tts's real host-RAM leak, found and fixed

## Verdict

**Found and fixed a real, severe bug: `chatterbox_tts`'s worker leaks host
RAM without bound, ~3.6MB per request, because `generate()` is never
wrapped in `torch.no_grad()`/`inference_mode()`.** Over a 30-minute
sustained `voice_pipeline` run (5,600 requests) this pushed the container
from a healthy baseline to its ~30GB RAM ceiling, triggered real swap
thrashing (one process alone racked up **1.5TB of disk reads**), and
produced a cluster of request timeouts right at the point of exhaustion —
this is a live, current-stack analogue of `kokoro-tts-capacity-fix.md`'s
finding, just in host RAM instead of GPU VRAM, and on a different (PyTorch
subprocess, not ONNX Runtime) runtime, exactly the thing this task asked
to check rather than assume immunity from. **Fixed with a one-line change**
(wrap the `generate()` call in `torch.inference_mode()`), verified two
ways: an isolated before/after A/B test (steady-state growth drops from
~3.6MB/request to ~0.1MB/request and plateaus within ~100 requests instead
of growing forever) and a live 10-minute/1,840-request sustained run
through the real Triton stack post-fix (RSS grew ~27MB **total**, zero
failures).

Everything else in this campaign came back clean: **zero GPU VRAM growth
anywhere** (unlike kokoro's issue, which was VRAM), **zero failures** in
every concurrency-sweep and overlapping-multi-client-contention test (up
to 16 real concurrent chains — the specific condition that triggered
kokoro's 24.3% failure rate), and `chatterbox_tts`'s real capacity
ceiling is now measured for the first time: **~3.05 req/s at
`instance_group.count: 1`, ~4.17 req/s at `count: 2`** (a real but
sublinear ~37% gain — GPU compute-bound, not instance-count-bound) —
**not shipped as the new default**, because this GPU doesn't have
comfortable VRAM headroom for it alongside the other three models (see
below).

## 1. Concurrency sweep, all four models (baseline)

`scripts/load_test.py`, concurrency 1/4/8, n=100 per model per level
(1,200 requests total):

| Model | Concurrency | Successes | p50 latency | Server avg_queue |
|---|---|---|---|---|
| whisper_asr | 1 / 4 / 8 | 100/100 each | 0.056s / 0.014s / 0.025s | 50ms / 1.9ms / 44ms |
| qwen_llm | 1 / 4 / 8 | 100/100 each | 0.073s / 0.081s / 0.081s | 0.02ms / 0.11ms / 0.44ms |
| chatterbox_tts | 1 / 4 / 8 | 100/100 each | 0.327s / **1.310s** / **2.638s** | 0.01ms / **962ms** / **2229ms** |
| voice_pipeline | 1 / 4 / 8 | 100/100 each | 0.383s / **1.248s** / **2.494s** | 0.01ms / — / — |

Zero failures anywhere. `chatterbox_tts`'s queue time growing linearly
with concurrency while its own `avg_compute` stays flat (~330ms) is the
expected signature of `instance_group.count: 1` — pure serialization, one
request processed at a time, everything else queues. This matches the
known gap already flagged in `chatterbox_tts/config.pbtxt`'s own comment;
see §4 for the actual capacity measurement.

## 2. The real finding: chatterbox_tts leaks host RAM, unbounded

### How it surfaced

A 30-minute sustained `voice_pipeline` run at concurrency 4 (chosen as a
realistic sustained-production proxy) ran clean for ~28 minutes — 5,560
requests, zero failures, GPU memory flat at 13,267 MiB the entire time —
then, in its final batch, `avg_compute` jumped 3.5x (1207ms → 4307ms) and
7/40 requests failed with `DEADLINE_EXCEEDED`. Checking the box
immediately after: **host RAM was at 29.63GB used against a 29.76GB
cgroup limit**, with only ~90MB available, and `free -h` showed active
swap usage. `chatterbox_tts`'s worker subprocess alone held **~20.7GB
resident** (`VmRSS`) — for a model whose own README-documented VRAM
budget is a few GB. `vmstat` during a follow-up probe showed `si`/`so`
(swap in/out) rates around 100K/s and 35-47% I/O-wait; `/proc/<pid>/io`
on the worker showed **1.56TB of cumulative disk reads** — genuine,
severe thrashing, not just high memory.

Sending 50 more requests directly at `chatterbox_tts` while the box was
in this state: **all 50 failed**, Triton's own `avg_compute` metric read
**887,995ms** (887 seconds) averaged — the model was still nominally
"working," just catastrophically slow under memory pressure. Recovering
required force-killing the stuck worker (it was in D-state / disk-sleep,
unresponsive to normal signals) and a full Triton restart; the restart
itself hit a known-but-previously-undocumented issue where the old
`tritonserver` process survives as an orphan holding the ports (see §5).

### Root cause, isolated and confirmed

`chatterbox_worker.py`'s per-request loop calls `model.generate(text)`
with no `torch.no_grad()`/`inference_mode()` anywhere — not in this
project's own code, and not inside `chatterbox/tts_turbo.py`'s
`ChatterboxTurboTTS.generate()` either (checked directly against the
installed library source: no such context manager anywhere in the file,
despite the call being pure inference). Without it, PyTorch tracks a full
autograd graph through the T3 backbone and S3Gen vocoder on every call,
by default.

Isolated before/after, driving the worker script directly (bypassing
Triton) with a fixed test string, sampling `VmRSS` after each 20-request
batch:

| Batch | Unpatched RSS | Patched (`inference_mode`) RSS |
|---|---|---|
| 0 (startup) | 2,410 MB | 2,411 MB |
| 1 | 3,007 MB (+597MB, allocator warmup) | 2,938 MB (+527MB, same warmup) |
| 2 | 3,085 MB (+78MB) | 2,942 MB (+4MB) |
| 3 | 3,159 MB (+74MB) | 2,947 MB (+5MB) |
| 4 | 3,235 MB (+76MB) | 2,949 MB (+2MB) |
| 5 | 3,300 MB (+65MB) | 2,951 MB (+2MB) |
| 6 | 3,370 MB (+70MB) | 2,951 MB (+0MB) |
| 7 | 3,449 MB (+79MB) | 2,955 MB (+4MB) |
| 8 | 3,518 MB (+70MB) | 2,955 MB (+0MB) |

Unpatched: steady-state growth **~3.6MB/request** (batches 2-8 averaged),
linear, no sign of plateauing across 160 requests — extrapolated to the
30-minute run's 5,600 requests, that's ~20GB, matching the ~20.7GB
actually observed almost exactly. Patched: growth drops to **~0.1MB/request**
and the last two batches show **zero measurable growth** — it plateaus.
Isolated the leak to `generate()` specifically (not `ta.save()`/torchcodec
downstream of it) by first running this comparison with a broken
`LD_LIBRARY_PATH` that made every `ta.save()` call fail outright — `generate()`
still ran to completion each time, and RSS still grew at the unpatched
rate, confirming the growth happens inside `generate()` itself.

### Fix

```python
with torch.inference_mode():
    wav = model.generate(text)
```

in `triton_model_repo/chatterbox_tts/1/chatterbox_worker.py`. One line,
plus the `import torch` it needed. Verified live, post-fix, through the
real Triton stack: 15/15 `tests/integration` pass, and a 10-minute/1,840-request
sustained run against `chatterbox_tts` directly shows **zero failures**
and RSS growing only ~27MB **total** across the entire run (not
per-request) — essentially flat.

### What this doesn't close out

- *Why* `generate()`'s autograd graph specifically manifests as growing
  **host** RAM rather than GPU VRAM (which stayed completely flat
  throughout, even in the unpatched leak) wasn't traced further at the
  PyTorch-internals level — plausible (CPU-side tensor bookkeeping,
  autograd graph node objects, tokenizer output tensors) but not proven
  down to the exact allocation site. The fix is confirmed effective either
  way.
- Whether `chatterbox/tts_turbo.py` (a third-party pip package, MIT
  licensed) should itself wrap `generate()` in `inference_mode()` upstream
  wasn't pursued (would mean filing/patching upstream) — the fix here is
  scoped to this project's own call site, which is sufficient to close the
  leak for this deployment.
- The very first allocator-warmup jump (~550-600MB in batch 1, present in
  *both* patched and unpatched runs) wasn't investigated — it's one-time
  and harmless, not part of the unbounded-growth problem, but its exact
  cause (first-call kernel/algorithm selection, first-call tensor shape
  cache population, etc.) is unconfirmed.

## 3. Overlapping multi-client contention (kokoro's actual trigger)

`kokoro-tts-capacity-fix.md`'s own key methodology finding: the 24.3%
failure rate never reproduced from a single client's `--concurrency` flag
alone — only from **multiple independent overlapping clients** hitting
the stack at once (closer to real multi-agent/multi-caller production
load). Reproduced that exact test shape here:

| Setup | Real concurrent chains | Result |
|---|---|---|
| 2 independent `load_test.py --concurrency 4` processes | 8 | 80/80 success, 0 failures |
| 4 independent `load_test.py --concurrency 4` processes | 16 | 160/160 success, 0 failures |

No failure mode surfaced even at 16 real concurrent chains (queue times
scale up as expected — `chatterbox_tts`'s serialization is still the
dominant cost — but nothing times out). The most plausible reason this
doesn't reproduce kokoro's problem: `load_test.py`'s client timeout is
60s, versus the 750ms admission-control timeout kokoro was tuned against
at the time of that investigation — there's much more slack before queued
requests actually fail here. **Caveat**: this means queue depth is a live,
just-not-yet-triggered risk — at high enough sustained concurrency for
long enough, `chatterbox_tts`'s pure-serialization queue could still grow
past 60s. Not observed in this campaign, but not proven impossible either.

## 4. chatterbox_tts's real capacity ceiling — measured for the first time

Isolated (other three models unloaded, so GPU compute/VRAM contention
from them doesn't confound the measurement), `scripts/load_test.py
--model chatterbox_tts` at increasing concurrency:

| `instance_group.count` | Concurrency 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 1 (current default) | 3.05 req/s | — | 3.09 req/s | 3.06 req/s | 3.02 req/s |
| 2 | 2.89 req/s | 4.16 req/s | 4.19 req/s | 4.16 req/s | 4.18 req/s |

`count: 1` throughput is flat regardless of concurrency — pure
serialization, exactly as its own config comment predicted. `count: 2`
raises the ceiling to ~4.17 req/s, a real ~37% improvement, but far short
of the ~2x a second fully-independent worker would suggest — consistent
with the two workers sharing the same physical GPU's compute units (each
gets its own VRAM allocation and CUDA context, but they still serialize
on the SMs), so this is GPU-compute-bound, not purely
instance-count-bound.

**Not shipped as the new default** (unlike `kokoro-tts-capacity-fix.md`'s
count 3→4, which was funded by VRAM freed from fixing a wasteful ONNX
Runtime default): each `chatterbox_tts` worker holds ~3.5GB VRAM, and this
16GB GPU has only ~2.6GB free with all four models loaded at their current
sizes — not enough headroom for a second worker without shrinking
something else's KV-cache/memory budget first. `count: 2` was measured by
temporarily unloading `qwen_llm`/`whisper_asr`/`voice_pipeline`, not in
the full-stack configuration. If more capacity is needed, the honest
options are: reduce `qwen_llm`'s `kv_cache_config.free_gpu_memory_fraction`
(currently 0.2) to free VRAM for a second `chatterbox_tts` worker, or move
to a larger GPU. Left as a decision for whoever needs the extra capacity,
not made here.

## 5. Smaller findings along the way

- **Orphan `tritonserver` process survives `supervisorctl restart`.**
  Confirmed twice in this campaign: after `supervisorctl restart
  speech-cascade-triton`, the *old* `tritonserver` process was still alive
  and holding some/all of ports 18000-18002, causing the new instance to
  fail to bind and the restart to silently not take effect until the old
  process was manually `kill -9`'d. Root cause not fully chased (likely:
  `speech-cascade-triton.sh` backgrounds `tritonserver` with `&` and
  `wait`s on it, and supervisord's SIGTERM to the wrapper script doesn't
  propagate to the backgrounded child without an explicit trap) — flagged
  here as a real operational gap, not fixed in this pass. Anyone
  restarting this service should verify with `ps aux | grep tritonserver`
  and `ss -tlnp | grep 1800` after a restart, not just trust
  `supervisorctl status`.
- **No repetition-loop / degenerate-output failures observed** in any TTS
  or LLM output during this campaign, cross-referenced against Task 1's
  classic-backend collapse findings — consistent with Task 1's conclusion
  that production's config isn't affected.
- **`scripts/pipeline_stress_test.py` was not used** for this campaign —
  it's built for isolated single-candidate TTS comparison (loads a raw
  Whisper TensorRT-LLM engine directly and talks to a separate
  OpenAI-compatible LLM server, not through Triton at all), a different
  deployment shape than the current 4-model Triton stack. `scripts/load_test.py`
  is the correct tool for testing what's actually deployed, and is what
  this entire campaign used.

## Files touched

- `triton_model_repo/chatterbox_tts/1/chatterbox_worker.py` — the
  `torch.inference_mode()` fix, with an inline comment pointing back to
  this doc.
- `docs/concurrent-stress-campaign.md` — this document.
