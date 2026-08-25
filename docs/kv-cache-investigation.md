# nemotron_llm KV cache investigation: FP8 quantization is not the fix

## Verdict

**Not implemented.** The per-request latency bottleneck in `nemotron_llm` is
real and is memory-bandwidth-influenced, consistent with the general
"autoregressive decode is memory-bound" pattern — but the specific bytes
that dominate that traffic are the **model weights**, not the KV cache.
Quantizing the KV cache to FP8 would touch a component that measures out to
roughly **0.5% of per-step memory traffic** at this pipeline's actual
context lengths, so even a free, zero-risk version of this change would be
latency-noise. On top of that, it is **not actually a config flip** for
this deployment's serving path (TensorRT-LLM 1.2.1's classic/JIT `_TrtLLM`
backend) — the runtime knob that looks like it should do this
(`kv_cache_config.dtype`) is silently a no-op there, and the real mechanism
requires a full recalibrated `nvidia-modelopt` re-export, the same
multi-hour undertaking as the original NVFP4 weight quantization.

This document records the measurements and source-level evidence behind
that conclusion, so the "FP8 KV cache" idea flagged as follow-up work in
PR #7 doesn't get picked up again without this context.

## Method

- Live GPU utilization sampling (`nvidia-smi dmon -s u -d 1`) during real
  generation requests against the already-loaded, `READY` `nemotron_llm`
  model on this instance's RTX 5070 Ti (16GB, GDDR7, 896 GB/s peak memory
  bandwidth per NVIDIA's published spec — no in-container tool exposed bus
  width directly, so this figure is the public spec, not a local query).
- `scripts/load_test.py --model nemotron_llm` at concurrency 1, 4, 8, and
  16 against the live server, read-only (no reload), to check how
  per-request latency scales with concurrency.
- Grep-level reading of the installed TensorRT-LLM 1.2.1 source
  (`/venv/main/lib/python3.12/site-packages/tensorrt_llm/llmapi/llm_args.py`,
  `llm.py`, `models/modeling_utils.py`) for how `KvCacheConfig` and
  `QuantConfig.kv_cache_quant_algo` actually flow into the classic
  TensorRT backend's executor, rather than assuming field names/behavior.
- Reading of the real TensorRT-LLM engine build log
  (`/workspace/logs/triton_launch.log`) for `nemotron_llm`'s actual loaded
  engine size and paged-KV-cache allocation numbers, to cross-check the
  hand-computed byte estimates below against what the engine actually
  built.

No model reload was performed — this was a read-only investigation against
the already-loaded model, per the mandate to characterize the bottleneck
*before* considering any change.

## 1. Bottleneck characterization

### GPU utilization during real decode

Sampling `nvidia-smi dmon` at 1s resolution through a concurrency=1
`nemotron_llm` load-test burst:

```
# gpu     sm    mem
    0     85     55
    0     86     55
    0     82     52
    0     87     56
    0     88     56
    0     81     51
    0     88     56
    ...(steady ~77-88% sm / ~48-56% mem throughout the burst, 0/0 idle before and after)
```

`sm` (fraction of time an SM was doing anything) runs consistently higher
than `mem` (fraction of time the memory controller was active) — 80-88%
vs. 48-56%. Neither is pinned at 100%, and neither metric measures
*saturation* of its resource's peak throughput, only *busy-ness*, so this
alone doesn't prove which resource is the hard ceiling. It does rule out
one thing outright: the GPU is not sitting idle waiting on something
off-GPU (e.g. Python/Triton stub overhead dominating) — it's actively
running kernels for the large majority of decode time.

### Weight-streaming bandwidth math (the real signal)

`nemotron_llm`'s actual built TensorRT-LLM engine, per the launch log:

```
[TensorRT-LLM][INFO] Loaded engine size: 3578 MiB
```

That's ~3.75 GB of weights (NVFP4-quantized transformer layers plus the
`lm_head`/`embed_tokens` matrices, which `hf_quant_config.json` explicitly
excludes from quantization — see `"ignore": ["lm_head", "model.embed_tokens"]`
— so they stay at native precision and are read in full for every decode
step's logits computation, i.e. they're a real, non-negligible part of
this number).

At batch size 1, autoregressive decode reads essentially the *entire*
weight set once per generated token (the classic reason single-stream LLM
decode tends to be memory-bound, independent of how few FLOPs the actual
matmuls need). Measured decode rate at concurrency=1
(`scripts/load_test.py --concurrency 1 --total-requests 30 --model nemotron_llm`):

```
p50=0.679s total, TTFT p50=0.013s, max_tokens=256 (current live max_tokens;
PR #9's cap-hit data confirms every measured prompt hits the token cap at
temperature=0, so total tokens generated == max_tokens)
```

Using the currently-live `max_tokens=96` figures instead (from PR #9's own
measurement, `max_tokens=96` -> elapsed ~0.67-0.92s for 96 tokens) or this
session's concurrency=1 run scaled the same way, decode settles at roughly:

```
decode step time  ≈ 7.0 ms/token
required bandwidth = 3.75 GB / 0.007 s ≈ 535 GB/s
535 GB/s / 896 GB/s (RTX 5070 Ti peak) ≈ 60% of peak
```

60% of a GPU's advertised peak bandwidth is a large, real number — well
above what a compute-bound kernel would need, and in the range where real
achieved bandwidth (typically 60-85% of theoretical peak for well-tuned
kernels) plausibly explains most of the per-token latency. This is the
strongest evidence that per-token latency is meaningfully
memory-bandwidth-influenced.

For comparison, the compute-side (FLOPs) requirement is not close to
being the limiter: a ~4B-parameter forward pass is roughly `2 x 4e9 ≈ 8
GFLOP`/token. Even heavily discounting for FP4-dequant overhead and
non-ideal utilization, a Blackwell-class GPU's tensor-core throughput
makes 8 GFLOP/token a sub-millisecond cost in isolation — nowhere near the
observed ~7ms/token. Raw compute throughput is not the bottleneck.

**Nuance worth flagging**: NVFP4 weight quantization already cut the bytes
that must move per token by roughly 4x versus fp16 (nominally; the actual
3.75GB engine size reflects real overhead from block scale factors, the
unquantized `lm_head`/`embed_tokens`, and engine metadata, so the
effective ratio is less than a clean 4x). That is itself a
bandwidth-reducing optimization already applied to this model — the
teammate's "quantize something to save bandwidth" instinct is directionally
right, it's just already been spent on the weights, which are 100-200x
larger than the KV cache at this pipeline's context lengths (see below).

### Concurrency scaling (live-measured, confirms the memory-bound-batched-decode signature)

```
concurrency= 1: p50=0.679s
concurrency= 4: p50=0.692s   (~flat)
concurrency= 8: p50=0.713s   (~flat -- matches MAX_CONCURRENT_STREAMS / engine max_batch_size=8)
concurrency=16: p50=1.426s   (~exactly 2x the concurrency=8 number)
```

This is close to a textbook memory-bound-batched-decode signature: up to
the engine's configured `max_batch_size=8`, TensorRT-LLM's in-flight
batching serves multiple concurrent decode streams from the *same* weight
read (the weight bytes moved per step don't scale with how many sequences
are batched into that step, only the small amount of extra compute does),
so latency stays nearly flat as concurrency climbs to 8. Past 8, requests
queue behind a second full batch, and latency scales in clean multiples of
the single-batch latency rather than degrading gradually — consistent with
hitting a hard batch-size ceiling rather than a resource that degrades
smoothly with load (which is more what you'd expect if raw compute
throughput, not batch-amortized bandwidth, were the limiter).

### Conclusion for step 1

The bottleneck is real and is memory-bandwidth-influenced, via
weight-streaming during batch-limited decode — not compute-bound, and not
dominated by scheduling/launch overhead (GPU is busy 80%+ of wall time).
This part of the teammate's hypothesis is directionally correct.

## 2. Is FP8 KV cache even usable here without a fresh calibrated export?

Two independent questions, both checked against the actual installed
TensorRT-LLM 1.2.1 source rather than assumed:

### 2a. Would the benefit even be bandwidth savings, or just capacity?

From the engine build log, `nemotron_llm`'s actual paged KV cache
allocation at `max_seq_len=4096`, `tokens_per_block=32`:

```
[TensorRT-LLM][INFO] [MemUsageChange] Allocated 1.56 GiB for max tokens in paged KV cache (12800).
```

`1.56 GiB / 12800 tokens ≈ 128 KiB/token` for the whole model (32 layers,
8 KV heads, head_dim 128, fp16, K+V combined) — this matches a from-scratch
hand calculation (`2 x 32 x 8 x 128 x 2 bytes = 128 KiB/token`) almost
exactly, which is a useful sanity check that the model's real KV-cache
footprint is understood correctly here.

At this pipeline's actual usage pattern (short voice-assistant turns,
well under the 4096 max_seq_len — a ~150-token context is a generous
estimate for prompt + `max_tokens=96` output), the KV cache read at each
decode step is:

```
128 KiB/token x ~150 tokens ≈ 19.6 MB
```

versus the ~3.75 GB weight read at that same step. **KV cache traffic is
~0.5% of per-step memory traffic** at this pipeline's real context
lengths. Halving it via FP8 would save roughly 0.25 percentage points of
total per-step bytes moved — undetectable against measurement noise, let
alone worth a multi-hour re-quantization pass.

This also directly answers the capacity-vs-bandwidth question the task
raised: the *capacity* benefit (smaller KV cache footprint, letting
`free_gpu_memory_fraction` shrink or support more concurrent sequences in
the same pool) is real but not needed — this session's own load testing
(and PR #5's admission-control work) never found KV-cache *capacity* to be
the constraint at the tested concurrency, and the pool is already
deliberately capped at a conservative 0.2 fraction specifically because
this GPU is shared with ASR/TTS. The *bandwidth* benefit is the one that
would matter for latency, and it's the one that's negligible here.

### 2b. Is it actually a config-only change on this serving path?

No — verified from the installed TensorRT-LLM 1.2.1 source, not assumed:

**`KvCacheConfig.dtype` is a PyTorch-backend-only field that's silently
dropped for the classic TensorRT backend.** In
`tensorrt_llm/llmapi/llm_args.py`:

```python
# This is a pure python field, not a pybind field. It is only for the Pytorch backend.
dtype: str = Field(default="auto",
                   description="The data type to use for the KV cache.")
```

and its `_to_pybind()` method (used to build the actual C++ executor
config) does not pass `dtype` through at all — the conversion enumerates
every other field explicitly and simply omits it.

**`_TrtLLM` (what `nemotron_llm/1/model.py` imports and uses) goes through
exactly this path.** In `tensorrt_llm/llmapi/llm.py`:

```python
self._executor_config.kv_cache_config = PybindMirror.maybe_to_pybind(...)
```

**The logic that *would* translate `kv_cache_config.dtype="fp8"` into a
real effect (`quant_config.kv_cache_quant_algo = QuantAlgo.FP8`) exists
only on `TorchLlmArgs`**, not on `TrtLlmArgs` (the args class backing
`_TrtLLM`):

```python
# TorchLlmArgs only:
@model_validator(mode='after')
def sync_quant_config_with_kv_cache_config_dtype(self) -> 'TorchLlmArgs':
    ...
    elif self.kv_cache_config.dtype == 'fp8':
        self.quant_config.kv_cache_quant_algo = QuantAlgo.FP8
```

`TrtLlmArgs` (lines 2358-2768 of `llm_args.py`) has no equivalent.

**Net effect**: adding `"dtype": "fp8"` to `nemotron_llm`'s
`kv_cache_config={...}` would not error — it would be silently accepted
and then silently discarded, producing an engine that is byte-for-byte
identical to today's, still full-precision KV cache. This would be worse
than doing nothing: it looks like a shipped optimization and isn't one.

**What actually controls FP8 KV cache for the classic backend**:
`QuantConfig.kv_cache_quant_algo` (`tensorrt_llm/models/modeling_utils.py`),
which is populated from the checkpoint's own quantization metadata
(`hf_quant_config.json`'s `kv_cache_quant_algo` field) at model-config
load time — i.e. it's baked into the checkpoint/engine at build time, not
a runtime toggle. This project's checkpoint has:

```json
"kv_cache_quant_algo": null
```

confirming the task's background note: this checkpoint was exported
weights-only (`nvidia-modelopt` with `--kv_cache_qformat none`, per
`reports/session-report.md`'s own quantization notes), with no KV-cache
calibration performed. Getting real FP8 KV cache here means re-running
`nvidia-modelopt` quantization with `--kv_cache_qformat fp8` and a
calibration pass — the same class of effort (512-sample calibration from a
gated dataset, multi-hour run) as the original NVFP4 weight quantization
documented in `reports/session-report.md`, not a follow-up config change.

## Why this isn't worth doing right now

Putting 1 and 2 together: even in the world where FP8 KV cache were a
free, zero-effort flag flip, the expected latency win is a fraction of a
percent, because weight-streaming (not KV-cache reads) dominates this
model's per-token memory traffic by roughly 200x at this pipeline's real
context lengths. It is, in addition, not a flag flip at all for this
serving path — it requires a full recalibrated `nvidia-modelopt` export.
Spending a multi-hour re-quantization pass to chase a sub-1%
theoretical-best latency improvement is out of proportion to the likely
gain, and the capacity headroom that would be the other plausible benefit
isn't currently needed.

## What would actually help, if this needs to be faster

Not implemented here (out of scope for this investigation, and each needs
its own measurement pass), but worth recording since the evidence above
points at them directly:

- **The pipeline is already getting the standard mitigation for
  memory-bound decode "for free": batching.** The concurrency=1..8 numbers
  above show latency is nearly flat up to `max_batch_size=8` — that's
  continuous/in-flight batching amortizing the same weight read across
  multiple concurrent sequences, which is the actual lever that addresses
  memory-bandwidth-bound decode (unlike KV cache precision, which doesn't
  touch the dominant weight-streaming term at all). If more throughput is
  needed under real concurrent voice-pipeline load, raising
  `max_batch_size` (and `MAX_CONCURRENT_STREAMS`/`MAX_ADMITTED` to match)
  — VRAM permitting — extends the "nearly free" concurrency zone further,
  rather than touching KV cache precision. This needs its own before/after
  measurement and VRAM headroom check before being proposed as a change.
- Any further latency reduction on the *single-stream* path fundamentally
  needs fewer weight bytes moved per token (more aggressive weight
  quantization than NVFP4, which is already the aggressive end of what
  `nvidia-modelopt` supports without larger accuracy loss) or fewer
  layers/parameters (a smaller checkpoint) — not KV cache precision.

## Files touched by this investigation

None in `triton_model_repo/` — this was a read-only investigation, no
reload performed, no code changed. This document is the only addition.
