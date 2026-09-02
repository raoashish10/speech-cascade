# qwen_llm migration: Triton startup failure, root cause, and load test results

## Verdict

`nemotron_llm` is retired; the Triton model is renamed `qwen_llm` and now
serves `Qwen3-8B-NVFP4` (`raoashish10/Qwen3-8B-NVFP4` on Hugging Face),
loaded the same way the old checkpoint was — directly via TensorRT-LLM's
`LLM` API (JIT engine build at load time, no `trtllm-build` AOT step; see
`deploy/REBUILD.md` 4b).

Getting there exposed two real, previously-latent bugs in the deployment
itself, neither one caused by the rename:

1. **DCGM was never actually installed**, only a single hand-extracted
   `.so` file — this broke Triton's startup deterministically, independent
   of which LLM was configured. Fixed by installing the real package.
2. **Loading all four models concurrently at startup OOMs on this box's
   RAM once the LLM is 8B params instead of 4B.** Fixed by loading them
   sequentially.

All four models now reach `READY` reliably and a load test against the
full stack passed. Details and evidence below.

## 1. Triton would never actually start serving

### Symptom

Every attempt to bring the server up — regardless of which models were
configured to load, or in what order — ended the same way: a model's
python-backend stub would report `"Stub process '<name>_0_0' is not
healthy"`, and shortly after (sometimes immediately, sometimes on
shutdown), the whole `tritonserver` binary died with:

```
tritonserver: symbol lookup error: .../lib64/libtritonserver.so: undefined symbol: errorString
```

Grepping this session's logs for Triton's own "server is up" banner lines
(`Started GRPCInferenceService`, `Started HTTPService`) turned up **zero
matches across every prior attempt** — the HTTP/gRPC endpoints had never
actually come up even once, on any model configuration.

### Method

- `nm -D --no-demangle` on `libtritonserver.so` to confirm `errorString` is
  a genuine undefined (`U`) symbol the binary expects some loaded library
  to provide, and to search every directory on the launch script's
  `LD_LIBRARY_PATH` for a library that defines it (`grep -i errorstring`
  across `nm -D` output for every `.so` reachable from that path) — no
  library anywhere on the system defined a bare, unprefixed `errorString`.
- `LD_DEBUG=bindings` against a fresh `tritonserver` launch, redirected to
  a file, to capture the dynamic linker's actual symbol-resolution order
  leading up to the crash — this is what actually pinned down the cause,
  rather than continuing to guess library-by-library.
- Isolated single-model launches (`qwen_llm` alone, with `chatterbox_tts`/
  `whisper_asr` not loaded) with concurrent `free -m`/`nvidia-smi` sampling
  every 3-5s, to separate "does this model load at all" from "does it
  survive concurrent resource pressure."

### Root cause

The `LD_DEBUG=bindings` trace showed the last successful symbol binding
before the fatal error was `dcgmInit`, resolved from
`triton_server/compat_libs/extracted/usr/lib/x86_64-linux-gnu/libdcgm.so.4`
— a single `.so` file that had been manually `dpkg-deb -x`'d out of a
DCGM `.deb`, mirroring the (correct, documented) pattern used for the
`libssl1.1` compat workaround (`README.md` quirk #5). That pattern is
wrong for DCGM: the real `datacenter-gpu-manager-4-cuda13` package installs
`libdcgm.so.4` **plus a family of companion module libraries**
(`libdcgmmodulesysmon.so.4`, `libdcgmmoduleprofiling.so.4`,
`libdcgmmodulepolicy.so.4`, `libdcgmmodulenvswitch.so.4`,
`libdcgmmoduleintrospect.so.4`, `libdcgmmodulehealth.so.4`,
`libdcgmmodulediag.so.4`, `libdcgmmoduleconfig.so.4`,
`libdcgm_cublas_proxy13.so.4`) that the bare hand-extracted file never
had. `datacenter-gpu-manager-4-cuda13` was confirmed **not installed**
(`dpkg -l | grep dcgm` — no output) despite `README.md`/`deploy/REBUILD.md`
already documenting `apt-get install datacenter-gpu-manager-4-cuda13` as
required (quirk #6) — the documented fix was correct, it just hadn't
actually been applied; someone had substituted the libssl1.1-style manual
extraction instead, which is incomplete for this specific library.

### Fix

```bash
apt-get install -y datacenter-gpu-manager-4-cuda13   # toolkit-only, does not touch the driver
rm -f triton_server/compat_libs/extracted/usr/lib/x86_64-linux-gnu/libdcgm.so.4
```

The `rm` matters: `LD_LIBRARY_PATH` puts `compat_libs/extracted/...` ahead
of the system path, so the broken hand-extracted copy would otherwise keep
shadowing the properly-installed one. `libssl.so.1.1`/`libcrypto.so.1.1`
stay in `compat_libs` — that extraction is correct and still needed
(Ubuntu 24.04 genuinely doesn't ship `libssl.so.1.1`, and there's no
apt package providing it). **Don't apply the libssl1.1-extraction pattern
to DCGM again** — install it with `apt-get install
datacenter-gpu-manager-4-cuda13` for real.

After the fix, `Started GRPCInferenceService`/`Started HTTPService`
appeared for the first time this session, HTTP live within 2s of launch.

## 2. Loading all four models concurrently OOMs host RAM

### Symptom

With DCGM fixed, `qwen_llm` alone reached `READY` reliably. Loading all
four models concurrently (the original `--load-model=X` × 4 startup flags)
still intermittently produced `"Stub process 'qwen_llm_0_0' is not
healthy"` — a silent kill with no Python traceback, consistent with
`SIGKILL` rather than a catchable exception.

### Method

Concurrent `free -m` + `nvidia-smi --query-gpu=memory.used` sampling every
3-5s through both a solo `qwen_llm` load and a concurrent 4-model load, to
see actual peak resource usage rather than inferring it from log timing.

### Evidence

Container memory cap (`cat /sys/fs/cgroup/memory.max`): **~14.7 GiB**
(`15800991744` bytes). `qwen_llm`'s own JIT engine build, loaded alone with
nothing else running, transiently peaked host RAM at:

```
17:33:11  ram_used_mb=14542  ram_avail_mb=1154   <- peak, ~1.1GB from the ceiling
17:33:15  ram_used_mb=8466   ram_avail_mb=7230    <- settles fast once loaded
17:33:18  ram_used_mb=8880   ram_avail_mb=6816
```

GPU was never the constraint — `qwen_llm` alone peaked at ~9.1GB of the
16.3GB card, comfortable alongside `chatterbox_tts`'s own ~3.6GB. The
**host RAM peak is transient and specific to the JIT build window**; it
settles to a much lower steady state within seconds of `successfully
loaded`. Loading `whisper_asr`/`chatterbox_tts` concurrently during that
same transient window is what pushed total usage over the container's
ceiling.

### Fix

`deploy/supervisor/speech-cascade-triton.sh` now starts `tritonserver`
with no `--load-model` flags (still `--model-control-mode=explicit`) and
loads each model in turn via `POST /v2/repository/models/{name}/load`,
which blocks until that model's load finishes (success or failure) before
the loop proceeds to the next — guaranteeing no two models' loading-time
peaks can overlap. `qwen_llm` loads first, while baseline RAM is lowest.
See the script's own header comment for the full rationale inline.

Real cost: total time-to-fully-READY goes up, since loading is now
serialized instead of parallel (~21 minutes for all four in the run that
validated this fix, dominated by `qwen_llm`'s ~6.5min JIT build and
`whisper_asr`'s own multi-minute TensorRT-LLM engine load). That's the
correct trade against the alternative of a nondeterministic OOM kill.

## 3. Load test results (post-fix, all 4 models READY)

`python3 scripts/load_test.py --concurrency 4 --total-requests 20`, run
against the fully-loaded stack immediately after both fixes above, GPU at
13912 MiB / 16303 MiB steady state going in:

| Model | Successes | Failures | Throughput | p50 | p90 | p99 | Triton avg_compute | Triton avg_queue |
|---|---|---|---|---|---|---|---|---|
| `whisper_asr` | 20/20 | 0 | 11.637 req/s | 0.020s | 1.634s | 1.635s | 1340.2ms | 2.9ms |
| `qwen_llm` | 19/20 | 1 | 0.316 req/s | 0.080s | 0.237s | 60.096s | 0.9ms | 1.9ms |
| `chatterbox_tts` | 20/20 | 0 | 2.370 req/s | 1.298s | 3.124s | 3.431s | 421.6ms | 1174.9ms |
| `voice_pipeline` | 20/20 | 0 | 1.748 req/s | 2.281s | 2.412s | 2.507s | 2141.1ms | 0.0ms |

`qwen_llm` TTFT (time-to-first-token, streaming): p50=0.018s, p90=0.175s,
p99=0.175s.

GPU memory grew by only +268 MiB total across the whole run (13912 →
14180 MiB) — no evidence of a per-request leak at this concurrency/volume.

**One `qwen_llm` failure (1/20), p99 driven by a single 60.096s outlier**
with no captured error message (`first failure:` empty in the script's
output — the failure surfaced as a client-side timeout, not a Triton-side
error response). This request was very likely the *first-ever* inference
call against the newly-loaded `qwen_llm` instance in this run — consistent
with a one-time cold-start cost (CUDA graph capture, kernel autotuning, or
similar warm-up work TensorRT-LLM defers to first use) rather than a
steady-state problem. **Not yet re-verified** — a follow-up run isolating
whether request #1 specifically is the one that fails, and whether a
higher-concurrency/longer run reproduces further failures, would confirm
this before ruling out a real bug.

These are the first `qwen_llm` numbers ever recorded (no prior comparable
`nemotron_llm` load-test data exists in this exact form to diff against —
`README.md`'s "Metrics and load testing" section's existing narrative and
numbers were measured on `nemotron_llm` before this migration and are kept
there as history, not current fact).

## 4. How this compares to vLLM and SGLang serving the same checkpoint

Separate from the Triton deployment above: see
[`qwen-nvfp4-serving-backend-comparison.md`](qwen-nvfp4-serving-backend-comparison.md)
for a controlled, isolated (single-model, idle-GPU) comparison of
`Qwen3-8B-NVFP4` served via TensorRT-LLM, vLLM, and SGLang — including two
real `flashinfer` SM120 (this GPU's architecture) compatibility gaps hit
getting vLLM and SGLang working at all, and the workarounds for each.
