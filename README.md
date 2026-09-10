# Speech Cascade Inference

A voice pipeline (ASR -> LLM -> TTS) served entirely through NVIDIA Triton
Inference Server, running bare-metal on this instance (no Docker — this
container can't run Docker-in-Docker). Matches the v1 architecture diagram:
Triton BLS python-backend models wrapping each stage, all GPU-accelerated.

GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120, 16GB VRAM), driver 595.84
(CUDA 13.2 max). CUDA 12.8 and 13.2 toolkits are both installed system-wide.

## Layout

```
speech-cascade-inference/            Deployment directory -- NOT this git repo, gitignored, lives
                                      only on the instance (see deploy/REBUILD.md "Layout recap")
  models/                          Downloaded/built model weights + compiled engines
    Qwen3-8B-NVFP4/                     Pre-quantized checkpoint pulled directly from HF
                                         (raoashish10/Qwen3-8B-NVFP4) -- not built by a script in
                                         this repo; see deploy/REBUILD.md 4b
    whisper-base-trtllm/                Compiled TensorRT-LLM encoder+decoder engines + assets
                                         (mel filterbank, tokenizer vocab) -- see deploy/REBUILD.md 4c
  triton_model_repo/                 Live copy of this repo's own triton_model_repo/ (below)
  triton_server/
    extracted/tritonserver/            Redistributable no-Docker Triton server build
    compat_libs/                       Locally-extracted libssl.so.1.1 (not in Ubuntu 24.04)
  scripts/pipeline_output.wav        chatterbox_tts's reference voice clip (voice-cloning conditioning input)
```

This repo (`speech-cascade`, what you're reading now) holds everything
git-tracked: each model's `model.py`/`config.pbtxt`, `scripts/`, `tests/`,
`deploy/`. The deployment directory above is separate, gitignored (see
`.gitignore`), and holds the large binary artifacts (weights, compiled
engines, the extracted Triton binary) that don't belong in git — see
`deploy/REBUILD.md` for exactly how the two relate and how to rebuild the
second from the first on a fresh instance.

`chatterbox_tts`'s own weights aren't under `models/` either -- `ChatterboxTurboTTS.from_pretrained()`
pulls them from the HF cache (`HF_HOME`) the first time it's run, same as any
other `from_pretrained()`-based HF model, rather than a locally-checked-in
`.nemo`/ONNX-style checkpoint.

`/opt/supervisor-scripts/speech-cascade-triton.sh` and
`/etc/supervisor/conf.d/speech-cascade-triton.conf` run the server as a
managed service (outside this folder, per this instance's convention for
supervisor-managed apps).

## The four Triton models

| Model | Backend | What it wraps | Input -> Output |
|---|---|---|---|
| `qwen_llm` | python | Qwen3-8B-NVFP4, loaded via TensorRT-LLM's `LLM` API (classic backend, JIT graph build at load time — not an AOT-compiled `.engine`). Decoupled/streaming (`generate_async`, one per request on a bounded thread pool), with an admission gate (`admission.py`) that rejects fast once too many requests are in flight/queued instead of queueing unboundedly | `PROMPT` (string) -> `GENERATED_TEXT` (string), streamed |
| `whisper_asr` | python | `WhisperTRTLLM` (vendored TensorRT-LLM Whisper runtime under `triton_model_repo/whisper_asr/1/trtllm_whisper/`, compiled encoder+decoder engines — see `deploy/REBUILD.md` 4c) | `AUDIO_SAMPLES` (float32[]) + optional `SAMPLE_RATE` (int32) -> `TRANSCRIPT` (string) |
| `chatterbox_tts` | python | ResembleAI's Chatterbox-Turbo (`ChatterboxTurboTTS`), run as a subprocess in its own isolated venv (`/venv/chatterbox`). T3's autoregressive decode is batched across concurrent requests and served via vLLM by default (`CHATTERBOX_BACKEND=vllm`; `pytorch` kept as an instant rollback); S3Gen's flow-matching vocoder still runs per-item in plain PyTorch, unbatched — see `deploy/PROFILING.md` and the `config.pbtxt` comments for why | `TEXT` (string) + optional `VOICE` (string, currently ignored — single fixed reference voice) -> `AUDIO_SAMPLES` (float32[]) + `SAMPLE_RATE` (int32) |
| `voice_pipeline` | python | Calls the three above via Triton's in-process BLS API (`pb_utils.InferenceRequest`), turning any downstream admission rejection into a clean per-request error instead of a crash | `AUDIO_SAMPLES` + optional `SAMPLE_RATE`/`VOICE` -> `TRANSCRIPT`, `GENERATED_TEXT`, `AUDIO_SAMPLES`, `SAMPLE_RATE` |

The first three use Triton's **python backend** as a thin wrapper around a
Python-level runtime (TensorRT-LLM's `LLM` API, vendored TensorRT-LLM Whisper
runtime, a subprocess running ResembleAI's `ChatterboxTurboTTS`) rather than
Triton's native `onnxruntime`/`tensorrt` backends directly.
`voice_pipeline` is pure orchestration — no model weights of its own, no GPU
instance needed — chaining the other three into one audio-in/audio-out
request/response, matching what the diagram's arrows actually show. That's
what "Triton BLS" means: business-logic-scripting models, either wrapping a
runtime or orchestrating other models, as opposed to a raw compiled-graph
backend loading a model file with no custom code.

One non-obvious wrinkle when writing `voice_pipeline`: the models it calls
all declare `max_batch_size > 0` (an implicit leading batch dimension), but
`voice_pipeline` itself is unbatched (`max_batch_size: 0`). Every tensor
forwarded to a callee needs an explicit batch dim of 1 added before the
call (`arr.reshape(1, *arr.shape)`) — Triton doesn't do this automatically
for BLS-constructed requests. The asymmetric part: tensors coming *back*
from a BLS call do **not** carry that batch dimension — they're exactly
what the callee's own `execute()` constructed per-request, before any
wire-level batch aggregation. Getting this backwards produces confusing
failures ("batch size does not match other inputs" if you forget to add it
on the way in, silently wrong data — e.g. a whole audio array collapsed to
its first sample — if you strip a batch dim on the way out that was never
there).

## Process architecture (what's actually running)

`ps aux` / `nvidia-smi` show more processes than "one per model" — worth
knowing which is which before reading GPU memory numbers. A representative
snapshot with all four models loaded, one instance each:

| Process | Typical VRAM | What it is |
|---|---|---|
| `./bin/tritonserver` | ~290MiB | The main Triton server. Holds its own core CUDA memory pools (`pinned_memory_manager` / `cuda_memory_manager`, sized by `--pinned-memory-pool-byte-size` etc. at startup) — no model weights. |
| `triton_python_backend_stub` (one per model *instance*, not per model) | varies — see below | The actual Python process running a given model's `model.py`. `instance_group.count: N` means N of these per model, each a fully separate OS process (own Python interpreter, own GIL). |

Where the stub's own VRAM number lands depends on the model:

- **`chatterbox_tts`**: unlike every other model in this table, the
  `triton_python_backend_stub` process is **not** where the model actually
  runs. `chatterbox_tts/1/model.py` spawns a persistent subprocess running
  `/venv/chatterbox/bin/python3` (a separate, isolated venv — Chatterbox's
  torch/torchaudio pins conflict with this venv's TensorRT-LLM stack) and
  talks to it over a pipe; the stub process itself stays lightweight
  (Triton/Python overhead only), and the real GPU memory shows up against
  that separate `python3` process in `nvidia-smi`, not against
  `triton_python_backend_stub`. Whether vLLM (the default T3 backend as of
  the CUDA-graph/batching work — see the model table above) spawns any
  further subprocess of its own beyond this one hasn't been documented
  here yet.
- **`whisper_asr`**: TensorRT-LLM-based (`WhisperTRTLLM`, see the model
  table above) — whether it spawns a separate MPI worker like `qwen_llm`
  below, or stays in-process, hasn't been documented here yet.
- **`qwen_llm`**: architecturally different from the python-backend-stub
  case above. TensorRT-LLM's classic-backend executor spawns a *separate*
  MPI worker subprocess to actually run the model (the `MpiPoolSession`
  mechanism — see the "Environment quirks" section above, fix #1). That
  worker shows up in `ps aux` as a plain `/venv/main/bin/python3` process,
  **not** `triton_python_backend_stub`, and holds essentially all of the
  LLM's real footprint (weights + KV cache pool + framework overhead,
  capped by `kv_cache_config.free_gpu_memory_fraction: 0.2` in
  `qwen_llm/1/model.py`). The `qwen_llm` stub itself only relays requests
  to the MPI worker over shared memory — it doesn't run inference itself.

This split matters for capacity planning: bumping `qwen_llm`'s
`instance_group.count` doesn't add a cheap extra Python object the way it
does for `voice_pipeline` — each additional instance spawns its own MPI
worker, i.e. its own full copy of the engine + KV cache. `qwen_llm` stays
at `count: 1` today (see `config.pbtxt`); an earlier attempt at `count: 2`
on the old Nemotron checkpoint OOM'd under load — see git history for that
record — and hasn't been retried against the current model.

## VRAM budget

No fresh measured VRAM table for the current `qwen_llm` (Qwen3-8B-NVFP4) +
`whisper_asr` + `chatterbox_tts` combination is tracked in this repo — the
per-model levers that actually govern the budget are, though, and live in
git alongside the code that reads them:

- `qwen_llm/1/model.py` caps the LLM's KV cache pool at
  `kv_cache_config.free_gpu_memory_fraction: 0.2` — without it, TensorRT-LLM
  greedily claims most of the free VRAM for KV cache by default, which isn't
  appropriate on a GPU shared with three other models.
- `chatterbox_tts/config.pbtxt`'s `vllm_gpu_mem_util` parameter (currently
  `0.18`) caps vLLM's own KV-cache reservation for T3 — see that file's
  inline comment for why it had to be lowered from vLLM's `0.3` default once
  all four models load together.

Rerun `scripts/measure_vram.py` / `scripts/measure_vram_classic.py` (or
`nvidia-smi` while the service is up) for current numbers rather than
trusting a stale table here.

## Environment quirks fixed to make this run bare-metal

None of this is needed inside NVIDIA's NGC containers, which ship a
consistent, matched toolchain. Building it manually on this base image
required:

1. **OpenMPI wasn't installed.** TensorRT-LLM's executor uses MPI-based
   worker process spawning even for a single GPU. Fixed with
   `apt-get install openmpi-bin libopenmpi-dev`.
2. **The `tensorrt-llm` pip wheel pulls CUDA 13 bindings but not the CUDA 13
   runtime libraries.** Fixed by installing `cuda-libraries-13-2` and
   `cuda-toolkit-13-2` system-wide (both within the driver's CUDA 13.2
   ceiling — never touches the driver itself).
3. **Triton's Python-backend stub resolves the wrong `sys.prefix`.** It was
   picking up system Python's stdlib C-extensions (e.g. `_datetime.so`)
   instead of the venv's matching build, causing a numpy import crash whose
   real cause was buried under a misleading "importing from source
   directory" numpy error. Fixed with `PYTHONHOME=/venv/main` in the
   supervisor script.
4. **TensorRT-LLM's MPI worker-spawn resolves `python3` via `PATH`,** landing
   on system Python instead of the venv's, loading ABI-mismatched compiled
   extensions (`undefined symbol: _PyErr_SetLocaleString` in `_ctypes`).
   Fixed by prepending `/venv/main/bin` to `PATH`.
5. **Ubuntu 24.04 doesn't ship `libssl.so.1.1`,** which the Triton server
   binary needs. Extracted just the two `.so` files from the Ubuntu 20.04
   `libssl1.1` `.deb` into `triton_server/compat_libs/` rather than
   installing system-wide (avoids touching the system's OpenSSL 3).
6. **NVIDIA DCGM wasn't installed** (Triton links against `libdcgm.so.4` for
   GPU metrics). Fixed with `apt-get install datacenter-gpu-manager-4-cuda13`.

All of these are baked into `/opt/supervisor-scripts/speech-cascade-triton.sh`
(the `PATH`/`LD_LIBRARY_PATH`/`PYTHONHOME`/`PYTHONPATH` exports at the top).
`onnxruntime-gpu` and the numpy-pin fixes that earlier versions of this list
carried for it are gone — that package (and `kokoro-onnx`, which needed it)
is no longer part of this pipeline; see `deploy/requirements-main.txt`.

## Managing the service

```bash
supervisorctl status speech-cascade-triton
supervisorctl restart speech-cascade-triton   # full restart, ~5-6 min (LLM import + engine load)
tail -f /var/log/portal/speech-cascade-triton.log
```

The server runs in **explicit model control mode** (`--load-model=*` loads
all four at startup, but individual models can be reloaded without
restarting the others or paying the LLM's cold-start cost again). The load
API needs a JSON body — an empty POST returns "Method Not Allowed" — and
the repository index endpoint is POST, not GET:

```bash
# after editing whisper_asr/1/model.py, chatterbox_tts/1/model.py, or voice_pipeline/1/model.py:
curl -X POST http://localhost:18000/v2/repository/models/whisper_asr/load -d '{}'
curl -X POST http://localhost:18000/v2/repository/models/chatterbox_tts/load -d '{}'
curl -X POST http://localhost:18000/v2/repository/models/voice_pipeline/load -d '{}'
# qwen_llm reload still pays the ~5 min tensorrt_llm import cost, same as a full restart

curl -X POST http://localhost:18000/v2/repository/index   # list loaded models + state
```

**Ports:** Triton binds to `127.0.0.1` only (18000 HTTP / 18001 GRPC / 18002
metrics) and stays that way — it is **not** exposed externally, deliberately.
The streaming gateway (`streaming_gateway/`, see below) is the sanctioned
external surface: it's what a real client actually talks to, and it already
gives an external caller everything Triton would (transcript, LLM text, TTS
audio) without also handing them raw access to run/reload arbitrary models,
which is a materially bigger blast radius than a single voice endpoint.
Exposing Triton's ports too would widen the attack surface for no added
capability, so they stay internal-only, reachable only via `curl
localhost:1800{0,1,2}` on the box itself or over an SSH tunnel (top-level
agent guide, §7). See `streaming_gateway/README.md` for how the gateway
itself is exposed, on external port `10100`.

## External access

The streaming gateway (`streaming_gateway/`) is reachable from outside the
GPU box. It sits behind the instance's Caddy auth edge rather than on a bare
open port — anyone with the URL but not the token gets rejected before the
WebSocket upgrade even completes, whereas an unauthenticated open port would
be reachable by literally anyone.

**System-level wiring (not tracked in this repo, lives on the instance):**

```yaml
# /etc/portal.yaml — added under `applications:`
Streaming Gateway:
  hostname: localhost
  external_port: 10100
  internal_port: 18010
  open_path: /ws/stream
  name: Streaming Gateway
```
```bash
supervisorctl restart caddy   # picks up the new portal.yaml entry
```
This is the only system-level change; the gateway process itself is
unchanged (still `uvicorn ... --host 127.0.0.1 --port 18010`, run by the
existing `speech-cascade-gateway` supervisor service). To reproduce on a
fresh instance: add that YAML block to `/etc/portal.yaml` and restart caddy
— see `streaming_gateway/README.md` for the exact `python3 -c` one-liner
used to do this safely (load-modify-dump, so other portal.yaml entries are
preserved).

**Connecting from outside the box:**

```bash
python3 scripts/test_streaming_client.py --wav your_clip.wav \
  --gateway-url ws://<PUBLIC_IPADDR>:<VAST_TCP_PORT_10100>/ws/stream \
  --token "$OPEN_BUTTON_TOKEN"
```

Full details — auth methods, why Triton itself stays internal-only, the
concurrent-session cap, and what deliberately wasn't added — are in
`streaming_gateway/README.md`.

## Testing each stage

```bash
# LLM
curl -s -X POST http://localhost:18000/v2/models/qwen_llm/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"PROMPT","shape":[1,1],"datatype":"BYTES","data":["Hello, my name is"]}]}'

# ASR — needs a float32 audio array; see scripts/test_asr.py for the standalone version
# TTS
curl -s -X POST http://localhost:18000/v2/models/chatterbox_tts/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"TEXT","shape":[1,1],"datatype":"BYTES","data":["Hello there."]}]}'

# Full pipeline (audio in -> audio out, one call) — needs a float32 audio
# array same as ASR above, but unbatched (no leading [1, ...] dim, since
# voice_pipeline itself has max_batch_size: 0):
# {"inputs":[
#   {"name":"AUDIO_SAMPLES","shape":[N],"datatype":"FP32","data":[...]},
#   {"name":"SAMPLE_RATE","shape":[1],"datatype":"INT32","data":[24000]}
# ]}
curl -s -X POST http://localhost:18000/v2/models/voice_pipeline/infer \
  -H "Content-Type: application/json" --data @request.json
```

`scripts/` also has the standalone Python scripts used to validate each
stage before wiring it into Triton (`test_asr.py`, `test_tts.py`,
`quantize_fp8.py`, `measure_vram.py`, `measure_vram_classic.py`) — useful for
isolating a problem outside Triton's stub-process environment if something
breaks again.

`scripts/build_engine.sh` and `scripts/quantize_fp8.py`/`scripts/quantize_nvfp4.py`
are **not** part of how the currently-served `qwen_llm` checkpoint was
produced — `qwen_llm` loads `Qwen3-8B-NVFP4`, pulled pre-quantized directly
from Hugging Face (`raoashish10/Qwen3-8B-NVFP4`), not built by any script in
this repo. These scripts are kept from an earlier FP8/NVFP4-on-Nemotron
quantization workflow (AOT `trtllm-build` engine, in the older classic
pipeline) as reference for any future from-scratch quantization; see
`deploy/REBUILD.md` sections 4a/4b for exactly what they do and don't cover
today.

## Metrics and load testing

Triton exposes rich per-model Prometheus metrics at `/metrics` on port 18002
(localhost only) — request success/failure counts, cumulative queue and
compute durations per model, GPU utilization/memory/power. No built-in
percentile histograms though, just sums+counts, so latency percentiles need
client-side timing.

A standalone Prometheus server (`apt install prometheus`, not present on the
base image) scrapes that endpoint every 2s, configured via
`prometheus.yml`, running as another supervisor service
(`speech-cascade-prometheus`). It stays `127.0.0.1`-only, like Triton itself
— Grafana (below) is the externally-reachable surface for looking at this
data, not Prometheus directly. Query it locally:

```bash
curl -s http://localhost:9090/api/v1/query --data-urlencode 'query=nv_gpu_utilization'
```

A Grafana dashboard (`monitoring/grafana/`), Prometheus alerting rules
(`monitoring/alert_rules.yml`), and a custom exporter for Triton's model
READY/UNAVAILABLE state (not natively a Prometheus metric,
`scripts/triton_state_exporter.py`) sit on top of this. Grafana is exposed
externally on port `10200` through the Caddy-authed edge (same pattern as
the Streaming Gateway on `10100` — see
[`streaming_gateway/README.md`](streaming_gateway/README.md)); Prometheus
and the exporter stay internal-only. `Enable Triton's summary_latencies`
(`deploy/supervisor/speech-cascade-triton.sh`) turns the cumulative
queue/compute counters above into real Summary metrics with quantile
labels, so Grafana can show true p50/p90/p99, not just averages.

`scripts/load_test.py` fires concurrent requests via `tritonclient`'s native
gRPC binary protocol — hand-rolled JSON-over-HTTP was measurably more
expensive for both float-array (ASR/TTS) and even plain-text (LLM) payloads
in earlier testing on this pipeline — records client-side latency
percentiles, and diffs Triton's own metrics before/after to report
per-model exec counts and average compute/queue time. With no `--model`, it
runs `whisper_asr`, `qwen_llm`, `chatterbox_tts`, and `voice_pipeline` in
turn (each isolated, same concurrency) for a clean per-stage p50 breakdown:

```bash
python3 scripts/load_test.py --concurrency 4 --total-requests 20
python3 scripts/load_test.py --concurrency 4 --total-requests 20 --model chatterbox_tts  # just one
```

Prometheus/`load_test.py` answer *how much* time a stage takes; they don't
show *why* — for that, this repo also wires up `torch.profiler` (op-level,
`chatterbox_tts` only — the one plain-PyTorch stage) and Nsight Systems /
`nsys` (kernel-level, all four models, including inside TensorRT-LLM's
compiled engines for `qwen_llm`/`whisper_asr`, which `torch.profiler` can't
see into). See [`deploy/PROFILING.md`](deploy/PROFILING.md) and
`scripts/profile_nsys.sh`.

Two structural batching findings that shaped the current config, still
reflected in `config.pbtxt` today:

- **`voice_pipeline` needs `instance_group.count > 1`** — its `execute()`
  blocks synchronously through the whole ASR→LLM→TTS chain, so with only
  one instance, concurrent requests can't overlap at all; it holds no model
  weights, so extra instances are cheap (currently `count: 4`, see
  `triton_model_repo/voice_pipeline/config.pbtxt`).
- **Whichever stage doesn't natively batch multiple requests per
  `execute()` call needs either `dynamic_batching` plus code that actually
  passes the whole request list through in one call, or (for `qwen_llm`,
  streaming) relies on the underlying runtime's own concurrent-request
  handling instead** — `whisper_asr` batches its `processor()`/`generate()`
  call across requests (`max_batch_size: 8` + `dynamic_batching`);
  `chatterbox_tts` batches T3's decode the same way (`max_batch_size: 4` +
  `dynamic_batching`, see the model table above); `qwen_llm` dropped
  `dynamic_batching` entirely once it moved to per-request streaming
  (`generate_async`) — see that model's `config.pbtxt` comment for why a
  batching *window* is the wrong tradeoff for a streaming, real-time-voice
  workload.

Full historical numbers from the load-testing sessions that found these
(measured against the now-retired `nemotron_llm`/`kokoro_tts` stack) live in
git history, not here — they don't describe the current models.

## Tests and infrastructure as code

`tests/` — automated pytest suite, two tiers:

- `tests/unit/` — pure-logic tests for the streaming gateway's
  `UtteranceVAD` and `SentenceAccumulator`, no live server or GPU needed.
  Runs in GitHub Actions CI on every push/PR (`.github/workflows/tests.yml`).
- `tests/integration/` — real gRPC calls against the live Triton server
  (each of the 4 models individually, `voice_pipeline` end-to-end, and a
  small fixed regression set). Needs a live GPU server, so it does **not**
  run in CI — run it by hand on the instance:
  `/venv/main/bin/python -m pytest tests/integration -v`.

See `tests/README.md` for exactly which venv each tier needs and how to
select by marker instead of directory.

`deploy/` — infrastructure as code, so this deployment can be rebuilt from
nothing instead of only existing as a terminal history on one GPU box:

- `deploy/requirements-main.txt` / `deploy/requirements-gateway.txt` —
  exact `pip freeze` of both venvs.
- `deploy/supervisor/` — the supervisor wrapper scripts + conf.d files
  actually installed on the instance (Triton server, streaming gateway,
  Prometheus, Grafana, the Triton-state exporter, the alert notifier).
- `deploy/REBUILD.md` — the executable runbook: fresh instance -> working
  deployment, including where the source weights come from, how each
  TensorRT engine was built, and which steps are fully scripted vs.
  documented-but-manual.

See `deploy/README.md` for the full breakdown.

## Not yet done

- **S3Gen (the flow-matching vocoder half of `chatterbox_tts`) has no
  TensorRT/ONNX/compiled acceleration path yet.** T3 (the autoregressive
  half) got a real win by moving to vLLM (see the model table above and
  `deploy/PROFILING.md`); S3Gen still runs per-item in plain PyTorch,
  unbatched, and profiling to find out whether it's now the dominant cost
  hasn't happened.
- **`chatterbox_tts.instance_group.count` stays at 1.** Replicating it isn't
  free the way `voice_pipeline`'s replication is — see the reasoning in
  `chatterbox_tts/config.pbtxt`'s own comment (VRAM headroom on this
  16GB card, shared with the other three models).
- **`qwen_llm.instance_group.count` stays at 1** — an earlier `count: 2`
  attempt against the prior (Nemotron) checkpoint OOM'd under load; not
  retried against the current Qwen3-8B-NVFP4 checkpoint.
- **TensorRT-LLM engine build flags at near-defaults for `whisper_asr`'s
  compiled engines** (`deploy/REBUILD.md` 4c) — `qwen_llm` no longer builds
  an AOT engine at all (JIT via the `LLM` API, see the model table above),
  so flags like `reduce_fusion`/`multiple_profiles` only apply to Whisper's
  engines now, and haven't been revisited there.
