# Speech Cascade Inference

A voice pipeline (ASR -> LLM -> TTS) served entirely through NVIDIA Triton
Inference Server, running bare-metal on this instance (no Docker — this
container can't run Docker-in-Docker). Matches the v1 architecture diagram:
Triton BLS python-backend models wrapping each stage, all GPU-accelerated.

A containerized alternative to this bare-metal deployment (portable to any
GPU host with Docker + `nvidia-container-toolkit`, not tied to Vast.ai) is
in [`docker/`](docker/README.md). It has since been **verified end-to-end on
a real GPU** (RTX PRO 4500 Blackwell, sm_120): all four models reach Triton
state `READY` and the full `voice_pipeline` ensemble round-trips speech ->
ASR -> LLM -> TTS -> speech, with weights and engines built from scratch (no
S3) per `scripts/build_models_from_scratch.sh`. Engines are deliberately not
baked into the image — they're GPU-architecture-specific. What is still
open there is the CI image build itself, plus the `vllm` chatterbox backend
and the gateway container; see that directory's README, which tracks each.

GPU: NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120, 16GB VRAM), driver 595.84
(CUDA 13.2 max). CUDA 12.8 and 13.2 toolkits are both installed system-wide.

## Layout

```
speech-cascade-inference/
  models/                          Downloaded/built model weights
    Qwen3-8B-NVFP4/                    qwen_llm's engine_dir AND tokenizer_dir (pre-quantized, pulled from the Hub)
    Qwen3-8B-hf/                       Unquantized Qwen3-8B, only read when QWEN_BACKEND=pytorch (the no-TensorRT-LLM comparison path)
    whisper-base-trtllm/               whisper_asr's engine_dir: TensorRT-LLM encoder/ + decoder/ engines
      assets/                            multilingual.tiktoken + mel_filters.npz (whisper_asr's assets_dir)
    whisper-base-hf/                   Plain HF whisper-base, only read when WHISPER_BACKEND=pytorch
  triton_model_repo/                 Triton model repository (4 models, see below)
  triton_server/
    extracted/tritonserver/            Redistributable no-Docker Triton server build
    compat_libs/                       Locally-extracted libssl.so.1.1 (not in Ubuntu 24.04)
  scripts/                            Standalone validation scripts (used during setup)
    pipeline_output.wav                 chatterbox_tts's reference voice clip (voice-cloning conditioning input)
  deprecated/
    Nemotron-3-Nano-4B-FP8/            Abandoned model, see "Why not Nemotron-3-Nano" below
```

This listing used to show `Llama-3.1-Nemotron-Nano-4B-v1.1/`,
`llama_nemotron_fp8_ckpt/`, `llama_nemotron_engine/` and an ONNX
`whisper-base/`. None of those are served any more: the LLM is
`Qwen3-8B-NVFP4` and ASR runs TensorRT-LLM engines, per the model table
below. Those paths still appear in `scripts/quantize_fp8.py`,
`scripts/quantize_nvfp4.py`, `scripts/build_engine.sh`,
`scripts/measure_vram_classic.py` and `deploy/REBUILD.md` as the record of
how the earlier checkpoints were built — they describe history, not the
current deployment.

Every path above is exactly what `triton_model_repo/*/config.pbtxt` points
at, so this is the layout both the bare-metal deployment and
`docker/`'s image/volume reproduce.

`chatterbox_tts`'s own weights aren't under `models/` -- `ChatterboxTurboTTS.from_pretrained()`
pulls them from the HF cache (`HF_HOME`) the first time it's run, same as any
other `from_pretrained()`-based HF model, rather than a locally-checked-in
`.nemo`/ONNX-style checkpoint. Its `ref_audio_path` reference voice clip
(`scripts/pipeline_output.wav`) is a deployment artifact, not something
reproducible from public sources.

`/opt/supervisor-scripts/speech-cascade-triton.sh` and
`/etc/supervisor/conf.d/speech-cascade-triton.conf` run the server as a
managed service (outside this folder, per this instance's convention for
supervisor-managed apps).

## The four Triton models

| Model | Backend | What it wraps | Input -> Output |
|---|---|---|---|
| `qwen_llm` | python | TensorRT-LLM, classic AOT-compiled engine | `PROMPT` (string) -> `GENERATED_TEXT` (string) |
| `whisper_asr` | python | `WhisperTRTLLM` (vendored TensorRT-LLM Whisper runtime, compiled encoder+decoder engines — see `deploy/REBUILD.md` 4c; this table previously said `optimum`/ONNX Runtime, which is stale) | `AUDIO_SAMPLES` (float32[]) + optional `SAMPLE_RATE` (int32) -> `TRANSCRIPT` (string) |
| `chatterbox_tts` | python | ResembleAI's Chatterbox-Turbo reference implementation (`ChatterboxTurboTTS`, plain PyTorch), run as a subprocess in its own isolated venv (`/venv/chatterbox`) rather than in-process — replaces the former `magpie_tts` (rejected candidate, never actually deployed) / `kokoro_tts` (dead: package uninstalled, weights removed) — see "Not yet done" | `TEXT` (string) + optional `VOICE` (string, currently ignored — single fixed reference voice) -> `AUDIO_SAMPLES` (float32[]) + `SAMPLE_RATE` (int32) |
| `voice_pipeline` | python | Calls the three above via Triton's in-process BLS API (`pb_utils.InferenceRequest`) | `AUDIO_SAMPLES` + optional `SAMPLE_RATE`/`VOICE` -> `TRANSCRIPT`, `GENERATED_TEXT`, `AUDIO_SAMPLES`, `SAMPLE_RATE` |

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
  `triton_python_backend_stub`. Measured standalone footprint (outside
  Triton, same reference implementation): ~3.0-3.4GB — see
  `docs/tts-replacement-investigation.md`.
- **`whisper_asr`**: now TensorRT-LLM-based (`WhisperTRTLLM`, see the model
  table above) rather than the ONNX Runtime path this section originally
  described — whether it also spawns a separate MPI worker like
  `qwen_llm` below, or stays in-process, hasn't been re-verified since
  that migration.
- **`qwen_llm`**: architecturally different. TensorRT-LLM's executor
  spawns a *separate* MPI worker subprocess to actually run the engine (the
  `MpiPoolSession` mechanism — see the "Environment quirks" section above,
  fix #1). That worker shows up in `ps aux` as a plain
  `/venv/main/bin/python3` process, **not** `triton_python_backend_stub`,
  and holds essentially all of the LLM's real footprint (KV cache pool +
  weights + framework overhead). The `qwen_llm` stub itself only holds
  ~290MiB — it's just relaying requests to the MPI worker over shared
  memory, not running inference itself. (The ~7.5GB FP8 figure this bullet
  used to quote was measured against the old 4B Nemotron checkpoint on the
  RTX 5070 Ti; the current model is the 8B `Qwen3-8B-NVFP4`, and its
  footprint has not been re-measured — see the VRAM table below.)

This split matters for capacity planning: bumping `qwen_llm`'s
`instance_group.count` doesn't add a cheap extra Python object the way it
does for `voice_pipeline` — each additional instance spawns its own MPI
worker, i.e. its own full ~7.5GB copy of the engine + KV cache. See below
for what happened when this was actually tried.

## VRAM budget (measured)

**Stale — kept as the record, do not plan against it.** These numbers were
measured on the RTX 5070 Ti (16GB) with the old 4B Nemotron FP8 LLM and the
ONNX ASR path. Both have since been replaced (`Qwen3-8B-NVFP4`, TensorRT-LLM
Whisper engines), so the LLM is now twice the parameter count at a different
quantization, and nothing below has been re-measured against it.

| Stage | VRAM |
|---|---|
| Baseline (nothing loaded) | ~0 MiB / 16303 MiB |
| LLM (then `nemotron_llm`) loaded alone | ~8185 MiB |
| All three GPU models loaded together | ~9933 MiB (5910 MiB free) |

What is still current is the shape of the constraint. The LLM's KV cache
pool is capped at `free_gpu_memory_fraction: 0.2` in `qwen_llm/1/model.py`
and whisper_asr uses `kv_cache_free_gpu_memory_fraction=0.05` — without
those caps TensorRT-LLM greedily claims most of the free VRAM for KV cache
by default, which isn't appropriate on a GPU shared with two other models.

## Why the LLM is Llama-3.1-Nemotron-Nano-4B, not Nemotron-3-Nano-4B

**Superseded:** `qwen_llm` now serves `Qwen3-8B-NVFP4`, not a Nemotron
checkpoint at all — see the model table above and `deploy/REBUILD.md`. The
history below (including the VRAM/batching numbers further down that were
measured against the Nemotron checkpoint) is kept as the record of how that
choice was made and later revisited; it does not describe the current LLM.

The original attempt used `nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8` (see
`deprecated/`), a hybrid Mamba2+Transformer architecture. TensorRT-LLM can
only run that family through its newest **AutoDeploy** backend (JIT/eager
`torch.compile`-based, not a compiled engine) — there's no classic
`trtllm-build` path for Mamba SSM ops. AutoDeploy worked, but only in eager
mode (`compile_backend="torch-simple"`) after hitting a real crash in its
default `torch-compile` path (a shape-mismatch bug in the Mamba SSM metadata
kernel), and even then first-token latency was ~119s due to one-time
FlashInfer kernel JIT compilation.

Switched to `nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1` — a plain Llama
architecture, also branded Nemotron, same ~4B size. That goes through the
classic pipeline: HF checkpoint -> `nvidia-modelopt` FP8 calibration (32
short calibration samples, see `scripts/quantize_fp8.py`) ->
`export_tensorrt_llm_checkpoint` -> `trtllm-build` (see `scripts/build_engine.sh`
for the exact flags) -> a real ahead-of-time compiled `.engine`. Generation
latency dropped to under a second, with zero
JIT unpredictability at serve time. This is genuinely the mature, intended
TensorRT-LLM path — AutoDeploy exists specifically for architectures the
classic path can't represent, and this model doesn't need it.

## Environment quirks fixed to make this run bare-metal

None of this is needed inside NVIDIA's NGC containers, which ship a
consistent, matched toolchain — that claim has now been exercised rather
than asserted: `docker/` builds on `nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3`
and needed none of fixes 2-9 below. It is not quite "zero quirks" either;
fix #1 (MPI) reappears inside the container in a different form, as
`OPAL_PREFIX` — see `docker/triton/Dockerfile`.

Fixes **7, 8 and 9 are historical**: they were for the ONNX Runtime ASR
path and the Kokoro TTS, both of which are gone (see the model table
above). They are kept as the record of how that setup was made to work,
not as steps to repeat. Building it manually on this base image required:

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
7. **`onnxruntime-gpu`'s CUDA EP wants cuBLAS 12.x, not 13.x** (cuBLAS 13
   has breaking API changes) — loading the system's cu13 `libcublasLt.so`
   produced `Cannot load symbol cublasLtCreate`. Fixed by pointing
   `LD_LIBRARY_PATH` at the cu12 cuBLAS/cuDNN shipped inside the venv's
   `nvidia-*` pip packages (pulled in by `torch`) instead.
8. **The Kokoro FP16 ONNX export produces NaN output on this GPU/onnxruntime
   combination** (raw audio had the right shape, every sample was NaN).
   Switched to the FP32 `model.onnx` variant — no measurable VRAM pressure
   difference at this model size (~325MB).
9. **`kokoro-onnx` expects a combined `voices-v1.0.bin`** (all voices in one
   file), not the individual per-voice `.bin` files `onnx-community`
   publishes. Downloaded the combined file from the `kokoro-onnx` project's
   own GitHub release instead.
10. **`pip install optimum[onnxruntime-gpu] kokoro-onnx librosa` silently
    upgraded numpy from 1.26.4 to 2.5.2**, which risked breaking
    TensorRT-LLM's compiled bindings (built against numpy 1.x). Pinned back
    to `numpy<2` immediately after — all the new packages still work fine
    with numpy 1.x.

All of these are baked into `/opt/supervisor-scripts/speech-cascade-triton.sh`
(the `PATH`/`LD_LIBRARY_PATH`/`PYTHONHOME`/`PYTHONPATH` exports at the top).

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

`scripts/build_engine.sh` runs the `trtllm-build` step against a checkpoint
produced by `quantize_fp8.py`, with the flags this project's engine was
actually built with (`--max_batch_size`/`--max_seq_len` overridable via env
vars). The output `.engine` is tied to the exact GPU architecture and
TensorRT-LLM version it was built with — copying it to different hardware
isn't reliable, so replicating this repo elsewhere means rerunning this
script there, not copying `models/llama_nemotron_engine/`.

`scripts/quantize_nvfp4.py` reconstructs the NVFP4 quantization step that
actually produced what `qwen_llm` serves today (see `deploy/REBUILD.md`
"4b") — the FP8 path above was superseded, not deleted; it's kept for
reference and any future re-quantization on hardware/versions where NVFP4
isn't the right call. Unlike `quantize_fp8.py`, this wraps NVIDIA's own
`hf_ptq.py` example CLI (cloned from `NVIDIA/TensorRT-Model-Optimizer`)
rather than calling the calibration API directly — see the script's own
docstring for exactly which flags are confirmed against upstream source vs.
reconstructed from `speech-cascade-inference/reports/session-report.md`.
Defaults to a scratch output directory, not the live-served checkpoint path.

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

A Grafana dashboard, Prometheus alerting rules, and a custom exporter for
Triton's model READY/UNAVAILABLE state (not natively a Prometheus metric)
sit on top of this — see [`docs/monitoring.md`](docs/monitoring.md) for
URLs, what each alert means, and what to do when one fires. Grafana is
exposed externally on port `10200` through the Caddy-authed edge (same
pattern as the Streaming Gateway on `10100` — see
[`streaming_gateway/README.md`](streaming_gateway/README.md)); Prometheus
and the exporter stay internal-only.

`scripts/load_test.py` fires concurrent requests via `tritonclient`'s native
gRPC binary protocol (see "Two further optimizations" below for why not
JSON-over-HTTP), records client-side latency percentiles, and diffs Triton's
own metrics before/after to report per-model exec counts and average
compute/queue time. With no `--model`, it runs `whisper_asr`, `qwen_llm`,
`chatterbox_tts`, and `voice_pipeline` in turn (each isolated, same concurrency)
for a clean per-stage p50 breakdown:

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

**The numbers and bugs below are historical, measured on `nemotron_llm`
before the `qwen_llm` migration** — kept as-is since they document real,
still-relevant findings about `voice_pipeline`/`whisper_asr`/batching
behavior that isn't specific to which LLM is loaded. For `qwen_llm`'s own
first post-migration load test results, see
[`docs/qwen-llm-migration.md`](docs/qwen-llm-migration.md) section 3.

**Two real bugs this surfaced**, both now fixed:

1. **`voice_pipeline` defaulted to `instance_group.count: 1`.** Its
   `execute()` blocks synchronously through the entire ASR→LLM→TTS chain, so
   with only one instance, concurrent requests couldn't overlap *at all* —
   they queued almost fully serially (4 concurrent requests: p50 latency
   15.85s, `voice_pipeline`'s own queue time 10.5s, while every model it
   calls showed ~0 queue time, since only one BLS chain was ever running).
   Bumped to `count: 4` — it holds no model weights, so extra instances are
   nearly free. That alone dropped p50 to a still-broken-but-informative
   state and, combined with fix #2 below, got throughput to 0.570 req/s (vs.
   0.242 broken, 0.205 baseline) with zero failures at concurrency 4.
2. **`optimum.onnxruntime.ORTModelForSpeechSeq2Seq`'s IOBinding isn't safe
   for back-to-back reuse.** Once `voice_pipeline` could issue truly
   concurrent requests, Triton correctly started bundling multiple queued
   requests into a single `whisper_asr.execute()` call — and IOBinding's
   pre-allocated output buffers, sized for one `generate()` call's shape,
   don't get reset before the next sequential call in that same batch,
   producing `INVALID_ARGUMENT: ... has shape {1,8,1500,64} but the computed
   output shape ... is {0,8,1,64}` and crashing 17 of 20 requests. Fixed
   with `use_io_binding=False` in `whisper_asr/1/model.py` — costs a small
   memory-copy overhead (`whisper_asr`'s own compute time went from ~25ms to
   ~956ms under load) but that was never the bottleneck, so it's a clean
   trade for correctness.

After both fixes, the metrics tell a coherent story instead of one bottleneck
hiding everything else: `whisper_asr` and `kokoro_tts` show real queue time
under concurrency (contention on their single GPU instance each), while
`nemotron_llm` shows ~0 queue time — TensorRT-LLM's own executor evidently
handles concurrent requests internally rather than needing Triton-level
instance queuing.

**`nemotron_llm.instance_group.count: 2` was tried and reverted** — unlike
`voice_pipeline`, the LLM's weights aren't cheap to duplicate (see "Process
architecture" above: each instance spawns its own MPI worker holding a full
~7.5GB copy). With the GPU already under load-test pressure (~3GB free at
the time), the second instance's engine deserialization hit a CUDA OOM
*during load* — and Triton fails the **whole model** when any instance
fails, so `nemotron_llm` went fully `UNAVAILABLE`, not just running at
reduced capacity. Reverted immediately to restore service. Worth retrying
on a colder GPU with more headroom, but the earlier ASR/TTS finding
generalizes here too: more instances cost real VRAM for real GPU-bound
models, no free lunch.

**Two further optimizations landed after load testing exposed where the
time was actually going**, both about actually using capabilities that were
either unconfigured or silently unused:

1. **Switched the load-testing client (and by implication, any real client)
   from hand-rolled JSON-over-HTTP to `tritonclient`'s native gRPC binary
   protocol.** JSON-encoding tens of thousands of floats as text numbers,
   both directions, was genuinely expensive — confirmed by `kokoro_tts`,
   which had no queueing problem, only serialization: throughput went
   4.57 → **11.96 req/s** (2.6x), p50 0.648s → **0.299s**, purely from the
   protocol change, no server-side change at all. **Not a universal win,
   though** — `whisper_asr`'s p50 barely moved (2.28s → 2.26s), because its
   gap was dominated by *queue time* (1553ms, single instance, no batching),
   not serialization. The lesson: measure the actual gap (client latency vs.
   Triton's own `avg_compute`) before assuming which fix applies.
2. **Enabled `dynamic_batching` on `nemotron_llm`.** Triton was dispatching
   exactly one request per `execute()` call (no batching configured
   anywhere), so `self.llm.generate(prompts, ...)` — despite already being
   written to accept a full list — never received more than one prompt.
   `dynamic_batching { preferred_batch_size: [4, 8]  max_queue_delay_microseconds: 50000 }`
   fixed that: at concurrency=4, Triton now bundles ~4 requests per
   `execute()` call (confirmed: 20 requests → 5 execs, not 20), and
   TensorRT-LLM's in-flight batching scheduler — which was always
   architecturally present, just never fed more than one sequence at a time
   — finally does real work. Result: p50 **6.54s → 1.70s** (3.85x), throughput
   **0.61 → 2.22 req/s** (3.6x), queue time dropped to ~0.

**The full `voice_pipeline` chain barely benefited from the LLM fix**
(p50 5.61s → 5.66s, essentially unchanged), which is itself an informative
result: `dynamic_batching`'s 50ms window only bundles requests that arrive
close together, and hitting `nemotron_llm` directly sends all N test
requests simultaneously — easy to batch. Inside the real chain, each
`voice_pipeline` instance only calls the LLM *after* its own ASR call
finishes, and those finish at staggered times (each queued behind
`whisper_asr`'s single instance), so LLM calls rarely arrive bunched enough
to batch. Net effect (at the time): `whisper_asr`'s queueing was the
dominant bottleneck in the full chain, not the LLM.

**`whisper_asr` got the same treatment** — its `execute()` looped one
request at a time even when Triton bundled several into one call, so
`dynamic_batching` alone wouldn't have helped (unlike `nemotron_llm`, whose
`generate(prompts, ...)` already accepted a full list). Rewrote it to
collect every request's audio into one batched `processor(...)` +
`generate()` call — Whisper's feature extractor already pads every input to
its fixed 30s window regardless of actual length, so batching
differently-sized audio arrays needed no extra padding logic, just passing
a list instead of one array. Added the same `dynamic_batching` config as
the LLM. Isolated result: p50 **2.26s → 0.75s** (3x), throughput
**1.71 → 3.48 req/s** (2x), queue time **1553ms → 5.8ms**. Confirmed via
exec count: 20 requests → 5 execs.

**The full chain still barely moved** (p50 ~5.5s, unchanged) — checking
`whisper_asr`'s own exec count *during* a chained `voice_pipeline` run
showed only 20→17 execs (vs. 20→5 when hit directly), confirming the same
staggering problem that limited the LLM fix's chain impact: only the first
wave of concurrent pipeline requests arrives synchronized enough to batch;
every wave after that drifts apart because each pipeline run takes a
slightly different total time (dominated by the LLM's variable output
length). Both fixes are real, validated, substantial wins when a stage is
hit with genuine concurrent bursts (confirmed 3x/2x for ASR, 3.85x/3.6x for
the LLM) — but that's a ceiling on what config-level batching alone can do
for *this specific* chained, staggered-arrival workload. Going further
would mean restructuring the pipeline itself (e.g. a genuinely
streaming/pipelined architecture instead of strict per-request sequential
ASR→LLM→TTS), not another batching tweak.

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
- `deploy/supervisor/` — the supervisor wrapper scripts + conf.d files for
  the streaming gateway and (new) the Triton server itself, which had been
  running as a manually-launched process with no supervisor entry at all.
- `deploy/REBUILD.md` — the executable runbook: fresh instance -> working
  deployment, including where the source weights come from, how each
  TensorRT engine was built, and which steps are fully scripted vs.
  documented-but-manual.

See `deploy/README.md` for the full breakdown.

## Not yet done

- **`torch-cudagraph`/optimized compile tiers** were only explored for the
  abandoned AutoDeploy path, not revisited for the current classic-engine
  LLM (which is already fast — sub-second generation under load testing too,
  see above — so this wasn't pursued further).
- **`voice_pipeline` staggered arrivals limit config-level batching.**
  Both `nemotron_llm` and `whisper_asr` now batch genuinely well when hit
  directly (3.6x/3.85x, 2x/3x — see above), but the full chain barely
  benefits because each pipeline run takes a different total time, so
  concurrent requests drift out of sync with each other after the first
  wave. Fixing this for real means restructuring the pipeline (e.g.
  streaming/overlapping stages) rather than another `dynamic_batching`
  tweak — a genuinely bigger change, not attempted.
- **`kokoro_tts` is now dead and `magpie_tts` was never actually a working
  replacement — corrected here, see `docs/tts-replacement-investigation.md`
  for the full record.** A prior commit (`f440a32`, "Replace Kokoro with
  Magpie") merged a `magpie_tts` Triton backend into this repo claiming to
  be "confirmed working on this Blackwell (sm_120) GPU" — this line used to
  repeat that claim. It wasn't true: `nemo_toolkit` (the package
  `magpie_tts/1/model.py` imports) was never added to
  `deploy/requirements-main.txt`, the actual deploy directory
  (`speech-cascade-inference/triton_model_repo`) still had `kokoro_tts`, not
  `magpie_tts`, and Magpie was separately investigated per
  `task2-tts.md`/`docs/magpie-tts-investigation.md` and **rejected**
  (confirmed real GPU-bandwidth contention with `nemotron_llm` — 5.2x
  TTFT/3.1x total-latency degradation at concurrency=8 — plus a phonemizer
  bug dropping currency symbols like `$45.99`). Meanwhile `kokoro_tts`
  itself went fully dead on this instance independently of any of that
  (`kokoro-onnx` uninstalled from `/venv/main`, its weight files deleted
  during an unrelated disk-space cleanup) — so TTS had no working backend
  at all for a period. **Replaced with `chatterbox_tts`** (ResembleAI
  Chatterbox-Turbo, MIT license — see the investigation doc for why it won
  over F5-TTS/XTTS-v2/IndexTTS-2.5), run as a subprocess in its own
  `/venv/chatterbox` venv (`triton_model_repo/chatterbox_tts/1/model.py`)
  since Chatterbox's torch/torchaudio pins conflict with `/venv/main`'s
  TensorRT-LLM stack. No TensorRT/ONNX export path exists for it yet either
  — same story as Magpie's autoregressive-decoder-plus-vocoder shape, no
  existing example to port from — that porting work is a deliberate
  follow-up phase, not attempted here. **Actually verified this time**
  (unlike the false "confirmed working" claim this replaces): a real call
  through the live Triton `voice_pipeline` gRPC endpoint produced a real
  transcript, LLM response, and 6.44s of genuine synthesized audio, and a
  15-minute/471-turn composed-pipeline soak test against that same live
  endpoint completed with 0 errors and no leak signature beyond early
  allocator warmup — see `docs/tts-replacement-investigation.md`'s "Task 3"
  section for the exact numbers. Not yet tuned for concurrency
  (`instance_group.count: 1`, unlike `kokoro_tts`'s measured `count: 4`).
- **`whisper_asr` migrated off ONNX Runtime to a vendored TensorRT-LLM
  runtime (`WhisperTRTLLM`, see `deploy/REBUILD.md` 4c)** — this resolves
  what used to be listed here as a "not yet done" TensorRT item for ASR.
  Whether the same FP32-vs-quantized-precision question below still applies
  under the new TensorRT-LLM engines (vs. the old ONNX weights) hasn't been
  revisited.
- **ASR/TTS precision.** The old ONNX-based `whisper_asr`/`kokoro_tts`
  defaulted to FP32 ONNX weights; smaller precision variants (INT8, `q8f16`,
  etc.) existed in the same HF repos and were never tried (FP16 specifically
  produced NaN output on this GPU/onnxruntime combination — see above). Now
  moot for `whisper_asr` (different runtime entirely) and for `chatterbox_tts`
  (different model, plain PyTorch, no quantized variant tried), but worth
  revisiting once chatterbox_tts has a compiled path.
- **TensorRT-LLM engine build flags left at near-defaults.** `reduce_fusion`,
  `multiple_profiles`, and `use_fp8_context_fmha` were all logged as
  disabled at `trtllm-build` time. The last one needs FP8 KV cache
  quantization too (currently only weights are FP8) — worth doing together
  in a rebuild.
- **No Grafana** — Prometheus is up and scraping (see above), but there's no
  dashboard on top of it, just direct PromQL queries or the Prometheus UI's
  own graph tab.
