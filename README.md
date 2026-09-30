# Speech Cascade Inference

A voice pipeline (ASR -> LLM -> TTS) served through NVIDIA Triton Inference
Server: four Triton BLS python-backend models, chained end to end, all
GPU-accelerated.

## Requirements

- An NVIDIA GPU with enough VRAM to hold all four models concurrently
  (tested on a 16GB card).
- CUDA 12.8+ and a matching driver — or Docker + `nvidia-container-toolkit`.
  [`docker/`](docker/README.md) is a containerized alternative to the
  bare-metal deployment, **verified end-to-end on a real GPU** (Runpod pod,
  RTX PRO 4500 Blackwell): all four models reach Triton state `READY` and
  the full `voice_pipeline` ensemble round-trips speech -> ASR -> LLM -> TTS
  -> speech. See that directory's README for build status and what's still
  open there.
- Runs either bare-metal (see `deploy/REBUILD.md` for the environment fixes
  that requires on a plain Ubuntu 24.04 base image) or via the Docker images
  above; `.gitignore`/`deploy/REBUILD.md` cover where model weights and
  compiled engines live (they're not part of this git repo).

## Architecture

| Model | Backend | What it wraps | Input -> Output |
|---|---|---|---|
| `qwen_llm` | python | Qwen3-8B-NVFP4, loaded via TensorRT-LLM's `LLM` API (classic backend, JIT graph build at load time — not an AOT-compiled `.engine`). Decoupled/streaming (`generate_async`, one per request on a bounded thread pool), with an admission gate (`admission.py`) that rejects fast once too many requests are in flight/queued instead of queueing unboundedly | `PROMPT` (string) -> `GENERATED_TEXT` (string), streamed |
| `whisper_asr` | python | `WhisperTRTLLM` (vendored TensorRT-LLM Whisper runtime under `triton_model_repo/whisper_asr/1/trtllm_whisper/`, compiled encoder+decoder engines — see `deploy/REBUILD.md` 4c) | `AUDIO_SAMPLES` (float32[]) + optional `SAMPLE_RATE` (int32) -> `TRANSCRIPT` (string) |
| `chatterbox_tts` | python | ResembleAI's Chatterbox-Turbo (`ChatterboxTurboTTS`), run as a subprocess in its own isolated venv (`/venv/chatterbox`). T3's autoregressive decode is batched across concurrent requests and served via vLLM by default (`CHATTERBOX_BACKEND=vllm`; `pytorch` kept as an instant rollback); S3Gen's flow-matching vocoder still runs per-item in plain PyTorch, unbatched — see `deploy/PROFILING.md` and the `config.pbtxt` comments for why | `TEXT` (string) + optional `VOICE` (string, currently ignored — single fixed reference voice) -> `AUDIO_SAMPLES` (float32[]) + `SAMPLE_RATE` (int32) |
| `voice_pipeline` | python | Calls the three above via Triton's in-process BLS API (`pb_utils.InferenceRequest`), turning any downstream admission rejection into a clean per-request error instead of a crash | `AUDIO_SAMPLES` + optional `SAMPLE_RATE`/`VOICE` -> `TRANSCRIPT`, `GENERATED_TEXT`, `AUDIO_SAMPLES`, `SAMPLE_RATE` |

- The first three use Triton's **python backend** as a thin wrapper around a
  Python-level runtime (TensorRT-LLM's `LLM` API, vendored TensorRT-LLM
  Whisper runtime, a subprocess running ResembleAI's `ChatterboxTurboTTS`)
  rather than Triton's native `onnxruntime`/`tensorrt` backends directly.
- `voice_pipeline` is pure orchestration — no model weights of its own, no
  GPU instance needed — chaining the other three into one audio-in/audio-out
  request/response. That's what "Triton BLS" means: business-logic-scripting
  models, either wrapping a runtime or orchestrating other models.
- One gotcha when working on `voice_pipeline`: it has to manually add a
  batch dimension to every tensor it sends to the other three models, and
  manually strip it back off every tensor it gets back — Triton doesn't do
  this for you here. See the comments in
  `triton_model_repo/voice_pipeline/1/model.py` for details.

## Project layout

```
triton_model_repo/          Triton model repository (4 models above):
                             each model's model.py + config.pbtxt
scripts/                    Standalone validation/quantization/load-testing scripts
tests/                      pytest suite (tests/unit/, tests/integration/)
deploy/                     Infra as code: requirements files, supervisor
                             configs, REBUILD.md, ansible/
streaming_gateway/          The external-facing WebSocket gateway in front of Triton
monitoring/                 Grafana dashboards + Prometheus alert rules
docs/                       A few narrative/investigation docs
                             (see docs/README.md for the rest of the story)
.github/workflows/          CI (tests.yml)
```

## Setup

Building the model weights and compiled engines this deployment needs, and
installing the three isolated Python venvs it runs across, is a multi-step
process specific to the target GPU/TensorRT-LLM version — see
`deploy/REBUILD.md` for the full runbook (fresh instance -> working
deployment) and `deploy/README.md` for what's in `deploy/`. Quantizing/
compiling a TensorRT-LLM engine from a different checkpoint uses the same
generic `scripts/quantize_fp8.py`/`quantize_nvfp4.py` -> `trtllm-build`
pipeline; only the resulting `.engine` file is tied to the exact GPU
architecture and TensorRT-LLM version it was built on, so it's rebuilt per
target rather than copied.

## Running the service

```bash
supervisorctl status speech-cascade-triton
supervisorctl restart speech-cascade-triton   # full restart, ~5-6 min (LLM import + engine load)
```

The server runs in **explicit model control mode**: all four models load at
startup, and individual models can be reloaded without restarting the
others or paying the LLM's cold-start cost again:

```bash
curl -X POST http://localhost:18000/v2/repository/models/whisper_asr/load -d '{}'
curl -X POST http://localhost:18000/v2/repository/index   # list loaded models + state
```

Triton itself binds to `127.0.0.1` only (18000 HTTP / 18001 GRPC / 18002
metrics) and is never exposed externally. The streaming gateway
(`streaming_gateway/`) is the sanctioned external surface — it gives a
client everything Triton would (transcript, LLM text, TTS audio) without
handing them raw access to run/reload models. See
`streaming_gateway/README.md` for how it's exposed and authenticated.

### Testing each stage directly

```bash
curl -s -X POST http://localhost:18000/v2/models/qwen_llm/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"PROMPT","shape":[1,1],"datatype":"BYTES","data":["Hello, my name is"]}]}'

curl -s -X POST http://localhost:18000/v2/models/chatterbox_tts/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs":[{"name":"TEXT","shape":[1,1],"datatype":"BYTES","data":["Hello there."]}]}'
```

`scripts/test_asr.py` covers ASR (needs a float32 audio array, awkward to
pass via raw curl). `scripts/` also has the other standalone scripts used
to validate each stage outside Triton (`test_tts.py`, `measure_vram.py`,
`measure_vram_classic.py`, quantization scripts above) if something breaks.

## Configuration

Key per-model tuning knobs, all in each model's `config.pbtxt`:

| Model | Knob | What it controls |
|---|---|---|
| `qwen_llm` | `kv_cache_config.free_gpu_memory_fraction` (`model.py`) | Caps how much free VRAM the LLM's KV cache pool claims — without a cap, TensorRT-LLM grabs most of it by default. |
| `qwen_llm` | `MAX_ADMITTED` (`model.py`) | Admission-gate ceiling on in-flight + queued requests before fast-rejecting instead of queueing unboundedly. |
| `chatterbox_tts` | `vllm_gpu_mem_util` | vLLM's own KV-cache VRAM reservation for T3's decode. |
| `chatterbox_tts` / `whisper_asr` | `max_batch_size` + `dynamic_batching` | Batches concurrent requests through one shared model call instead of serializing them. |
| every model | `instance_group.count` | Replicas per model — cheap for `voice_pipeline` (no weights), expensive for the GPU-resident models (a full extra copy each). |

Rerun `scripts/measure_vram.py` / `scripts/measure_vram_classic.py` (or
`nvidia-smi` while the service is up) for current VRAM numbers on your
hardware rather than trusting a stale table.

## Monitoring

Triton exposes per-model Prometheus metrics at `/metrics` (port 18002,
localhost only). A Prometheus server scrapes it, and a Grafana dashboard
(`monitoring/grafana/`) + alert rules (`monitoring/alert_rules.yml`) sit on
top, along with a custom exporter for Triton's model READY/UNAVAILABLE
state. Grafana is the externally-reachable surface for this data (same
Caddy-authed pattern as the streaming gateway).

`scripts/load_test.py` fires concurrent requests via `tritonclient`'s
native gRPC protocol and reports per-stage latency/throughput:

```bash
python3 scripts/load_test.py --concurrency 4 --total-requests 20
```

For *why* a stage is slow rather than just *how much* time it takes, see
[`deploy/PROFILING.md`](deploy/PROFILING.md) (`torch.profiler` + Nsight
Systems support) and `scripts/profile_nsys.sh`.

## Development

- `tests/unit/` — pure-logic tests, no live server or GPU needed. Runs in
  GitHub Actions CI on every push/PR.
- `tests/integration/` — real gRPC calls against a live Triton server (all
  four models + `voice_pipeline` end-to-end). Needs a live GPU server, so
  it does not run in CI: `python -m pytest tests/integration -v`.

See `tests/README.md` for which venv each tier needs.
