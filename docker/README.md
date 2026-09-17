# docker/ — containerized deployment (no Vast.ai)

An alternative to `deploy/`'s bare-metal runbook: the same four-model
Triton voice pipeline + streaming gateway, packaged as two containers
(`docker/triton/`, `docker/gateway/`) run via the root `docker-compose.yml`,
instead of hand-installed onto a specific Vast.ai instance via supervisor.

**Why this exists**: `deploy/REBUILD.md` and the main `README.md`'s
"Environment quirks fixed to make this run bare-metal" section document ten
hand-fixed environment issues (MPI, CUDA runtime libs, `PYTHONHOME`/`PATH`
fights, missing `libssl.so.1.1`, missing DCGM, a cuBLAS version clash, numpy
pin drift, ...) and say outright none of it is needed inside NVIDIA's NGC
containers, which ship a matched toolchain. `docker/triton/Dockerfile`
builds on one of those images instead of reassembling the toolchain by
hand, which is what makes this portable to any GPU host with Docker +
`nvidia-container-toolkit` — not just this one Vast.ai instance.

**Build status**: both images now build successfully and are pushed to
`ghcr.io/raoashish10/speech-cascade-gateway:latest` and
`ghcr.io/raoashish10/speech-cascade-triton:latest` — verified by actually
building them (not just reading the Dockerfile) on a Runpod CPU pod,
since Runpod pods don't support Docker-in-Docker (the sandbox blocks the
`unshare()`/`clone()` syscalls nested containers need — confirmed the hard
way; `kaniko`, which builds via chroot instead of a daemon, was used in
place of `docker build`). That process found and fixed 8 real bugs in
`docker/triton/Dockerfile` — missing `--extra-index-url` for `+cu128`
torch wheels, a `numpy`/`librosa` pip resolver conflict, a
`resolution-too-deep` error, a genuinely broken `_ssl` module in this NGC
image's `/usr/local/bin/python3` (bridged around with a `.pth` file), and
a `chatterbox-tts` pin conflict, among others — see the Dockerfile's own
inline comments for the full detail on each.

**Verified end-to-end on a real GPU** (RTX PRO 4500 Blackwell, sm_120,
Runpod): all four models reach Triton state `READY` and the full
`voice_pipeline` ensemble round-trips speech -> ASR -> LLM -> TTS -> speech.
A 11.0s input clip produced a correct transcript, a coherent LLM reply, and
7.0s of generated 24kHz audio in 13.2s wall time, with weights/engines built
from scratch (no S3) per `scripts/build_models_from_scratch.sh`.

Environment bugs found and fixed by that GPU run, none of which a
successful `docker build` would have caught:
- the original base image (`nvcr.io/nvidia/tensorrt-llm/release`) shipped no
  `tritonserver` binary at all -- switched to
  `nvcr.io/nvidia/tritonserver:<date>-trtllm-python-py3`
- `OPAL_PREFIX` (this image's OpenMPI can't find its own runtime data, which
  kills `tensorrt_llm` and therefore `qwen_llm`)
- PyPI's pinned `tensorrt`/`tensorrt_cu13` wheels are metadata-only and
  *delete* the base image's working `tensorrt` module -- restored from
  NVIDIA's index
- `chatterbox_tts` defaults to a `CHATTERBOX_BACKEND=vllm` path needing a
  `/venv/vllm` and a model export that don't exist in this repo; pinned to
  the `pytorch` backend instead

**Still unverified**: the `vllm` chatterbox backend (~5x faster per its own
docstring) -- see `scripts/export_chatterbox_t3_for_vllm.py`, a from-scratch
reconstruction of the missing T3 export step, never run. Also the
`gateway` container against a live `triton` (only the triton half was
exercised directly), and sustained/concurrent load.

## Quick start

Weights and compiled engines are **not** in the image -- they live in the
`inference-data` volume `docker-compose.yml` mounts at
`/workspace/speech-cascade-inference/models`. Populate it either by
restoring from S3 (set `S3_BUCKET_URI`, see `docker/triton/entrypoint.sh`)
or from scratch with no S3 at all:

```bash
docker compose run --rm triton bash /workspace/speech-cascade/scripts/build_models_from_scratch.sh
```

That script downloads the Qwen NVFP4 checkpoint and builds whisper's
TensorRT-LLM engines on whatever GPU it runs on -- verified end-to-end on
an RTX PRO 4500 (both models reached Triton state `READY`). Engines stay
out of the image on purpose: they're GPU-architecture-specific, so baking
them in would tie the image to one card. `chatterbox_tts` additionally
needs its reference voice clip (`ref_audio_path`), which is a deployment
artifact you supply yourself.

```bash
cp .env.example .env   # fill in S3 creds (or skip and populate the volume yourself) + GATEWAY_AUTH_TOKEN
docker compose up --build
```

Requires a host with an NVIDIA GPU and `nvidia-container-toolkit` installed
(`nvidia-ctk runtime configure --runtime=docker`, or whatever your provider's
own Docker-native image/template already sets up — RunPod and SaladCloud
both do, per the "which GPU provider" discussion this came out of).

`triton` takes ~5-6 minutes to become healthy (TensorRT-LLM import + engine
load, same cost `README.md`'s "Managing the service" section documents for
the bare-metal deployment) — `gateway` waits on `service_healthy` before
starting.

## What's different from the bare-metal deployment

| | Bare metal (`deploy/`) | Docker (`docker/`) |
|---|---|---|
| Toolchain | Hand-assembled to match the driver/CUDA on one specific instance | Comes matched from the NGC base image |
| Process manager | `supervisor` | `docker compose` (`restart: unless-stopped`) |
| Triton <-> gateway | both bind `127.0.0.1` on the same box | separate containers, `TRITON_URL=triton:18001` over the compose network |
| External auth | Vast.ai's Caddy edge checks `$OPEN_BUTTON_TOKEN` in front of the gateway | `GATEWAY_AUTH_TOKEN` checked by the gateway itself (see `streaming_gateway/server.py`) — no Vast.ai-specific edge to depend on |
| Weights/engines | restored via `aws s3 sync` into `/workspace` directly | same `aws s3 sync`, run by `docker/triton/entrypoint.sh` into a named volume |
| Monitoring stack (Prometheus/Grafana/exporters) | supervisor services, see `monitoring/` | **not containerized yet** — see "Not yet done" below |

Everything under `triton_model_repo/*/config.pbtxt` still points at
absolute paths like `/workspace/speech-cascade-inference/models/...` —
deliberately unchanged, since `docker/triton/Dockerfile` reproduces that
same layout inside the image/volume rather than editing every config file.

## Auth

The bare-metal deployment relies on Vast.ai's own Caddy reverse-proxy edge
to check a shared token in front of the gateway (see
`streaming_gateway/README.md`). That edge doesn't exist here, so
`streaming_gateway/server.py` now has its own optional shared-token check
(`GATEWAY_AUTH_TOKEN` env var, `?token=` query param or `Authorization:
Bearer` header — same client-facing convention). Leave it unset only if
something else in front of the gateway (a reverse proxy you add, a VPN,
your provider's own auth) is already handling this — an empty token means
the check is skipped entirely, same as today's bare-metal behavior with no
Caddy in front of it.

## GPU portability caveat

Moving off the current Vast.ai RTX 5070 Ti instance doesn't make the
*compiled TensorRT engines* portable — `README.md`'s "Testing each stage"
and `scripts/build_engine.sh`'s own header both say the `.engine` output is
tied to the exact GPU architecture + TensorRT-LLM version it was built
with. A different GPU (even another Blackwell card — RTX 5080/5090 instead
of 5070 Ti) needs `scripts/build_engine.sh` / `scripts/quantize_*.py` rerun
against it (see `deploy/REBUILD.md` section 4, or `deploy/ansible`'s
`build_source: scratch`), not the `.engine` files copied over. Docker
solves the toolchain-matching problem, not the engine-portability one.

## Not yet done

- Monitoring stack (`monitoring/`, `prometheus.yml`, Grafana dashboards,
  the alert-notifier and Triton-state-exporter supervisor services) isn't
  containerized here yet — still only documented for the bare-metal/
  supervisor deployment.
- No CI build/push of these images (`.github/workflows/tests.yml` only runs
  the CPU-only unit tests, same as before).
- `docker/triton/entrypoint.sh`'s sequential model-load order carries over
  the bare-metal OOM workaround (see its own comments) without
  re-verifying the memory ceiling that caused it against this image/host.
