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

**UNVERIFIED end-to-end**: written and reviewed, but never built or run
against a real GPU (the environment this was written in has neither a GPU
nor Docker-in-Docker). Same posture as `deploy/REBUILD.md`'s own "Known
gaps" section — treat the first real build as the actual verification, not
this directory's existence. See `docker/triton/Dockerfile`'s header for the
specific things to double-check first (the exact NGC base image tag, and
whether `python3 -m venv` inside it produces a complete stdlib the way
README.md quirk #3 says bare Ubuntu's doesn't).

## Quick start

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
