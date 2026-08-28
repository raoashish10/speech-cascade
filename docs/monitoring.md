# Monitoring and alerting

Grafana dashboard + Prometheus alerting for the 4-model Triton pipeline
(`nemotron_llm`, `whisper_asr`, `chatterbox_tts`, `voice_pipeline`) on this
single-GPU box. Everything here is grounded in this session's actual
load-testing data (`reports/session-report.md`), not generic defaults.

## URLs

- **Grafana**: `http://127.0.0.1:13000` (localhost-only; SSH port-forward
  to reach it externally: `ssh -L 13000:127.0.0.1:13000 <instance>`).
  Login `admin` / `speechcascade` (override via `GRAFANA_ADMIN_PASSWORD`
  in `${WORKSPACE}/.env`, restart `speech-cascade-grafana` to pick it up).
  Dashboard: **Speech Cascade — Triton Monitoring** (also provisioned
  automatically, so it survives a Grafana restart).
- **Prometheus**: `http://127.0.0.1:9090` — raw metrics browser, and
  `/api/v1/alerts` for current alert state, `/api/v1/rules` to confirm
  rules loaded.
- **Triton state exporter**: `http://127.0.0.1:9109/metrics` — republishes
  Triton's model READY/UNAVAILABLE state as `triton_model_ready` (a gauge
  per `{model, version}`), since Triton's own `/metrics` doesn't expose
  this natively; only the HTTP repository-index API does.

## What's on the dashboard

10 panels: a "read me first" text panel, per-model exec throughput,
request failures/sec, queue time and compute time per model (both
**excluding `nemotron_llm`** — see below for why), `nemotron_llm`
throughput on its own panel, GPU utilization, model READY state, GPU
memory used-vs-free, and a GPU free-memory redline gauge.

## Why `nemotron_llm` is excluded from queue/compute panels and alerts

`nemotron_llm` is decoupled/streaming: its `execute()` returns almost
instantly after handing the request to a background thread pool, so
Triton's own per-exec queue/compute duration timers no longer capture
real generation time or real backend saturation for it. Charting or
alerting on those two metrics for `nemotron_llm` would show a healthy
number even when its thread pool is the actual bottleneck — worse than
no signal, a *misleading* one. Its request throughput and failure counts
are still real and are charted normally.

## Alerts (Prometheus-evaluated, see `monitoring/alert_rules.yml`)

No Alertmanager — see `scripts/alert_notifier.py`'s docstring for why
(single-box, single-operator deployment, no existing notification
channel; the script polls Prometheus's `/api/v1/alerts` and logs every
FIRING/RESOLVED transition to a supervisor-managed log, with an optional
webhook via `ALERT_WEBHOOK_URL` in `.env`).

| Alert | Fires when | What it means | What to do |
|---|---|---|---|
| `TritonQueueTimeHigh` | avg queue time > 2s for 30s, any model except `nemotron_llm` | Early warning — this session's load test showed queue time near 0ms up to concurrency 4, crossing 2s only once a stage started saturating, well before client-visible failures | Check the dashboard's queue-time panel to see which stage. If load is expected, consider `instance_group.count` or batching for that model. If not, look for a stuck downstream call or a competing GPU process |
| `TritonQueueTimeCritical` | avg queue time > 5s for 30s | In this session's testing, 5s+ queue time reliably preceded client-visible failures as concurrency climbed further | Active degradation — shed load if you control the client. Don't reload/unload a model to try to fix this without checking whether another session already owns in-progress model changes |
| `GPUFreeMemoryLow` | free VRAM < 2GiB for 30s | Steady-state with all 4 models loaded normally leaves ~5.5GB free; this suggests real memory pressure building | `nvidia-smi` to see what's using it. Don't start a model reload yourself while free memory is already this tight |
| `GPUFreeMemoryCritical` | free VRAM < 1GiB for 15s | The one real OOM this session happened with free memory at 26-424MB during a model *reload* (not steady request load) — this gives real advance warning | Immediate attention. If a load/reload is already in progress, let it finish or fail rather than adding more GPU load |
| `TritonModelNotReady` | a model reports not-READY for 20s+ | The 20s window absorbs a normal reload's brief UNAVAILABLE blip; sustained past that means the model failed to (re)load | `curl -X POST http://localhost:18000/v2/repository/index -d '{}'` for exact state, check the Triton server log for the load error |

Note on the GPU memory thresholds: `nvidia-smi`'s advertised
`memory.total` (16303 MiB) over-reports the real usable ceiling —
`memory.used + memory.free` only ever summed to ~15843 MiB in practice
this session (a ~460MiB reserved gap). The alert expressions subtract
`nv_gpu_memory_used_bytes` from that empirically observed 15843 MiB
ceiling, not the advertised total — using the advertised total would
overstate real headroom by ~460MiB, which matters right at the redline.

## Services (all supervisor-managed, `supervisorctl status` / restart by name)

`speech-cascade-prometheus`, `speech-cascade-grafana`,
`speech-cascade-triton-state-exporter`, `speech-cascade-alert-notifier`.
Config/data under `monitoring/` and `prometheus_data/` in this repo's
working tree (not git-tracked — weights-adjacent, same as everything
else under `speech-cascade-inference/`).
