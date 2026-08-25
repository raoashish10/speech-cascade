#!/usr/bin/env python3
"""Small Prometheus exporter for Triton model READY/UNAVAILABLE state.

Triton's own /metrics endpoint (18002) does not expose per-model
load-state as a Prometheus metric -- the only way to get it is the HTTP
repository-index API (`POST /v2/repository/index`, a read-only listing
call, not a load/unload/reload action). This script polls that endpoint
on an interval and republishes the result as a gauge, so it can be
scraped like any other target, graphed in Grafana, and alerted on with a
normal Prometheus rule (see monitoring/alert_rules.yml,
TritonModelNotReady).

Run as its own tiny supervisor service (see
/opt/supervisor-scripts/triton-state-exporter.sh) on an internal-only
port; it is scraped by prometheus.yml's `triton_state_exporter` job.

Env vars:
    TRITON_HTTP_URL              default http://127.0.0.1:18000
    TRITON_STATE_POLL_INTERVAL   seconds between polls, default 10
    TRITON_STATE_EXPORTER_PORT   port to serve /metrics on, default 9109
"""

import os
import time

import requests
from prometheus_client import Gauge, start_http_server

TRITON_URL = os.environ.get("TRITON_HTTP_URL", "http://127.0.0.1:18000")
POLL_INTERVAL_SECONDS = float(os.environ.get("TRITON_STATE_POLL_INTERVAL", "10"))
LISTEN_PORT = int(os.environ.get("TRITON_STATE_EXPORTER_PORT", "9109"))

model_ready = Gauge(
    "triton_model_ready",
    "1 if Triton's repository index reports this model version as READY, "
    "0 for any other state (UNAVAILABLE, LOADING, UNLOADING, ...).",
    ["model", "version"],
)
exporter_up = Gauge(
    "triton_state_exporter_up",
    "1 if the most recent poll of Triton's repository index API succeeded, 0 if it failed.",
)


def poll_once() -> None:
    try:
        resp = requests.post(f"{TRITON_URL}/v2/repository/index", json={}, timeout=5)
        resp.raise_for_status()
        entries = resp.json()
    except Exception as exc:  # noqa: BLE001 - keep polling regardless of failure mode
        print(f"[triton_state_exporter] poll failed: {exc}", flush=True)
        exporter_up.set(0)
        return

    for entry in entries:
        name = entry.get("name", "unknown")
        version = str(entry.get("version", "1"))
        state = entry.get("state", "UNKNOWN")
        model_ready.labels(model=name, version=version).set(1.0 if state == "READY" else 0.0)
    exporter_up.set(1)


def main() -> None:
    start_http_server(LISTEN_PORT, addr="127.0.0.1")
    print(
        f"[triton_state_exporter] serving /metrics on 127.0.0.1:{LISTEN_PORT}, "
        f"polling {TRITON_URL} every {POLL_INTERVAL_SECONDS}s",
        flush=True,
    )
    while True:
        poll_once()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
