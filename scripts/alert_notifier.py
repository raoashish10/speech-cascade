#!/usr/bin/env python3
"""Lightweight alert notifier for the speech-cascade monitoring stack.

Why this instead of Alertmanager: this is a single-box, single-operator
research/demo deployment with no existing notification channel wired up
(no Slack/PagerDuty/SMTP credentials configured on this instance -- see
`vast-capabilities | jq .credentials`). Alertmanager's actual value --
routing, grouping, and de-duplicating alerts across multiple receivers and
on-call operators -- doesn't apply when there's one box and one person who
would ever look at it. Standing up Alertmanager's separate config/routing
tree for that would be complexity with no payoff.

Prometheus already evaluates the rules in monitoring/alert_rules.yml and
exposes current firing/pending state at /api/v1/alerts. This script just
polls that on an interval, logs every state transition (FIRING/RESOLVED)
to a supervisor-managed, portal-visible log -- so "someone has to be
staring at terminal output" stops being the detection mechanism -- and
optionally forwards each transition to a webhook if one is configured.

If this ever needs real multi-channel routing/on-call escalation, swap in
Alertmanager without touching anything upstream: the alerting rules
already live in Prometheus, this script and Alertmanager would be
consuming the exact same evaluated state.

Env vars:
    PROMETHEUS_URL      default http://127.0.0.1:9090
    ALERT_POLL_INTERVAL seconds between polls, default 10
    ALERT_WEBHOOK_URL   optional; if set, POSTs a small JSON payload
                         ({"text": "..."}) -- Slack-incoming-webhook
                         compatible -- to this URL on every FIRING/RESOLVED
                         transition. Set it in /workspace/.env and restart
                         the alert-notifier supervisor service to pick it up.
"""

import os
import time
from datetime import datetime, timezone

import requests

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090")
POLL_INTERVAL_SECONDS = float(os.environ.get("ALERT_POLL_INTERVAL", "10"))
WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "").strip()


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{ts} {msg}", flush=True)


def fetch_active_alerts() -> list:
    resp = requests.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=5)
    resp.raise_for_status()
    return resp.json()["data"]["alerts"]


def alert_key(alert: dict):
    labels = alert.get("labels", {})
    return (labels.get("alertname"), tuple(sorted(labels.items())))


def notify_webhook(event: str, alert: dict) -> None:
    if not WEBHOOK_URL:
        return
    labels = alert.get("labels", {})
    annotations = alert.get("annotations", {})
    text = (
        f"[{event}] {labels.get('alertname')} (severity={labels.get('severity', '?')}) "
        f"model={labels.get('model', '-')}: {annotations.get('summary', '')}"
    )
    try:
        requests.post(WEBHOOK_URL, json={"text": text}, timeout=5)
    except Exception as exc:  # noqa: BLE001 - never let a webhook failure kill the loop
        log(f"webhook POST failed: {exc}")


def main() -> None:
    log(
        f"alert-notifier starting, polling {PROMETHEUS_URL}/api/v1/alerts every "
        f"{POLL_INTERVAL_SECONDS}s"
        + (", forwarding transitions to webhook" if WEBHOOK_URL else ", no webhook configured (log-only)")
    )
    firing: dict = {}  # alert_key -> alert dict, only entries currently in state "firing"

    while True:
        try:
            alerts = fetch_active_alerts()
        except Exception as exc:  # noqa: BLE001
            log(f"poll failed: {exc}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        current_firing = {alert_key(a): a for a in alerts if a.get("state") == "firing"}

        for key, alert in current_firing.items():
            if key not in firing:
                labels = alert.get("labels", {})
                annotations = alert.get("annotations", {})
                log(
                    f"FIRING   {labels.get('alertname')} severity={labels.get('severity', '?')} "
                    f"model={labels.get('model', '-')} :: {annotations.get('summary', '')} "
                    f":: ACTION: {annotations.get('action', '')}"
                )
                notify_webhook("FIRING", alert)

        for key, alert in list(firing.items()):
            if key not in current_firing:
                labels = alert.get("labels", {})
                log(f"RESOLVED {labels.get('alertname')} model={labels.get('model', '-')}")
                notify_webhook("RESOLVED", alert)

        firing = current_firing
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
