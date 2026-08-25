#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/speech-cascade-inference
pty /venv/main/bin/python3 /workspace/speech-cascade-inference/scripts/alert_notifier.py 2>&1
