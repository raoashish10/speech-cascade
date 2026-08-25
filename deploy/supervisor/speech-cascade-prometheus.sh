#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

mkdir -p /workspace/speech-cascade-inference/prometheus_data
cd /workspace/speech-cascade-inference
pty /usr/bin/prometheus \
  --config.file=/workspace/speech-cascade-inference/prometheus.yml \
  --storage.tsdb.path=/workspace/speech-cascade-inference/prometheus_data \
  --web.listen-address=127.0.0.1:9090 \
  --web.enable-lifecycle 2>&1
