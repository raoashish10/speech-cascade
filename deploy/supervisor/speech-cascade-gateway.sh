#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

cd /workspace/speech-cascade-inference
source /venv/gateway/bin/activate
# Default 1MB WS frame limit is too small for base64-encoded float32 TTS
# audio chunks (a several-second sentence alone can exceed it).
pty uvicorn streaming_gateway.server:app --host 127.0.0.1 --port 18010 --ws-max-size 20971520 2>&1
