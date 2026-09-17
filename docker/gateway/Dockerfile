# syntax=docker/dockerfile:1
#
# Streaming gateway image. CPU-only -- no GPU, no CUDA, nothing model-
# specific. Talks to the triton service over the Docker network
# (TRITON_URL env var, see streaming_gateway/triton_client.py) instead of
# `localhost`, since the two now run in separate containers.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

COPY streaming_gateway/requirements.txt /app/streaming_gateway/requirements.txt

# CPU wheel index first, same reason .github/workflows/tests.yml does this:
# silero-vad depends on torch/torchaudio even in ONNX mode (tensor plumbing
# in VADIterator only, no GPU inference), so pulling the default (CUDA)
# wheel here would just bloat the image for nothing this container ever
# uses.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchaudio \
    && pip install --no-cache-dir -r /app/streaming_gateway/requirements.txt

COPY streaming_gateway /app/streaming_gateway

# Default WS frame limit (1MB) is too small for base64-encoded float32 TTS
# audio chunks -- same flag deploy/supervisor/speech-cascade-gateway.sh
# passes on bare metal.
EXPOSE 18010
CMD ["uvicorn", "streaming_gateway.server:app", "--host", "0.0.0.0", "--port", "18010", "--ws-max-size", "20971520"]
