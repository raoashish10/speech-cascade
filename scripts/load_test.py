"""Load test voice_pipeline, cross-referenced against Triton's own Prometheus
metrics (per-model exec counts/durations, GPU utilization) rather than just
client-side timing alone.

Usage:
    python3 load_test.py --concurrency 4 --total-requests 40
"""

import argparse
import asyncio
import json
import time

import httpx
import soundfile as sf

TRITON_URL = "http://localhost:18000"
METRICS_URL = "http://localhost:18002/metrics"
AUDIO_PATH = "/workspace/speech-cascade-inference/scripts/test_tts_output.wav"


def build_payload():
    audio, sr = sf.read(AUDIO_PATH)
    data = audio.astype("float32").tolist()
    return {
        "inputs": [
            {"name": "AUDIO_SAMPLES", "shape": [len(data)], "datatype": "FP32", "data": data},
            {"name": "SAMPLE_RATE", "shape": [1], "datatype": "INT32", "data": [sr]},
        ]
    }


def parse_metrics(text):
    """Minimal Prometheus text-format parser -- just the counters/gauges we need."""
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name_labels, value = line.rsplit(" ", 1)
        if "{" in name_labels:
            name, labels = name_labels.split("{", 1)
            labels = labels.rstrip("}")
        else:
            name, labels = name_labels, ""
        metrics.setdefault(name, []).append((labels, float(value)))
    return metrics


def get_metric(metrics, name, label_substr=""):
    for labels, value in metrics.get(name, []):
        if label_substr in labels:
            return value
    return None


async def fire_request(client, payload, results, sem):
    async with sem:
        t0 = time.time()
        try:
            resp = await client.post(
                f"{TRITON_URL}/v2/models/voice_pipeline/infer", json=payload, timeout=60.0
            )
            elapsed = time.time() - t0
            body = resp.json()
            ok = resp.status_code == 200 and "error" not in body
            results.append((elapsed, ok, None if ok else body.get("error", f"HTTP {resp.status_code}")))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((elapsed, False, str(e)))


async def main(concurrency, total_requests):
    payload = build_payload()
    print(f"Payload built: {len(payload['inputs'][0]['data'])} audio samples", flush=True)

    async with httpx.AsyncClient() as client:
        before = parse_metrics((await client.get(METRICS_URL)).text)

        sem = asyncio.Semaphore(concurrency)
        results = []
        wall_t0 = time.time()
        tasks = [fire_request(client, payload, results, sem) for _ in range(total_requests)]
        await asyncio.gather(*tasks)
        wall_elapsed = time.time() - wall_t0

        after = parse_metrics((await client.get(METRICS_URL)).text)

    latencies = sorted(r[0] for r in results)
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    def pct(p):
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx]

    print("\n=== Client-observed results ===")
    print(f"concurrency={concurrency} total_requests={total_requests}")
    print(f"wall time: {wall_elapsed:.2f}s")
    print(f"successes: {len(successes)}  failures: {len(failures)}")
    print(f"throughput: {len(successes) / wall_elapsed:.3f} req/s")
    print(f"latency  min={latencies[0]:.2f}s  p50={pct(0.50):.2f}s  p90={pct(0.90):.2f}s  "
          f"p95={pct(0.95):.2f}s  p99={pct(0.99):.2f}s  max={latencies[-1]:.2f}s")
    if failures:
        print(f"\nfirst failure: {failures[0][2]}")

    print("\n=== Triton server-side metrics (delta over the test window) ===")
    for model in ["voice_pipeline", "whisper_asr", "nemotron_llm", "kokoro_tts"]:
        exec_before = get_metric(before, "nv_inference_exec_count", f'model="{model}"') or 0
        exec_after = get_metric(after, "nv_inference_exec_count", f'model="{model}"') or 0
        dur_before = get_metric(before, "nv_inference_compute_infer_duration_us", f'model="{model}"') or 0
        dur_after = get_metric(after, "nv_inference_compute_infer_duration_us", f'model="{model}"') or 0
        queue_before = get_metric(before, "nv_inference_queue_duration_us", f'model="{model}"') or 0
        queue_after = get_metric(after, "nv_inference_queue_duration_us", f'model="{model}"') or 0
        d_exec = exec_after - exec_before
        d_dur_us = dur_after - dur_before
        d_queue_us = queue_after - queue_before
        avg_compute_ms = (d_dur_us / d_exec / 1000) if d_exec else 0
        avg_queue_ms = (d_queue_us / d_exec / 1000) if d_exec else 0
        print(f"{model:15s} execs={int(d_exec):4d}  avg_compute={avg_compute_ms:8.1f}ms  avg_queue={avg_queue_ms:8.1f}ms")

    gpu_before = get_metric(before, "nv_gpu_memory_used_bytes") or 0
    gpu_after = get_metric(after, "nv_gpu_memory_used_bytes") or 0
    print(f"\nGPU memory used: {gpu_before/1e9:.2f}GB -> {gpu_after/1e9:.2f}GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total-requests", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.total_requests))
