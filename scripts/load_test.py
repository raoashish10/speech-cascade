"""Load test voice_pipeline (or any individual model), cross-referenced
against Triton's own Prometheus metrics (per-model exec counts/durations,
GPU utilization). Triton's metrics only expose sums+counts (averages), not
percentiles -- percentiles here come from client-side timing.

Uses tritonclient's native gRPC binary protocol, not hand-rolled JSON HTTP --
JSON-encoding tens of thousands of floats as text numbers (both directions)
was previously the dominant cost for whisper_asr/voice_pipeline requests,
dwarfing Triton's own reported compute time.

Usage:
    python3 load_test.py --concurrency 4 --total-requests 20
        # runs whisper_asr, nemotron_llm, kokoro_tts, voice_pipeline in turn,
        # each isolated (no cross-stage GPU contention), same concurrency,
        # for a clean per-stage p50/p90 breakdown.

    python3 load_test.py --concurrency 4 --total-requests 20 --model kokoro_tts
        # just one model.
"""

import argparse
import asyncio
import time

import httpx
import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpcclient

TRITON_GRPC_URL = "localhost:18001"
METRICS_URL = "http://localhost:18002/metrics"
AUDIO_PATH = "/workspace/speech-cascade-inference/scripts/test_tts_output.wav"


def build_request(model):
    if model == "whisper_asr":
        audio, sr = sf.read(AUDIO_PATH)
        audio = audio.astype(np.float32).reshape(1, -1)
        sr_arr = np.array([[sr]], dtype=np.int32)
        inputs = [grpcclient.InferInput("AUDIO_SAMPLES", audio.shape, "FP32")]
        inputs[0].set_data_from_numpy(audio)
        sr_input = grpcclient.InferInput("SAMPLE_RATE", sr_arr.shape, "INT32")
        sr_input.set_data_from_numpy(sr_arr)
        inputs.append(sr_input)
        return inputs, [grpcclient.InferRequestedOutput("TRANSCRIPT")]

    if model == "nemotron_llm":
        arr = np.array([["Hello, my name is"]], dtype=object)
        inp = grpcclient.InferInput("PROMPT", arr.shape, "BYTES")
        inp.set_data_from_numpy(arr)
        return [inp], [grpcclient.InferRequestedOutput("GENERATED_TEXT")]

    if model == "kokoro_tts":
        arr = np.array([["Hello there, this is a test."]], dtype=object)
        inp = grpcclient.InferInput("TEXT", arr.shape, "BYTES")
        inp.set_data_from_numpy(arr)
        return [inp], [
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ]

    if model == "voice_pipeline":
        audio, sr = sf.read(AUDIO_PATH)
        audio = audio.astype(np.float32)
        sr_arr = np.array([sr], dtype=np.int32)
        inputs = [grpcclient.InferInput("AUDIO_SAMPLES", audio.shape, "FP32")]
        inputs[0].set_data_from_numpy(audio)
        sr_input = grpcclient.InferInput("SAMPLE_RATE", sr_arr.shape, "INT32")
        sr_input.set_data_from_numpy(sr_arr)
        inputs.append(sr_input)
        return inputs, [
            grpcclient.InferRequestedOutput("TRANSCRIPT"),
            grpcclient.InferRequestedOutput("GENERATED_TEXT"),
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ]

    raise ValueError(model)


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


async def fire_request(client, model, results, sem):
    async with sem:
        inputs, outputs = build_request(model)
        t0 = time.time()
        try:
            await client.infer(model_name=model, inputs=inputs, outputs=outputs, client_timeout=60.0)
            elapsed = time.time() - t0
            results.append((elapsed, True, None))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((elapsed, False, str(e)))


def pct(latencies, p):
    idx = min(int(len(latencies) * p), len(latencies) - 1)
    return latencies[idx]


async def run_one(grpc_client, http_client, model, concurrency, total_requests):
    before = parse_metrics((await http_client.get(METRICS_URL)).text)

    sem = asyncio.Semaphore(concurrency)
    results = []
    wall_t0 = time.time()
    tasks = [fire_request(grpc_client, model, results, sem) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    wall_elapsed = time.time() - wall_t0

    after = parse_metrics((await http_client.get(METRICS_URL)).text)

    latencies = sorted(r[0] for r in results)
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    exec_before = get_metric(before, "nv_inference_exec_count", f'model="{model}"') or 0
    exec_after = get_metric(after, "nv_inference_exec_count", f'model="{model}"') or 0
    dur_before = get_metric(before, "nv_inference_compute_infer_duration_us", f'model="{model}"') or 0
    dur_after = get_metric(after, "nv_inference_compute_infer_duration_us", f'model="{model}"') or 0
    queue_before = get_metric(before, "nv_inference_queue_duration_us", f'model="{model}"') or 0
    queue_after = get_metric(after, "nv_inference_queue_duration_us", f'model="{model}"') or 0
    d_exec = exec_after - exec_before
    avg_compute_ms = ((dur_after - dur_before) / d_exec / 1000) if d_exec else 0
    avg_queue_ms = ((queue_after - queue_before) / d_exec / 1000) if d_exec else 0

    print(f"\n=== {model} (concurrency={concurrency}, n={total_requests}) ===")
    print(f"successes={len(successes)} failures={len(failures)}  throughput={len(successes)/wall_elapsed:.3f} req/s")
    if latencies:
        print(f"client latency  min={latencies[0]:.3f}s  p50={pct(latencies,0.50):.3f}s  "
              f"p90={pct(latencies,0.90):.3f}s  p95={pct(latencies,0.95):.3f}s  max={latencies[-1]:.3f}s")
    print(f"Triton avg_compute={avg_compute_ms:.1f}ms  avg_queue={avg_queue_ms:.1f}ms (server-side, execs={int(d_exec)})")
    if failures:
        print(f"first failure: {failures[0][2][:300]}")

    return {"model": model, "latencies": latencies, "successes": successes, "failures": failures}


async def main(concurrency, total_requests, model):
    models = [model] if model else ["whisper_asr", "nemotron_llm", "kokoro_tts", "voice_pipeline"]
    async with httpx.AsyncClient() as http_client:
        grpc_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        all_results = []
        for m in models:
            all_results.append(await run_one(grpc_client, http_client, m, concurrency, total_requests))
        await grpc_client.close()

    if len(all_results) > 1:
        print(f"\n=== p50 breakdown (concurrency={concurrency}) ===")
        for r in all_results:
            lat = r["latencies"]
            p50 = pct(lat, 0.50) if lat else float("nan")
            print(f"{r['model']:15s} p50={p50:.3f}s  ({len(r['successes'])}/{len(r['successes'])+len(r['failures'])} ok)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--total-requests", type=int, default=20)
    parser.add_argument("--model", choices=["whisper_asr", "nemotron_llm", "kokoro_tts", "voice_pipeline"], default=None)
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.total_requests, args.model))
