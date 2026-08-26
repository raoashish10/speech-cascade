"""Load test voice_pipeline (or any individual model), cross-referenced
against Triton's own Prometheus metrics (per-model exec counts/durations,
GPU utilization). Triton's metrics only expose sums+counts (averages), not
percentiles -- percentiles here come from client-side timing.

Uses tritonclient's native gRPC binary protocol, not hand-rolled JSON HTTP --
JSON-encoding tens of thousands of floats as text numbers (both directions)
was previously the dominant cost for whisper_asr/voice_pipeline requests,
dwarfing Triton's own reported compute time.

nemotron_llm is decoupled/streaming (see triton_model_repo/nemotron_llm) --
a plain unary infer() call to it fails outright ("ModelInfer RPC doesn't
support models with decoupled transaction policy"). This script measures it
via stream_infer(), timing from request start to the final response chunk,
so its latency numbers stay comparable to the other (unary) models' full-
completion latency.

Time-to-first-token (TTFT): measured only for nemotron_llm, the one model
that actually streams. It's the time from request start to the first
response chunk carrying a non-empty GENERATED_TEXT text_diff (the server
sends one more, empty, chunk at the very end just to close the stream --
that one is not counted as "first token"). whisper_asr, magpie_tts, and
voice_pipeline are unary: the whole response arrives as one atomic reply,
so there's no meaningful sub-request granularity to time -- TTFT and total
response time would be the same number by construction, which is not a
real distinction. Rather than fabricate one, those three report only total
response time; their TTFT csv columns are left blank and the console output
says so explicitly instead of leaving it ambiguous.

Usage:
    python3 load_test.py --concurrency 4 --total-requests 20
        # runs whisper_asr, nemotron_llm, magpie_tts, voice_pipeline in turn,
        # each isolated (no cross-stage GPU contention), same concurrency,
        # for a clean per-stage p50/p90 breakdown.

    python3 load_test.py --concurrency 4 --total-requests 20 --model magpie_tts
        # just one model.

    python3 load_test.py --concurrency 16 --total-requests 64 --model nemotron_llm \
        --csv results.csv
        # appends one row of every metric below to results.csv, for building
        # a batch-size/concurrency sweep table across many separate runs.
        # For nemotron_llm this also fills ttft_p50_s/ttft_p90_s/ttft_p99_s
        # (time-to-first-token, seconds); those three columns are left blank
        # for whisper_asr/magpie_tts/voice_pipeline since those models are
        # unary and have no sub-request first-token event to time separately
        # from total response time (p50_s/p90_s/p99_s above).
"""

import argparse
import asyncio
import csv
import os
import subprocess
import time

import httpx
import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpcclient

TRITON_GRPC_URL = "localhost:18001"
METRICS_URL = "http://localhost:18002/metrics"
AUDIO_PATH = "/workspace/speech-cascade-inference/scripts/test_tts_output.wav"

CSV_FIELDS = [
    "model", "concurrency", "total_requests", "successes", "failures",
    "throughput_req_s", "p50_s", "p90_s", "p99_s", "max_s",
    "ttft_p50_s", "ttft_p90_s", "ttft_p99_s",
    "server_avg_compute_ms", "server_avg_queue_ms", "execs", "avg_batch_size",
    "gpu_mem_used_mb_before", "gpu_mem_used_mb_after", "first_failure",
]


def gpu_mem_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


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
        # max_batch_size: 0 now (decoupled/streaming) -- no leading batch dim.
        arr = np.array(["Hello, my name is".encode("utf-8")], dtype=object)
        inp = grpcclient.InferInput("PROMPT", arr.shape, "BYTES")
        inp.set_data_from_numpy(arr)
        return [inp], [grpcclient.InferRequestedOutput("GENERATED_TEXT")]

    if model == "magpie_tts":
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
    # Sum every matching line rather than returning the first match. Triton
    # exposes each GPU-instance model's counters as TWO lines in /metrics --
    # one tagged with gpu_uuid=... (the real, incrementing value) and one
    # without (stays 0 on this single-GPU box) -- and which one appears
    # first varies by model: nemotron_llm/kokoro_tts have the gpu_uuid line
    # first, but whisper_asr has it second. Returning "the first match" was
    # silently reading the always-0 line for whisper_asr specifically,
    # meaning every prior load_test.py run against whisper_asr (directly or
    # via voice_pipeline's own per-model breakdown) reported avg_compute=0ms/
    # avg_queue=0ms/execs=0 for it -- verified against raw `curl .../metrics`
    # output while investigating voice_pipeline's queueing (see
    # docs/voice-pipeline-queueing.md). Summing is correct under either
    # ordering, since 0 + real == real, and stays correct even if a model
    # genuinely spans multiple GPUs (the real per-GPU values just add up).
    total, seen = 0.0, False
    for labels, value in metrics.get(name, []):
        if label_substr in labels:
            total += value
            seen = True
    return total if seen else None


async def _stream_one_request(client, inputs, outputs, model_name, t0):
    """nemotron_llm is decoupled -- fire a single-request stream_infer call
    and drain every chunk, since a plain infer() is rejected outright.

    Also captures time-to-first-token: the model.py backend (see
    triton_model_repo/nemotron_llm/1/model.py) sends one response per
    non-empty generated text_diff, then a final response with no output
    tensors at all just to close the stream (sender.send(None, flags=
    COMPLETE_FINAL)). TTFT is measured to the first chunk that actually
    carries GENERATED_TEXT data, not that closing chunk.

    Returns ttft (seconds, float) or None if no data-bearing chunk was
    ever seen before the stream ended.
    """
    async def _gen():
        yield {"model_name": model_name, "inputs": inputs, "outputs": outputs}

    got_any = False
    ttft = None
    async for result, error in client.stream_infer(_gen()):
        if error is not None:
            raise RuntimeError(str(error))
        got_any = True
        if ttft is None:
            chunk = result.as_numpy("GENERATED_TEXT")
            if chunk is not None and chunk.size > 0:
                ttft = time.time() - t0
    if not got_any:
        raise RuntimeError("decoupled stream produced no responses")
    return ttft


async def fire_request(client, model, results, sem):
    async with sem:
        inputs, outputs = build_request(model)
        t0 = time.time()
        try:
            if model == "nemotron_llm":
                ttft = await asyncio.wait_for(
                    _stream_one_request(client, inputs, outputs, model, t0), timeout=60.0
                )
            else:
                await client.infer(model_name=model, inputs=inputs, outputs=outputs, client_timeout=60.0)
                ttft = None  # unary: no sub-request first-token event, see module docstring
            elapsed = time.time() - t0
            results.append((elapsed, True, None, ttft))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((elapsed, False, str(e), None))


def pct(latencies, p):
    idx = min(int(len(latencies) * p), len(latencies) - 1)
    return latencies[idx]


def append_csv(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


async def run_one(grpc_client, http_client, model, concurrency, total_requests, csv_path=None):
    before = parse_metrics((await http_client.get(METRICS_URL)).text)
    mem_before = gpu_mem_used_mb()

    sem = asyncio.Semaphore(concurrency)
    results = []
    wall_t0 = time.time()
    tasks = [fire_request(grpc_client, model, results, sem) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    wall_elapsed = time.time() - wall_t0

    mem_after = gpu_mem_used_mb()
    after = parse_metrics((await http_client.get(METRICS_URL)).text)

    latencies = sorted(r[0] for r in results)
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]
    # TTFT only exists for nemotron_llm (streaming); r[3] is None for the
    # three unary models and for any request that never got a data chunk.
    ttfts = sorted(r[3] for r in successes if r[3] is not None)

    exec_before = get_metric(before, "nv_inference_exec_count", f'model="{model}"') or 0
    exec_after = get_metric(after, "nv_inference_exec_count", f'model="{model}"') or 0
    dur_before = get_metric(before, "nv_inference_compute_infer_duration_us", f'model="{model}"') or 0
    dur_after = get_metric(after, "nv_inference_compute_infer_duration_us", f'model="{model}"') or 0
    queue_before = get_metric(before, "nv_inference_queue_duration_us", f'model="{model}"') or 0
    queue_after = get_metric(after, "nv_inference_queue_duration_us", f'model="{model}"') or 0
    d_exec = exec_after - exec_before
    avg_compute_ms = ((dur_after - dur_before) / d_exec / 1000) if d_exec else 0
    avg_queue_ms = ((queue_after - queue_before) / d_exec / 1000) if d_exec else 0
    avg_batch_size = (len(successes) / d_exec) if d_exec else float("nan")

    print(f"\n=== {model} (concurrency={concurrency}, n={total_requests}) ===")
    print(f"successes={len(successes)} failures={len(failures)}  throughput={len(successes)/wall_elapsed:.3f} req/s")
    if latencies:
        print(f"client latency  min={latencies[0]:.3f}s  p50={pct(latencies,0.50):.3f}s  "
              f"p90={pct(latencies,0.90):.3f}s  p99={pct(latencies,0.99):.3f}s  max={latencies[-1]:.3f}s")
    if model == "nemotron_llm":
        if ttfts:
            print(f"TTFT (time-to-first-token, streaming)  p50={pct(ttfts,0.50):.3f}s  "
                  f"p90={pct(ttfts,0.90):.3f}s  p99={pct(ttfts,0.99):.3f}s")
        else:
            print("TTFT: no data-bearing chunk observed on any request")
    else:
        print("TTFT: not reported separately -- unary model, whole response arrives in one "
              "chunk (identical to total response time above by construction)")
    print(f"Triton avg_compute={avg_compute_ms:.1f}ms  avg_queue={avg_queue_ms:.1f}ms  "
          f"execs={int(d_exec)}  avg_batch_size={avg_batch_size:.2f}")
    if mem_before is not None and mem_after is not None:
        print(f"GPU memory used: {mem_before} MiB -> {mem_after} MiB (delta {mem_after - mem_before:+d})")
    if failures:
        print(f"first failure: {failures[0][2][:300]}")

    if csv_path:
        append_csv(csv_path, {
            "model": model, "concurrency": concurrency, "total_requests": total_requests,
            "successes": len(successes), "failures": len(failures),
            "throughput_req_s": round(len(successes) / wall_elapsed, 3) if wall_elapsed else 0,
            "p50_s": round(pct(latencies, 0.50), 4) if latencies else "",
            "p90_s": round(pct(latencies, 0.90), 4) if latencies else "",
            "p99_s": round(pct(latencies, 0.99), 4) if latencies else "",
            "max_s": round(latencies[-1], 4) if latencies else "",
            # Blank (not 0, not duplicated from p50_s/etc.) for the three unary
            # models -- they have no sub-request first-token event to time
            # separately from total response time. See module docstring.
            "ttft_p50_s": round(pct(ttfts, 0.50), 4) if ttfts else "",
            "ttft_p90_s": round(pct(ttfts, 0.90), 4) if ttfts else "",
            "ttft_p99_s": round(pct(ttfts, 0.99), 4) if ttfts else "",
            "server_avg_compute_ms": round(avg_compute_ms, 2),
            "server_avg_queue_ms": round(avg_queue_ms, 2),
            "execs": int(d_exec),
            "avg_batch_size": round(avg_batch_size, 3) if d_exec else "",
            "gpu_mem_used_mb_before": mem_before,
            "gpu_mem_used_mb_after": mem_after,
            "first_failure": failures[0][2][:200].replace("\n", " ") if failures else "",
        })

    return {"model": model, "latencies": latencies, "successes": successes, "failures": failures}


async def main(concurrency, total_requests, model, csv_path):
    models = [model] if model else ["whisper_asr", "nemotron_llm", "magpie_tts", "voice_pipeline"]
    async with httpx.AsyncClient() as http_client:
        grpc_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
        all_results = []
        for m in models:
            all_results.append(await run_one(grpc_client, http_client, m, concurrency, total_requests, csv_path))
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
    parser.add_argument("--model", choices=["whisper_asr", "nemotron_llm", "magpie_tts", "voice_pipeline"], default=None)
    parser.add_argument("--csv", default=None, help="append one summary row per model to this CSV file")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.total_requests, args.model, args.csv))
