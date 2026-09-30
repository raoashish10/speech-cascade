"""End-to-end soak test for the `voice_pipeline` endpoint only.

This is an EXTENSION of `load_test.py`, not a replacement -- it reuses that
module's request builders, Triton metrics parsing, and CSV field
conventions, and adds exactly what the whole-cascade soak-test brief needs
that load_test.py doesn't have:

  1. Sustained duration per concurrency point (load_test.py fires a fixed
     `--total-requests` and stops; a soak test needs continuous load for
     real wall-clock minutes so leak/degradation classes of bug can surface).
  2. A concurrency SWEEP in one invocation (load_test.py takes one
     `--concurrency` value per run).
  3. Continuous GPU utilization sampling (`nvidia-smi dmon`), not just a
     before/after memory snapshot.
  4. Continuous host CPU utilization + host RAM sampling (psutil) --
     load_test.py has neither; this project's own docs flag host-CPU-bound
     kernel-launch overhead and a host-RAM leak class of bug as the reasons
     this matters, not GPU stats alone.
  5. A `config_label` CSV column (e.g. "accelerated_vllm") and a Grafana
     annotation pushed at the start of each concurrency point's window, so
     dashboard exports/screenshots are unambiguous about which run they
     belong to.

HARD REQUIREMENT carried over from the brief: only `voice_pipeline` (the
real end-to-end ASR->LLM->TTS round trip) is measured as the headline
number. Triton's own per-model metrics for whisper_asr/qwen_llm/
chatterbox_tts are pulled ONLY as a supplementary queue-time breakdown
(qwen_llm's own queue metric is known-unreliable -- decoupled/streaming,
see docs/voice-pipeline-queueing.md and qwen_llm's config.pbtxt comment --
so it is fetched but explicitly never used as the headline queue number).

Usage:
    python3 pipeline_soak_test.py --config-label accelerated_vllm \\
        --concurrency-sweep 1,4,8,16 --duration-s 900 \\
        --csv soak_results_accelerated.csv --outdir soak_out/accelerated
"""

import argparse
import asyncio
import csv as csvmod
import json
import os
import subprocess
import sys
import threading
import time

import httpx
import psutil
import tritonclient.grpc.aio as grpcclient

sys.path.insert(0, os.path.dirname(__file__))
from load_test import (  # noqa: E402
    TRITON_GRPC_URL, METRICS_URL, build_request, parse_metrics, get_metric, pct,
)

GRAFANA_URL = "http://127.0.0.1:13000"
GRAFANA_ANNOTATIONS_API = f"{GRAFANA_URL}/api/annotations"

SOAK_CSV_FIELDS = [
    "config_label", "concurrency", "duration_s", "warmup_requests",
    "total_requests", "successes", "failures",
    "throughput_req_s", "p50_s", "p90_s", "p95_s", "p99_s", "max_s",
    # voice_pipeline's OWN Triton-native queue/compute (BLS-level only --
    # does not see queueing inside whisper_asr/qwen_llm/chatterbox_tts,
    # which happens inside voice_pipeline's own execute() call and is
    # counted as voice_pipeline "compute" time from Triton's point of view).
    "vp_server_avg_compute_ms", "vp_server_avg_queue_ms", "vp_execs",
    # Supplementary per-model breakdown -- whisper_asr and chatterbox_tts's
    # OWN queue metrics ARE reliable (unary, non-decoupled). qwen_llm's own
    # queue metric is NOT reliable (decoupled/streaming -- see module
    # docstring) and is intentionally excluded from this CSV entirely so it
    # can't be misread as a real number.
    "whisper_asr_avg_queue_ms", "chatterbox_tts_avg_queue_ms",
    "gpu_util_avg_pct", "gpu_util_max_pct",
    "gpu_mem_used_mb_avg", "gpu_mem_used_mb_max",
    "host_cpu_util_avg_pct", "host_cpu_util_max_pct",
    "host_ram_used_mb_start", "host_ram_used_mb_end", "host_ram_delta_mb",
    "first_failure",
]


def push_grafana_annotation(text, tags, grafana_auth=None):
    try:
        payload = {"text": text, "tags": tags, "time": int(time.time() * 1000)}
        headers = {}
        if grafana_auth:
            headers["Authorization"] = f"Basic {grafana_auth}"
        r = httpx.post(GRAFANA_ANNOTATIONS_API, json=payload, headers=headers, timeout=5.0)
        return r.status_code in (200, 201)
    except Exception as e:
        print(f"[warn] Grafana annotation push failed (non-fatal): {e}")
        return False


class GpuSampler:
    """Background `nvidia-smi dmon` sampler -- GPU util % and mem used MB,
    once per second, for the whole soak window (not a single point read)."""

    def __init__(self, interval_s=1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = []  # list of (util_pct, mem_used_mb)
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                util_s, mem_s = out.stdout.strip().split(",")
                self.samples.append((float(util_s), float(mem_s)))
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return None, None, None, None
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return (sum(utils) / len(utils), max(utils), sum(mems) / len(mems), max(mems))


class CpuSampler:
    """Background host-CPU-utilization sampler via psutil, once per second.

    Directly relevant per this project's own diagnosis: the unaccelerated
    chatterbox_tts path was found to be CPU-dispatch-bound (885,262
    cudaLaunchKernel calls, 41.2% of CUDA API time in an nsys trace) --
    this is the system-level, independent confirmation of that mechanism,
    not a substitute for the nsys trace itself.
    """

    def __init__(self, interval_s=1):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = []
        self._thread = None

    def _run(self):
        psutil.cpu_percent(interval=None)  # prime it, first call is always 0
        while not self._stop.is_set():
            self.samples.append(psutil.cpu_percent(interval=None))
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def summary(self):
        if not self.samples:
            return None, None
        return sum(self.samples) / len(self.samples), max(self.samples)


async def _worker_loop(client, results, stop_event, per_request_log):
    """One concurrency slot: fire voice_pipeline requests back-to-back,
    sequentially within this task, until stop_event is set (i.e. sustained
    load for a wall-clock duration, not a fixed request count). Overall
    concurrency is simply the number of these tasks running at once."""
    while not stop_event.is_set():
        inputs, outputs = build_request("voice_pipeline")
        t0 = time.time()
        try:
            await client.infer(
                model_name="voice_pipeline", inputs=inputs, outputs=outputs,
                client_timeout=60.0,
            )
            elapsed = time.time() - t0
            results.append((elapsed, True, None))
        except Exception as e:
            elapsed = time.time() - t0
            results.append((elapsed, False, str(e)))
        per_request_log.append({"t": t0, "elapsed": elapsed, "ok": results[-1][1]})


async def run_soak_point(concurrency, duration_s, warmup_requests, config_label,
                          csv_path, outdir, grafana_auth=None):
    os.makedirs(outdir, exist_ok=True)
    per_request_log = []

    async with httpx.AsyncClient() as http_client:
        grpc_client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)

        # Warmup (untimed, excluded from stats) -- avoids cold-start skew
        # (first-call CUDA graph capture / JIT paths) contaminating the
        # timed window's percentiles.
        print(f"[{config_label} c={concurrency}] warmup: {warmup_requests} requests...")
        warm_results = []
        warm_sem = asyncio.Semaphore(min(concurrency, 4))
        await asyncio.gather(*[
            _fire_once(grpc_client, warm_sem, warm_results) for _ in range(warmup_requests)
        ])
        n_warm_ok = sum(1 for r in warm_results if r[1])
        print(f"[{config_label} c={concurrency}] warmup done: {n_warm_ok}/{warmup_requests} ok")

        # Grafana annotation + Triton metrics + host RAM, all snapshotted
        # at the true start of the TIMED window (post-warmup).
        push_grafana_annotation(
            text=f"SOAK START: {config_label} concurrency={concurrency} duration={duration_s}s",
            tags=["soak-test", config_label, f"concurrency-{concurrency}"],
            grafana_auth=grafana_auth,
        )
        before_metrics = parse_metrics((await http_client.get(METRICS_URL)).text)
        ram_start = psutil.virtual_memory().used / (1024 * 1024)

        gpu_sampler = GpuSampler(interval_s=1)
        cpu_sampler = CpuSampler(interval_s=1)
        gpu_sampler.start()
        cpu_sampler.start()

        results = []
        stop_event = asyncio.Event()
        workers = [
            asyncio.create_task(_worker_loop(grpc_client, results, stop_event, per_request_log))
            for _ in range(concurrency)
        ]
        print(f"[{config_label} c={concurrency}] soaking for {duration_s}s...")
        t_start = time.time()
        await asyncio.sleep(duration_s)
        stop_event.set()
        # let in-flight requests finish rather than cutting them off mid-call
        await asyncio.gather(*workers)
        wall_elapsed = time.time() - t_start

        gpu_sampler.stop()
        cpu_sampler.stop()
        ram_end = psutil.virtual_memory().used / (1024 * 1024)
        after_metrics = parse_metrics((await http_client.get(METRICS_URL)).text)
        push_grafana_annotation(
            text=f"SOAK END: {config_label} concurrency={concurrency}",
            tags=["soak-test", config_label, f"concurrency-{concurrency}"],
            grafana_auth=grafana_auth,
        )
        await grpc_client.close()

    latencies = sorted(r[0] for r in results)
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]

    def _vp(name, label_substr='model="voice_pipeline"'):
        b = get_metric(before_metrics, name, label_substr) or 0
        a = get_metric(after_metrics, name, label_substr) or 0
        return a - b

    d_exec = _vp("nv_inference_exec_count")
    vp_compute_us = _vp("nv_inference_compute_infer_duration_us")
    vp_queue_us = _vp("nv_inference_queue_duration_us")
    avg_compute_ms = (vp_compute_us / d_exec / 1000) if d_exec else 0
    avg_queue_ms = (vp_queue_us / d_exec / 1000) if d_exec else 0

    def _submodel_queue_ms(model_name):
        b_q = get_metric(before_metrics, "nv_inference_queue_duration_us", f'model="{model_name}"') or 0
        a_q = get_metric(after_metrics, "nv_inference_queue_duration_us", f'model="{model_name}"') or 0
        b_e = get_metric(before_metrics, "nv_inference_exec_count", f'model="{model_name}"') or 0
        a_e = get_metric(after_metrics, "nv_inference_exec_count", f'model="{model_name}"') or 0
        d_e = a_e - b_e
        return ((a_q - b_q) / d_e / 1000) if d_e else None

    whisper_q = _submodel_queue_ms("whisper_asr")
    chatterbox_q = _submodel_queue_ms("chatterbox_tts")
    # qwen_llm's own queue metric is deliberately NOT computed here -- see
    # module docstring and docs/voice-pipeline-queueing.md. It is not a CSV
    # column at all so it can never be silently substituted for a real one.

    gpu_util_avg, gpu_util_max, gpu_mem_avg, gpu_mem_max = gpu_sampler.summary()
    cpu_util_avg, cpu_util_max = cpu_sampler.summary()

    row = {
        "config_label": config_label, "concurrency": concurrency, "duration_s": duration_s,
        "warmup_requests": warmup_requests,
        "total_requests": len(results), "successes": len(successes), "failures": len(failures),
        "throughput_req_s": round(len(successes) / wall_elapsed, 3) if wall_elapsed else 0,
        "p50_s": round(pct(latencies, 0.50), 4) if latencies else "",
        "p90_s": round(pct(latencies, 0.90), 4) if latencies else "",
        "p95_s": round(pct(latencies, 0.95), 4) if latencies else "",
        "p99_s": round(pct(latencies, 0.99), 4) if latencies else "",
        "max_s": round(latencies[-1], 4) if latencies else "",
        "vp_server_avg_compute_ms": round(avg_compute_ms, 2),
        "vp_server_avg_queue_ms": round(avg_queue_ms, 2),
        "vp_execs": int(d_exec),
        "whisper_asr_avg_queue_ms": round(whisper_q, 2) if whisper_q is not None else "",
        "chatterbox_tts_avg_queue_ms": round(chatterbox_q, 2) if chatterbox_q is not None else "",
        "gpu_util_avg_pct": round(gpu_util_avg, 1) if gpu_util_avg is not None else "",
        "gpu_util_max_pct": round(gpu_util_max, 1) if gpu_util_max is not None else "",
        "gpu_mem_used_mb_avg": round(gpu_mem_avg, 1) if gpu_mem_avg is not None else "",
        "gpu_mem_used_mb_max": round(gpu_mem_max, 1) if gpu_mem_max is not None else "",
        "host_cpu_util_avg_pct": round(cpu_util_avg, 1) if cpu_util_avg is not None else "",
        "host_cpu_util_max_pct": round(cpu_util_max, 1) if cpu_util_max is not None else "",
        "host_ram_used_mb_start": round(ram_start, 1),
        "host_ram_used_mb_end": round(ram_end, 1),
        "host_ram_delta_mb": round(ram_end - ram_start, 1),
        "first_failure": failures[0][2][:200].replace("\n", " ") if failures else "",
    }

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=SOAK_CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)

    with open(os.path.join(outdir, f"c{concurrency}_per_request.jsonl"), "w") as f:
        for entry in per_request_log:
            f.write(json.dumps(entry) + "\n")
    with open(os.path.join(outdir, f"c{concurrency}_gpu_samples.json"), "w") as f:
        json.dump(gpu_sampler.samples, f)
    with open(os.path.join(outdir, f"c{concurrency}_cpu_samples.json"), "w") as f:
        json.dump(cpu_sampler.samples, f)

    print(f"\n=== {config_label} voice_pipeline concurrency={concurrency} "
          f"duration={duration_s}s ===")
    print(f"successes={len(successes)} failures={len(failures)}  "
          f"throughput={row['throughput_req_s']} req/s")
    if latencies:
        print(f"E2E client latency  p50={row['p50_s']}s  p90={row['p90_s']}s  "
              f"p95={row['p95_s']}s  p99={row['p99_s']}s  max={row['max_s']}s")
    print(f"voice_pipeline server avg_compute={row['vp_server_avg_compute_ms']}ms  "
          f"avg_queue(BLS-level only)={row['vp_server_avg_queue_ms']}ms  execs={row['vp_execs']}")
    print(f"supplement: whisper_asr avg_queue={row['whisper_asr_avg_queue_ms']}ms  "
          f"chatterbox_tts avg_queue={row['chatterbox_tts_avg_queue_ms']}ms  "
          f"(qwen_llm queue metric intentionally omitted -- unreliable, decoupled/streaming)")
    print(f"GPU util avg/max={row['gpu_util_avg_pct']}/{row['gpu_util_max_pct']}%  "
          f"GPU mem avg/max={row['gpu_mem_used_mb_avg']}/{row['gpu_mem_used_mb_max']}MB")
    print(f"host CPU util avg/max={row['host_cpu_util_avg_pct']}/{row['host_cpu_util_max_pct']}%  "
          f"host RAM {row['host_ram_used_mb_start']}->{row['host_ram_used_mb_end']}MB "
          f"(delta {row['host_ram_delta_mb']:+.1f}MB)")
    if failures:
        print(f"first failure: {row['first_failure']}")

    return row


async def _fire_once(client, sem, results):
    async with sem:
        inputs, outputs = build_request("voice_pipeline")
        t0 = time.time()
        try:
            await client.infer(model_name="voice_pipeline", inputs=inputs, outputs=outputs,
                                client_timeout=60.0)
            results.append((time.time() - t0, True, None))
        except Exception as e:
            results.append((time.time() - t0, False, str(e)))


async def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-label", required=True,
                         help='e.g. "accelerated_vllm" or "unaccelerated_pytorch" -- '
                              "tags every CSV row and every Grafana annotation")
    parser.add_argument("--concurrency-sweep", default="1,4,8,16",
                         help="comma-separated concurrency levels, run back-to-back")
    parser.add_argument("--duration-s", type=int, default=900,
                         help="sustained soak duration per concurrency point, seconds")
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True,
                         help="per-request logs + raw GPU/CPU sample dumps go here")
    parser.add_argument("--grafana-auth", default=None,
                         help="base64 user:pass for Grafana basic auth, if annotations need it")
    args = parser.parse_args()

    sweep = [int(x) for x in args.concurrency_sweep.split(",")]
    print(f"Soak test plan: config={args.config_label} sweep={sweep} "
          f"duration/point={args.duration_s}s -> total ~{len(sweep) * args.duration_s / 60:.1f} min")

    for c in sweep:
        await run_soak_point(
            concurrency=c, duration_s=args.duration_s, warmup_requests=args.warmup_requests,
            config_label=args.config_label, csv_path=args.csv, outdir=args.outdir,
            grafana_auth=args.grafana_auth,
        )


if __name__ == "__main__":
    asyncio.run(main())
