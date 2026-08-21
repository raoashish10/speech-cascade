import subprocess
import time


def gpu_mem():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"]
    ).decode().strip()
    used, free, total = (int(x) for x in out.split(","))
    return used, free, total


def report(label):
    used, free, total = gpu_mem()
    print(f"[{label}] used={used} MiB free={free} MiB total={total} MiB", flush=True)
    return used, free, total


def main():
    report("baseline (before import)")

    t0 = time.time()
    from tensorrt_llm import LLM
    print(f"import took {time.time()-t0:.1f}s", flush=True)

    report("after import (no model loaded yet)")

    MODEL_PATH = "/workspace/nemotron/Nemotron-3-Nano-4B-FP8"

    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        backend="_autodeploy",
        trust_remote_code=True,
        compile_backend="torch-cudagraph",
        max_batch_size=8,
        cuda_graph_batch_sizes=[1, 2, 4, 8],
        kv_cache_config={"enable_block_reuse": False, "free_gpu_memory_fraction": 0.2},
    )
    print(f"LLM() construction took {time.time()-t0:.1f}s", flush=True)

    report("after model load (weights + KV cache pool reserved)")

    from tensorrt_llm import SamplingParams

    t0 = time.time()
    outputs = llm.generate(
        ["Hello, my name is"],
        SamplingParams(max_tokens=32, temperature=0),
    )
    print(f"generate() took {time.time()-t0:.1f}s", flush=True)
    for o in outputs:
        print("OUTPUT:", o.outputs[0].text, flush=True)

    report("after first generation (peak steady-state)")

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
