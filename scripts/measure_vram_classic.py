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
    from tensorrt_llm.llmapi.llm import _TrtLLM
    print(f"import took {time.time()-t0:.1f}s", flush=True)

    report("after import (no model loaded yet)")

    ENGINE_DIR = "/workspace/nemotron/llama_nemotron_engine"
    TOKENIZER_DIR = "/workspace/nemotron/Llama-3.1-Nemotron-Nano-4B-v1.1"

    t0 = time.time()
    llm = _TrtLLM(
        model=ENGINE_DIR,
        tokenizer=TOKENIZER_DIR,
        kv_cache_config={"free_gpu_memory_fraction": 0.2},
    )
    print(f"LLM() construction took {time.time()-t0:.1f}s", flush=True)

    report("after engine load (weights + KV cache pool reserved)")

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
