import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_tensorrt_llm_checkpoint

MODEL_DIR = "/workspace/nemotron/Llama-3.1-Nemotron-Nano-4B-v1.1"
EXPORT_DIR = "/workspace/nemotron/llama_nemotron_fp8_ckpt"

CALIB_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank at sunset.",
    "In machine learning, quantization reduces the numerical precision of model weights.",
    "Can you help me book a flight to Tokyo next Tuesday morning?",
    "The stock market fell sharply today amid concerns over rising interest rates.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
    "What's the weather like in San Francisco this weekend?",
    "The French Revolution began in 1789 and reshaped European political history.",
    "Please summarize the following article in three concise bullet points.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "I'm feeling a bit under the weather, do you have any home remedy suggestions?",
    "The GDP of the country grew by 3.2 percent in the last fiscal quarter.",
    "Explain the difference between supervised and unsupervised learning.",
    "Once upon a time, in a small village surrounded by mountains, there lived a curious child.",
    "SELECT customer_id, SUM(total) FROM orders GROUP BY customer_id ORDER BY SUM(total) DESC;",
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "How do I convert a list of strings to integers in Python?",
    "The committee voted unanimously to approve the new budget proposal.",
    "Deep breathing exercises can help reduce stress and anxiety levels.",
    "The Great Barrier Reef is the world's largest coral reef system.",
    "What are the main differences between TCP and UDP protocols?",
    "She packed her bags and left for the airport before dawn.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Quantum computers use qubits instead of classical bits to perform computations.",
    "Can you translate 'good morning' into French, Spanish, and German?",
    "The museum's new exhibit features artifacts from ancient Mesopotamia.",
    "Rising sea levels pose a significant threat to coastal cities worldwide.",
    "Write a short poem about autumn leaves falling in the wind.",
    "The company's quarterly earnings exceeded analyst expectations.",
    "How does the immune system distinguish between healthy cells and pathogens?",
    "The marathon route winds through five historic neighborhoods downtown.",
    "In chess, the Sicilian Defense is one of the most popular responses to e4.",
]


def main():
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    print(f"Loaded HF model in {time.time()-t0:.1f}s", flush=True)

    def forward_loop(m):
        for text in CALIB_TEXTS:
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                m(**inputs)

    t0 = time.time()
    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)
    print(f"Quantized (calibrated FP8) in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    export_tensorrt_llm_checkpoint(
        model,
        decoder_type="llama",
        dtype=torch.float16,
        export_dir=EXPORT_DIR,
        inference_tensor_parallel=1,
    )
    print(f"Exported TensorRT-LLM checkpoint in {time.time()-t0:.1f}s to {EXPORT_DIR}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
