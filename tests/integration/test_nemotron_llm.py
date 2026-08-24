"""Integration tests for nemotron_llm in isolation, against the live Triton
server. nemotron_llm is decoupled/streaming (max_batch_size: 0) -- every
call here goes through stream_infer(), never plain infer() (which Triton
rejects outright for a decoupled model). See tests/integration/conftest.py.
"""

import numpy as np
import tritonclient.grpc.aio as grpcclient


def _prompt_request(prompt):
    arr = np.array([prompt.encode("utf-8")], dtype=object)
    inp = grpcclient.InferInput("PROMPT", arr.shape, "BYTES")
    inp.set_data_from_numpy(arr)
    return [inp], [grpcclient.InferRequestedOutput("GENERATED_TEXT")]


async def test_nemotron_llm_responds_to_a_real_prompt(grpc_client, require_ready, stream_infer_collect):
    await require_ready("nemotron_llm")
    inputs, outputs = _prompt_request("Hello, my name is")
    text = await stream_infer_collect(grpc_client, "nemotron_llm", inputs, outputs)

    assert isinstance(text, str)
    assert text.strip() != "", "generated text should be non-empty"


async def test_nemotron_llm_plain_infer_is_rejected(grpc_client, require_ready):
    """Documents the decoupled-model contract this whole suite (and
    scripts/load_test.py) works around: a plain unary infer() call must
    fail, not silently hang or succeed."""
    await require_ready("nemotron_llm")
    inputs, outputs = _prompt_request("Hello")
    try:
        await grpc_client.infer(
            model_name="nemotron_llm", inputs=inputs, outputs=outputs, client_timeout=15.0
        )
    except Exception:
        return  # expected
    raise AssertionError("plain infer() against a decoupled model should have raised")
