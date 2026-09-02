"""Shared fixtures for integration tests against the live Triton server.

Every test module under tests/integration/ needs a real Triton gRPC
endpoint with all four models loaded. These do NOT run in GitHub Actions CI
-- there's no GPU on GitHub-hosted runners, and this whole suite is meant
to be run by hand on the Vast.ai instance itself (see tests/README.md).

Coordination note (read before touching this file): other agents may be
concurrently reloading/tuning these models (capacity/backpressure changes
in particular). These fixtures only ever call the read-only inference API
against whatever is *currently* loaded -- never repository-control
(load/unload/reload) endpoints -- specifically so this suite is safe to run
at any time without racing another agent's model reload.

Helpers are exposed as fixtures (rather than plain importable functions) so
test modules under this directory need no cross-file imports -- avoids
depending on tests/ being set up as a proper import package.
"""

import asyncio

import numpy as np
import pytest
import pytest_asyncio
import tritonclient.grpc.aio as grpcclient

TRITON_GRPC_URL = "localhost:18001"
CLIENT_TIMEOUT_S = 60.0


def pytest_collection_modifyitems(config, items):
    # This hook is global once this conftest is loaded, so scope it: only
    # auto-mark items actually collected from tests/integration/ (unit
    # tests are collected via a separate conftest-free path and must stay
    # unmarked).
    for item in items:
        if "tests/integration/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.integration)


@pytest_asyncio.fixture(scope="session")
async def grpc_client():
    client = grpcclient.InferenceServerClient(url=TRITON_GRPC_URL)
    try:
        live = await client.is_server_live()
    except Exception as e:
        await client.close()
        pytest.skip(
            f"Triton server not reachable at {TRITON_GRPC_URL} ({e}) -- "
            "integration tests require a live server, run these manually "
            "on the instance (see tests/README.md)."
        )
        return
    if not live:
        await client.close()
        pytest.skip(f"Triton server at {TRITON_GRPC_URL} reports not-live")
        return
    yield client
    await client.close()


@pytest_asyncio.fixture
async def require_ready(grpc_client):
    """require_ready("whisper_asr") -- skip the test (not fail it) if that
    model isn't currently READY, e.g. mid-reload by another agent."""

    async def _check(model_name):
        ready = await grpc_client.is_model_ready(model_name)
        if not ready:
            pytest.skip(f"{model_name} is not READY on the live server right now")

    return _check


@pytest_asyncio.fixture(scope="session")
async def synth_speech(grpc_client):
    """A short real speech clip ("This is a known good test sentence.")
    synthesized via the live chatterbox_tts model -- used as ASR/pipeline input
    so ASR-dependent tests exercise real speech content instead of silence
    or noise, without committing a binary audio fixture that would go stale
    against whatever voice/model version is actually deployed.

    Plain inference call (no reload) -- safe alongside other agents' work.
    """
    ready = await grpc_client.is_model_ready("chatterbox_tts")
    if not ready:
        pytest.skip("chatterbox_tts is not READY on the live server right now")

    text = "This is a known good test sentence."
    arr = np.array([[text]], dtype=object)
    inp = grpcclient.InferInput("TEXT", arr.shape, "BYTES")
    inp.set_data_from_numpy(arr)
    result = await grpc_client.infer(
        model_name="chatterbox_tts",
        inputs=[inp],
        outputs=[
            grpcclient.InferRequestedOutput("AUDIO_SAMPLES"),
            grpcclient.InferRequestedOutput("SAMPLE_RATE"),
        ],
        client_timeout=CLIENT_TIMEOUT_S,
    )
    audio = result.as_numpy("AUDIO_SAMPLES").astype(np.float32).flatten()
    sample_rate = int(result.as_numpy("SAMPLE_RATE").flatten()[0])
    assert audio.size > 0
    return text, audio, sample_rate


@pytest.fixture
def stream_infer_collect():
    """qwen_llm is decoupled/streaming (max_batch_size: 0) -- a plain
    infer() is rejected outright ("ModelInfer RPC doesn't support models
    with decoupled transaction policy"). Returns an async helper that fires
    one stream_infer() request, drains every chunk, and concatenates
    GENERATED_TEXT diffs into the full completion. Mirrors
    scripts/load_test.py's _stream_one_request()."""

    async def _collect(client, model_name, inputs, outputs, timeout_s=CLIENT_TIMEOUT_S):
        async def _gen():
            yield {"model_name": model_name, "inputs": inputs, "outputs": outputs}

        chunks = []
        got_any = False

        async def _drain():
            nonlocal got_any
            async for result, error in client.stream_infer(_gen()):
                if error is not None:
                    raise RuntimeError(str(error))
                got_any = True
                piece = result.as_numpy("GENERATED_TEXT")
                if piece is not None:
                    value = piece.flatten()[0]
                    chunks.append(value.decode("utf-8") if isinstance(value, bytes) else value)

        await asyncio.wait_for(_drain(), timeout=timeout_s)
        if not got_any:
            raise RuntimeError("decoupled stream produced no responses")
        return "".join(chunks)

    return _collect
