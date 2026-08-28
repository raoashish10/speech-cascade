# tests/ — automated test suite

Two tiers, with different dependencies and different Python environments.

## Unit tests (`tests/unit/`) — no live server, no GPU needed

Pure-logic tests for `streaming_gateway`'s two standalone-testable pieces:

- `test_vad.py` — `UtteranceVAD` (Silero VAD wrapper). Feeds a real short
  speech clip (`tests/fixtures/vad_sample_16k.wav`, synthesized offline via
  this project's own Kokoro TTS weights, not via the live Triton server)
  with silence padding and asserts exactly one start/end event pair lands
  at a plausible offset.
- `test_sentence.py` — `SentenceAccumulator`. Feeds text piecewise, checks
  sentence boundaries, the `MAX_ACCUM_CHARS` fallback-flush path, and exact
  reconstruction of the original text.

Silero VAD's own model runs on CPU via plain `onnxruntime` (not the GPU
build) — these run anywhere, including GitHub Actions CI, which is exactly
why they're the tier CI runs. Run with the **gateway venv**:

```bash
/venv/gateway/bin/python -m pytest tests/unit -v
```

(CI installs a fresh venv per `streaming_gateway/requirements.txt` — see
`.github/workflows/tests.yml`. Either environment works identically.)

## Integration tests (`tests/integration/`) — needs a live Triton server

Real gRPC calls (`tritonclient.grpc.aio`) against whatever's actually
loaded on `localhost:18001` right now:

- `test_whisper_asr.py`, `test_nemotron_llm.py`, `test_chatterbox_tts.py` — each
  of the three real models, in isolation.
- `test_voice_pipeline.py` — the BLS orchestrator, end to end (audio in,
  audio out, one call).
- `test_regression.py` — a small fixed set of "known good" prompts/texts
  per model (see the file's own docstring) — a lightweight sanity guard,
  not exhaustive model-quality testing. Confirms the serving stack still
  produces *something* non-empty and non-erroring, not that the output is
  "correct" in any deeper sense.

Every test in this directory is auto-marked `integration` (see
`tests/integration/conftest.py`'s `pytest_collection_modifyitems`) and
needs the tensorrt_llm/tritonclient stack, so run with the **main venv**:

```bash
/venv/main/bin/python -m pytest tests/integration -v
```

If the server isn't reachable, or a specific model isn't `READY` (e.g.
mid-reload by another agent), the relevant tests **skip** with a clear
reason rather than failing — see `grpc_client`/`require_ready` in
`tests/integration/conftest.py`.

**These do not run in CI** (`.github/workflows/tests.yml` only runs
`tests/unit/`) — there's no GPU on GitHub-hosted runners, and this whole
tier needs a real Triton server with 4 real GPU models loaded. Run these
by hand on the instance that actually serves the models.

**Coordination note**: these tests are read-only inference calls only —
they never load/unload/reload a model, so they're safe to run at any time
without racing another agent's model changes. They were written against
whatever behavior was actually observed on the live server at the time
this PR was authored; if the capacity/backpressure PR changes things like
`chatterbox_tts`'s instance count or adds admission-control rejection under
load, these single-request (non-concurrent) assertions should still pass,
but are worth rechecking once that PR lands.

## Selecting by marker instead of directory

Both directory-based selection (above) and the `integration` pytest marker
work:

```bash
python -m pytest -m "not integration"    # unit tests only, regardless of layout
python -m pytest -m integration          # integration tests only
```

## Running everything

There's no single venv with both the `tensorrt_llm`/`tritonclient` stack
and `silero-vad`'s CPU onnxruntime that's guaranteed conflict-free (that's
the whole reason the two venvs are separate — see main `README.md`'s numpy
pin note). Run each tier with its own venv as shown above; there's no
single invocation that runs both without risking that pin.
