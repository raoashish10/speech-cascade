# deploy/ — infrastructure as code

Everything needed to reproduce this project's deployment on a fresh
Vast.ai instance, captured from the actually-running instance rather than
written from memory. See `deploy/REBUILD.md` for the full step-by-step
runbook; this file just orients you to what's in this directory.

| File | What it is |
|---|---|
| `REBUILD.md` | The runbook: fresh instance -> working 4-model deployment, in order, including which steps are fully scripted/reproducible and which are documented-but-manual (see its "Known gaps" section). |
| `requirements-main.txt` | `pip freeze` of `/venv/main` (the main serving venv: `tensorrt_llm`, `torch`, `onnxruntime-gpu`, `kokoro-onnx`, ...). Install with `pip install -r deploy/requirements-main.txt`. **Fragile**: pins `numpy<2` transitively -- see the note at the top of `REBUILD.md` step 3 before adding anything new to this venv. |
| `requirements-gateway.txt` | `pip freeze` of `/venv/gateway` (the streaming gateway's isolated venv: `fastapi`, `uvicorn`, `silero-vad`, `tritonclient[grpc]`, ...), deliberately separate from `requirements-main.txt` so gateway deps can never touch the numpy pin above. |
| `supervisor/speech-cascade-gateway.{sh,conf}` | Copies of the actually-installed supervisor service for the streaming gateway (`/opt/supervisor-scripts/speech-cascade-gateway.sh`, `/etc/supervisor/conf.d/speech-cascade-gateway.conf`). |
| `supervisor/speech-cascade-triton.{sh,conf}` | A **new** supervisor service for the Triton server itself, capturing the exact command line and environment the live process is actually running with. Not previously formalized: Triton was found running as a manually-launched process with no supervisor entry, no autostart, and no autorestart-on-crash. **Captured here but not installed live** by this PR (see the file's own header) -- install it per `REBUILD.md` step 6 next time Triton needs a restart or a fresh instance needs to be brought up. |

## Regenerating the requirements files

```bash
/venv/main/bin/python -m pip freeze > deploy/requirements-main.txt
uv pip freeze --python /venv/gateway/bin/python > deploy/requirements-gateway.txt
```

Do this after any intentional dependency change, and check
`/venv/main/bin/python -c "import numpy; print(numpy.__version__)"` still
prints `1.26.4` (or whatever the current pin is) afterward.

## Regenerating the supervisor captures

```bash
cp /opt/supervisor-scripts/speech-cascade-gateway.sh deploy/supervisor/
cp /etc/supervisor/conf.d/speech-cascade-gateway.conf deploy/supervisor/
```

(`speech-cascade-triton.{sh,conf}` aren't installed live yet -- see above
-- so there's nothing on the instance to re-copy from until they are.)
