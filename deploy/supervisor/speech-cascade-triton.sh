#!/bin/bash
# Runs the project's Triton Inference Server (the 4-model voice pipeline:
# qwen_llm, whisper_asr, chatterbox_tts, voice_pipeline) as a managed
# supervisor service.
#
# Installed and running live as a supervisor service (this file's history:
# it started out captured-but-not-installed, documenting a manually-launched
# process with no supervisor entry, autorestart, or place in the boot
# sequence -- since fixed by actually installing it).
#
# Models are loaded ONE AT A TIME via the explicit-mode repository control
# API below, not with --load-model flags at startup. Measured directly on
# this box: qwen_llm's own JIT engine build transiently peaks host RAM at
# ~14.5GB (settling back to ~8-9GB once loaded) against this container's
# ~15GB cgroup memory.max. Loading all four models concurrently at startup
# (the previous --load-model=* approach) overlapped that peak with
# whisper_asr/chatterbox_tts's own concurrent startup RAM use, exceeding the
# container's memory ceiling and getting qwen_llm's python-backend stub
# OOM-killed -- surfaced as "Stub process 'qwen_llm_0_0' is not healthy"
# with no Python traceback (consistent with SIGKILL, not a catchable
# exception). Triton's /v2/repository/models/{name}/load call blocks until
# that model's load finishes (success or failure), so looping over it
# sequentially guarantees no two models' loading-time peaks overlap.
# qwen_llm loads first, while baseline RAM is lowest.
#
# To install on a live instance:
#   cp deploy/supervisor/speech-cascade-triton.sh /opt/supervisor-scripts/
#   cp deploy/supervisor/speech-cascade-triton.conf /etc/supervisor/conf.d/
#   chmod +x /opt/supervisor-scripts/speech-cascade-triton.sh
#   supervisorctl reread && supervisorctl update
#
# CAUTION: do not install/restart this on an instance where Triton is
# already running with models another process or agent depends on --
# starting this service binds ports 18000-18002 and loads all four models
# onto the GPU again, colliding with (or duplicating VRAM alongside) a
# manually-launched server. Coordinate before restarting a live one; see
# README.md "Managing the service" for the ~5-6 minute restart cost
# (dominated by the LLM's TensorRT-LLM import + engine load).
#
# Binds to 127.0.0.1 only (matches the currently-running process) -- no
# portal.yaml entry needed here. External exposure of anything Triton-
# adjacent (if wanted) is a separate concern, being handled by the
# external-access/Caddy work in a parallel PR.

utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

# --- Environment quirks from README.md's "Environment quirks fixed to
# make this run bare-metal" (#1-#7) -- required for the Python-backend
# stub's CPython ABI to resolve consistently across the system Python, the
# venv's conda-forge Python, and Triton's own bundled libpython, and for
# onnxruntime's CUDA EP to find cuBLAS/cuDNN 12.x (shipped inside the
# venv's own nvidia-* pip packages) instead of the system's cuBLAS 13.x. ---
export PYTHONHOME=/venv/main
# Instance-specific fix, discovered during this deployment: a plain
# `python3.12 -m venv /venv/main` never contains a full stdlib copy (only
# site-packages) -- forcing PYTHONHOME=/venv/main (quirk #3) then leaves
# Triton's python-backend stub (which always dynamically loads its OWN
# bundled libpython3.12.so.1.0 via an $ORIGIN RUNPATH, regardless of
# PYTHONHOME/PYTHONPATH) unable to find `encodings` at all ("failed to get
# the Python codec of the filesystem encoding"). Worse: that bundled
# libpython (built on Red Hat/GCC-11.2.1 per its embedded build string) does
# NOT statically link several modules (_struct, _posixsubprocess, _ctypes,
# _datetime) that every distro's own CPython build here treats as
# compiled-in builtins -- so no local Python distribution has them as
# loadable .so files except conda-forge's (which builds almost everything
# as separate .so modules by policy). The exact conda-forge *version*
# matters, not just that it's conda-forge: an initial fix using
# conda-forge's latest (3.12.14) mismatched the bundled libpython's own
# patch version (3.12.3) at the C-API level for anything with its own
# capsule/struct-layout ABI -- worked fine for _struct/_posixsubprocess
# (plain functions), but corrupted _ctypes (undefined symbol:
# _PyErr_SetLocaleString) and, far more subtly, _datetime: datetime.date
# objects loaded fine and imported fine, but calling .strftime() on one
# produced garbage ("'datetime.date' object has no attribute 'tb_frame'"),
# which stdlib calendar.py hits on first use building its locale-name
# cache, which transformers.generation's lazy-loader then reports several
# frames later as the wildly misleading "cannot import name
# 'GenerationMixin' from 'transformers.generation'". Cost real time to
# root-cause specifically because it looked exactly like a threading race
# (intermittent-seeming, deep in an unrelated-looking import chain) rather
# than a version-pinned ABI mismatch -- it is 100% deterministic once you
# know to reproduce it directly (`datetime.date(2001,1,1).strftime('%a')`)
# rather than through transformers' own confusing error surface. Fixed by
# pinning conda-forge's python to the *exact* same version, not just
# distribution, as the bundled libpython: `mamba create -p
# /workspace/py312_dynload_exact python=3.12.3 --no-deps -c conda-forge`.
# Workspace-persistent but not provisioned by anything; recreate the same
# way after a recycle/destroy. Ubuntu's own lib-dynload is kept first in
# priority purely for the modules it does ship as real files (e.g.
# _ssl) -- redundant with the exact-match conda env for anything both
# provide, but harmless either way since they're now the same ABI.
# nvidia-cutlass-dsl (a tensorrt_llm dependency needed for its Blackwell
# fused-MoE CUTLASS DSL custom ops) ships as a .pth file
# (nvidia_cutlass_dsl.pth) pointing at a nested python_packages/ dir --
# .pth files are only processed by `site.addsitedir()` for a directory
# discovered through NORMAL site-packages resolution (derived from
# sys.prefix), and forcing PYTHONHOME to a bare venv here means that normal
# resolution path never actually runs, so the .pth file is silently never
# read. Symptom: `import cutlass` -> ModuleNotFoundError, which
# tensorrt_llm's own try/except around it swallows into
# IS_CUTLASS_DSL_AVAILABLE=False, which THEN surfaces many frames later and
# confusingly as "cannot import name
# 'Sm100BlockScaledContiguousGatherGroupedGemmSwigluFusionRunner'" (a class
# defined inside an `if IS_CUTLASS_DSL_AVAILABLE:` block, so it simply never
# exists when the flag is False). Fixed by adding the .pth file's target
# directory to PYTHONPATH directly, bypassing .pth processing entirely.
export PYTHONPATH="/usr/lib/python3.12:/usr/lib/python3.12/lib-dynload:/workspace/py312_dynload_exact/lib/python3.12/lib-dynload:/venv/main/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages:/venv/main/lib/python3.12/site-packages"
export PATH="/venv/main/bin:${PATH}"
export LD_LIBRARY_PATH="/venv/main/lib:/workspace/speech-cascade-inference/triton_server/compat_libs/extracted/usr/lib/x86_64-linux-gnu:/venv/main/lib/python3.12/site-packages/nvidia/cublas/lib:/venv/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/venv/gateway/lib/python3.12/site-packages/nvidia/cu13/lib:/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/lib64:${LD_LIBRARY_PATH:-}"

TRITON_BIN=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/bin/tritonserver
MODEL_REPO=/workspace/speech-cascade-inference/triton_model_repo
BACKEND_DIR=/workspace/speech-cascade-inference/triton_server/extracted/tritonserver/backends

cd /workspace

# Not run through pty/unbuffer here (unlike other supervisor scripts) --
# it needs to background cleanly under a plain $!/wait pair below, and
# tritonserver's own log output already flushes promptly without it.
"${TRITON_BIN}" \
  --model-repository="${MODEL_REPO}" \
  --backend-directory="${BACKEND_DIR}" \
  --http-port=18000 \
  --grpc-port=18001 \
  --metrics-port=18002 \
  --http-address=127.0.0.1 \
  --grpc-address=127.0.0.1 \
  --model-control-mode=explicit \
  --exit-on-error=false \
  --log-verbose=0 \
  --metrics-config summary_latencies=true 2>&1 &
  # summary_latencies=true: without it, Triton's nv_inference_*_duration_us
  # metrics are cumulative counters -- Prometheus/Grafana can only derive
  # averages from those, never real percentiles. This turns each into a
  # proper Summary (nv_inference_request_summary_us etc.) with quantile
  # labels (0.5/0.9/0.95/0.99/0.999), directly graphable/queryable for p50/
  # p90/p99 -- see chatterbox_profiling/ for the load-test numbers this
  # was added to surface.
TRITON_PID=$!

until curl -sf -o /dev/null http://127.0.0.1:18000/v2/health/live; do
  sleep 1
done

# qwen_llm first (baseline RAM lowest here), then the lighter models --
# see the header comment for why this must stay sequential, not parallel.
for model in qwen_llm whisper_asr chatterbox_tts voice_pipeline; do
  echo "loading ${model}..."
  if curl -sf -X POST "http://127.0.0.1:18000/v2/repository/models/${model}/load" -d '{}'; then
    echo "${model} load request succeeded"
  else
    echo "${model} load request FAILED -- see the server log above for the reason"
  fi
done

wait "${TRITON_PID}"
