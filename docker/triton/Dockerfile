# syntax=docker/dockerfile:1
#
# Triton + TensorRT-LLM serving image: qwen_llm, whisper_asr, chatterbox_tts,
# voice_pipeline (see README.md's "The four Triton models").
#
# Built FROM an NGC image on purpose -- README.md's "Environment quirks
# fixed to make this run bare-metal" section says outright that none of
# those ten hand-fixed quirks (MPI, CUDA runtime libs, PYTHONHOME/PATH
# fights, missing libssl.so.1.1, missing DCGM, cuBLAS version clash, numpy
# pin drift, ...) are needed inside NGC's containers, because they ship a
# consistent, matched toolchain. This Dockerfile deliberately does NOT
# replicate deploy/supervisor/speech-cascade-triton.sh's
# PYTHONHOME/PYTHONPATH/LD_LIBRARY_PATH gymnastics -- those were bare-metal
# fixes for a mismatched toolchain that shouldn't exist here.
#
# BASE IMAGE, confirmed the hard way across two real builds:
# `nvcr.io/nvidia/tensorrt-llm/release` (the ORIGINAL base here) is a bare
# TensorRT-LLM dev/runtime image -- PyTorch, CUDA, the tensorrt_llm Python
# package, but NOT NVIDIA's Triton Inference Server binary at all. Deployed
# and run for real on a Runpod GPU pod: entrypoint.sh's `tritonserver`
# invocation failed with "command not found". (Its TRITON_* env vars, e.g.
# TRITON_PTXAS_PATH, are for OpenAI's Triton -- the PyTorch kernel-JIT
# compiler pip package -- not NVIDIA Triton Inference Server; same name,
# unrelated projects, easy to mistake for evidence tritonserver was
# present.) The correct combined image is a separate NGC repo,
# `nvcr.io/nvidia/tritonserver`, tagged `<date>-trtllm-python-py3`.
# `26.03` through `26.07` all report identical TensorRT (10.14.1.48+
# cuda13.0) / cuBLAS (13.2.0.9) version labels to tensorrt-llm/release:1.2.1
# (checked via `skopeo inspect --override-os linux --override-arch amd64`
# against nvcr.io directly); 26.03 picked as the earliest exact match.
#
# This image's own tritonserver binary lives at
# /opt/tritonserver/bin/tritonserver but ships OFF the default PATH (unlike
# most official Triton images) -- confirmed by `find` on a real pod after
# the bare `tritonserver` invocation above failed here too, initially.
#
# Also confirmed on that same pod: this image's Python setup is much
# simpler than tensorrt-llm/release's -- ONE python3.12 (/usr/bin/python3),
# `pip`'s shebang matches it exactly (#!/usr/bin/python3.12), SSL works
# natively, and dist-packages is already on sys.path. None of
# tensorrt-llm/release's dual-interpreter/broken-_ssl/.pth-bridge/
# stale-site-packages-numpy workarounds (present in earlier revisions of
# this file) apply here -- removed accordingly. What's kept from that fix
# chain: the numpy/librosa/chatterbox-tts pip-resolver conflicts, which are
# about these packages' own declared metadata, not the interpreter.
#
# UNVERIFIED beyond the tritonserver-binary-exists check: whether it
# actually starts and loads all four models against real weights/engines
# (no weights were mounted for that check) is still to be confirmed against
# a real GPU run of this rebuilt image.

ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3
FROM ${TRITON_BASE_IMAGE}

# /opt/tritonserver/bin is ALREADY on this base image's PATH (verified via
# `skopeo inspect --config`) -- no PATH override needed here.
#
# Everything below was found by running this stack interactively on a GPU
# pod against this exact base image, not by guessing between rebuilds:
#
# 1. chatterbox_tts/1/model.py defaults CHATTERBOX_BACKEND to "vllm",
#    needing a /venv/vllm this Dockerfile never builds -- confirmed:
#    "FileNotFoundError: ... /venv/vllm/bin/python3". Building that venv
#    isn't enough on its own either: the vllm path also needs
#    vllm_t3_model_dir (an exported copy of chatterbox's T3 GPT2 backbone
#    for vLLM's out-of-tree loader), which doesn't exist anywhere in this
#    repo and has no documented recipe -- see
#    scripts/export_chatterbox_t3_for_vllm.py for a from-scratch
#    reconstruction, itself UNVERIFIED. Falling back to the pytorch
#    backend, which uses the /venv/chatterbox this file already builds.
#
# 2. OPAL_PREFIX: tensorrt_llm imports mpi4py, and MPI_Init_thread aborted
#    with "opal_init:startup:internal-failure". Root cause: this image's
#    OpenMPI was built with prefix /build-result/hpcx-v2.25.1-.../ompi but
#    is actually installed at /usr/local/mpi, so OPAL cannot find its own
#    runtime data files (the giveaway: it tried to open help text under
#    that nonexistent /build-result path). Setting OPAL_PREFIX to the real
#    install location fixes it -- verified directly:
#    `OPAL_PREFIX=/usr/local/mpi python3 -c "from mpi4py import MPI; ..."`
#    goes from fatal abort to "MPI OK". This is README.md quirk #1 (MPI)
#    reappearing inside an NGC container, contrary to this file's original
#    assumption that none of those quirks apply here.
#
# Deliberately NOT set: PYTHONPATH and a hand-rolled LD_LIBRARY_PATH for
# the pip nvidia-*-cu12 lib dirs. Both were earlier guesses at the
# librosa/libcublas load failures; running the identical pip sequence
# interactively on this base image showed torch 2.9.1+cu128 and librosa
# both import fine with the base image's own env, so neither override is
# needed and both risk shadowing the base image's settings.
ENV CHATTERBOX_BACKEND="pytorch"
ENV OPAL_PREFIX="/usr/local/mpi"

# Only installs what requirements-main.txt pins that the base image doesn't
# already ship matched versions of -- pip install is idempotent for
# anything already satisfied. The numpy<2 assertion mirrors README.md
# quirk #10 (several packages in this list silently upgrade numpy if
# installed carelessly, which breaks TensorRT-LLM's compiled bindings).
# --extra-index-url is required: torch==2.9.1+cu128's "+cu128" local
# version tag only exists on PyTorch's own wheel index, not PyPI.
#
# numpy==1.26.4 and librosa==1.0.0 are BOTH pinned in requirements-main.txt,
# but pip's resolver here reports librosa 1.0.0 itself requires
# numpy>=2.1.0 -- a hard ResolutionImpossible, confirmed the hard way, not
# a hypothetical. deploy/numpy_override.txt exists for exactly this (it's
# just never wired into deploy/ansible/playbook.yml's fresh-venv install,
# which apparently hasn't hit this particular resolver conflict).
#
# Just dropping the numpy== pin isn't enough on its own: with numpy left
# fully unconstrained across ~200 other exact-pinned packages, pip's
# resolver has to backtrack across every one of their individual numpy
# ranges while librosa alone pulls toward >=2.1.0 -- confirmed the hard
# way too, pip gives up with "resolution-too-deep" (dependency graph too
# complex). Dropping librosa as well removes the actual outlier forcing
# that backtrack, so numpy settles near 1.26.x on its own from the other
# ~198 packages' (compatible) constraints. Both are then force-installed
# with --no-deps: librosa==1.0.0 ignoring its own numpy>=2.1.0 claim, then
# numpy==1.26.4 pinned back down -- same "ignore the now-stale metadata
# constraint, it still works at runtime" trade-off numpy_override.txt's
# own existence implies. numpy_override runs LAST (after awscli too), not
# right after librosa -- confirmed on the previous base image that a
# later, unrelated pip install can still nudge numpy again in between.
# TENSORRT, confirmed the hard way on a real GPU pod: PyPI's
# `tensorrt==10.14.1.48.post1` and `tensorrt_cu13==...post1` (both pinned in
# requirements-main.txt) are METADATA-ONLY wheels -- dist-info and nothing
# else, zero module files (`pip show -f tensorrt` lists only dist-info
# entries). Installing them makes pip uninstall the base image's real,
# working `tensorrt 10.14.1.48` and leave nothing in its place, so
# `import tensorrt` -- and therefore all of tensorrt_llm, and therefore
# qwen_llm -- dies with ModuleNotFoundError. The real wheels live on
# NVIDIA's own index, not PyPI, so the module is restored explicitly from
# pypi.nvidia.com after the main install. The trailing `import tensorrt`
# check exists so this can never regress silently again.
COPY deploy/requirements-main.txt /tmp/requirements-main.txt
COPY deploy/numpy_override.txt /tmp/numpy_override.txt
RUN grep -v -E '^(numpy|librosa)==' /tmp/requirements-main.txt > /tmp/requirements-main-nonumpy.txt \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements-main-nonumpy.txt \
    && pip install --no-cache-dir --no-deps librosa==1.0.0 \
    && pip install --no-cache-dir awscli \
    && pip install --no-cache-dir --no-deps -r /tmp/numpy_override.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps --extra-index-url https://pypi.nvidia.com tensorrt-cu13==10.14.1.48 \
    && python3 -c "import numpy; assert numpy.__version__.startswith('1.26'), f'numpy drifted to {numpy.__version__}, see README.md quirk #10'" \
    && python3 -c "import tensorrt; print('tensorrt', tensorrt.__version__)"

# Chatterbox TTS runs in its own isolated venv, launched as a subprocess by
# triton_model_repo/chatterbox_tts/1/model.py at the hardcoded path
# /venv/chatterbox/bin/python3 -- its torch/torchaudio pins conflict with
# the main TensorRT-LLM stack above, same reason deploy/REBUILD.md keeps it
# isolated on bare metal. Do not `pip install chatterbox-tts` into the
# image's main environment instead of building this.
RUN python3 -m venv /venv/chatterbox
COPY deploy/requirements-chatterbox.txt /tmp/requirements-chatterbox.txt
# Same +cu128 wheel-index requirement as requirements-main.txt above --
# torch/torchaudio/torchcodec here are pinned to +cu128 builds too. Same
# numpy/librosa resolver split as requirements-main.txt above (this file
# pins numpy==1.26.4 and librosa==0.11.0 -- older than main's 1.0.0 pin,
# but excluded here too rather than betting a 130-package requirements
# file won't also hit "resolution-too-deep" the way the 233-package one
# did).
#
# chatterbox-tts==0.1.7 also excluded from the main pass, confirmed the
# hard way: its own metadata hard-pins torch==2.6.0, directly conflicting
# with this file's torch==2.11.0+cu128 pin (a newer CUDA build presumably
# chosen deliberately). Installed after with --no-deps, same trade-off as
# numpy/librosa -- ignore chatterbox-tts's stale torch constraint, it
# still works at runtime against the newer pin.
RUN /venv/chatterbox/bin/pip install --no-cache-dir --upgrade pip \
    && grep -v -E '^(numpy|librosa|chatterbox-tts)==' /tmp/requirements-chatterbox.txt > /tmp/requirements-chatterbox-nonumpy.txt \
    && /venv/chatterbox/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements-chatterbox-nonumpy.txt \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7 \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-deps librosa==0.11.0 \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-deps -r /tmp/numpy_override.txt

# config.pbtxt files across triton_model_repo/ use absolute paths under
# /workspace/speech-cascade-inference/... (e.g. qwen_llm's engine_dir,
# whisper_asr's assets_dir, chatterbox_tts's reference-voice path) --
# matching that exact layout here means nothing in triton_model_repo/
# needs editing for Docker. Only models/ (weights + compiled engines) is
# expected to arrive via the volume docker-compose.yml mounts at that same
# path -- see docker/triton/entrypoint.sh for the optional S3 restore.
RUN mkdir -p /workspace/speech-cascade-inference
COPY triton_model_repo /workspace/speech-cascade-inference/triton_model_repo

COPY docker/triton/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace/speech-cascade-inference

# Same three ports as the bare-metal deployment (README.md's "Ports"
# section) -- kept off the host's published ports in docker-compose.yml so
# they stay reachable only from other containers on the compose network,
# matching the "internal-only" property README.md describes.
EXPOSE 18000 18001 18002

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
