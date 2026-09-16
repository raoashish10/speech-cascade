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
# UNVERIFIED: not built or run against a real GPU (this development
# environment has neither a GPU nor Docker-in-Docker). Treat the first
# build on real hardware as the actual verification -- same spirit as
# deploy/REBUILD.md's own "Known gaps" section, not a claim this already
# works. Before relying on this, at minimum:
#   - confirm TRITON_BASE_IMAGE against
#     https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/release
#     for the tag that actually matches deploy/requirements-main.txt's
#     tensorrt_llm==1.2.1 pin (and ships tritonserver + the python backend,
#     not just the bare tensorrt_llm wheel)
#   - confirm `python3 -m venv` inside that image produces a *complete*
#     stdlib (README.md quirk #3's root cause on bare Ubuntu was a venv
#     that didn't) -- if not, the chatterbox venv below needs the same
#     conda-forge-pinned-to-the-bundled-libpython-version workaround
#     deploy/supervisor/speech-cascade-triton.sh documents, which is not
#     replicated here

ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:1.2.1
FROM ${TRITON_BASE_IMAGE}

# This image ships two Python 3.12 builds that don't agree with each
# other, confirmed the hard way across several failed attempts here:
# /usr/local/bin/python3 (what Triton's python backend and everything
# below actually runs as `python3`) has a broken _ssl -- its bundled
# _ssl.cpython-312-x86_64-linux-gnu.so wants OPENSSL_3.3.0 symbols that
# /lib/x86_64-linux-gnu/libssl.so.3 (what's actually installed) doesn't
# have, so `python3 -m pip` can't reach any HTTPS index at all. Bare
# `pip` (shebang #!/usr/bin/python -- a separate, Debian-provided 3.12
# build with working SSL) installs fine, but into dist-packages, which
# plain `python3`'s sys.path doesn't include by default (only
# site-packages) -- so those installs were invisible to python3 and to
# the assertion checks below, across multiple earlier attempts.
# Bridging the two via a standard `site`-module .pth file (both are
# literally cpython 3.12, so this is ABI-safe) lets bare `pip` keep doing
# the actual network installs while python3 -- and therefore Triton
# itself at runtime, not just this build -- can import everything it
# puts in dist-packages.
RUN echo "/usr/local/lib/python3.12/dist-packages" > /usr/local/lib/python3.12/site-packages/dist-packages-bridge.pth

# Only installs what requirements-main.txt pins that the base image doesn't
# already ship matched versions of -- pip install is idempotent for
# anything already satisfied. The numpy<2 assertion mirrors README.md
# quirk #10 (several packages in this list silently upgrade numpy if
# installed carelessly, which breaks TensorRT-LLM's compiled bindings).
# --extra-index-url is required: torch==2.9.1+cu128's "+cu128" local
# version tag only exists on PyTorch's own wheel index, not PyPI -- a
# plain `pip install` 404s on it (confirmed the hard way: the base image
# doesn't ship this exact pin pre-installed, so pip actually has to fetch
# it, unlike the packages above it in the file that resolve as "already
# satisfied").
#
# numpy==1.26.4 and librosa==1.0.0 are BOTH pinned in requirements-main.txt,
# but on this base image (Python 3.12) pip's resolver reports librosa 1.0.0
# itself requires numpy>=2.1.0 -- a hard ResolutionImpossible, confirmed the
# hard way, not a hypothetical. deploy/numpy_override.txt exists for exactly
# this (it's just never wired into deploy/ansible/playbook.yml's fresh-venv
# install, which apparently hasn't hit this particular resolver conflict).
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
# own existence implies.
#
# Bare `pip`, not `python3 -m pip`: `pip`'s shebang is #!/usr/bin/python,
# a separate, working-SSL cpython 3.12 build from /usr/local/bin/python3
# (the one Triton and the assertion below run as) -- see the top-of-file
# note on why python3 itself can't do network installs here. Bare pip
# installs into dist-packages, which the top-of-file .pth bridge now
# makes visible to python3's imports and to the assertion below too.
#
# The .pth bridge alone still isn't enough for numpy specifically: this
# base image's site-packages already ships its OWN numpy 2.5.3 (NVIDIA's
# own install, presumably, since it predates anything we've done), and a
# .pth-appended path is searched AFTER what's already in site-packages --
# so python3's `import numpy` kept resolving the stale 2.5.3 first no
# matter what dist-packages had, confirmed the hard way across attempt 8.
# Deleting that stale copy is the only way to make the correct
# dist-packages one (numpy_override, just above) actually win the import.
COPY deploy/requirements-main.txt /tmp/requirements-main.txt
COPY deploy/numpy_override.txt /tmp/numpy_override.txt
RUN grep -v -E '^(numpy|librosa)==' /tmp/requirements-main.txt > /tmp/requirements-main-nonumpy.txt \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /tmp/requirements-main-nonumpy.txt \
    && pip install --no-cache-dir --no-deps librosa==1.0.0 \
    && pip install --no-cache-dir awscli \
    && pip install --no-cache-dir --no-deps -r /tmp/numpy_override.txt \
    && rm -rf /usr/local/lib/python3.12/site-packages/numpy /usr/local/lib/python3.12/site-packages/numpy.libs /usr/local/lib/python3.12/site-packages/numpy-*.dist-info \
    && python3 -c "import numpy; assert numpy.__version__.startswith('1.26'), f'numpy drifted to {numpy.__version__}, see README.md quirk #10'"

# Chatterbox TTS runs in its own isolated venv, launched as a subprocess by
# triton_model_repo/chatterbox_tts/1/model.py at the hardcoded path
# /venv/chatterbox/bin/python3 -- its torch/torchaudio pins conflict with
# the main TensorRT-LLM stack above, same reason deploy/REBUILD.md keeps it
# isolated on bare metal. Do not `pip install chatterbox-tts` into the
# image's main environment instead of building this.
# Created FROM /usr/bin/python, NOT /usr/local/bin/python3.12 -- the
# latter is the one with the broken _ssl (see the top-of-file note), and
# a venv built on top of a broken-SSL interpreter inherits that same
# break (a venv references its base interpreter's lib-dynload C
# extensions, it doesn't get a fresh working _ssl for free). Both are
# literally cpython 3.12 so this is ABI-identical for everything the
# venv installs afterward -- it's specifically the _ssl.so/libssl
# mismatch being sidestepped, not a version difference.
RUN /usr/bin/python -m venv /venv/chatterbox
COPY deploy/requirements-chatterbox.txt /tmp/requirements-chatterbox.txt
# Same +cu128 wheel-index requirement as requirements-main.txt above --
# torch/torchaudio/torchcodec here are pinned to +cu128 builds too. Same
# numpy/librosa resolver split as requirements-main.txt above (this file
# pins numpy==1.26.4 and librosa==0.11.0 -- older than main's 1.0.0 pin,
# but excluded here too rather than betting a 130-package requirements
# file won't also hit "resolution-too-deep" the way the 233-package one
# did). Plain `bin/pip`, not `bin/python -m pip` -- this venv's own pip
# is already self-consistent with its own (working-SSL) python now that
# the venv itself is built on the working interpreter.
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
