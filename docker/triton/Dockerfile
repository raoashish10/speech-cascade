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
# IMAGE SIZE: this image is large but no longer gratuitously so. It was
# ~40GB unpacked (~27GB compressed), of which ~19GB was duplicate CUDA
# userspace: both pip installs below were replaying `pip freeze` dumps from
# a CUDA 12.8 bare-metal box onto a CUDA 13.1 base image, so each one
# installed its own complete cu12 wheel set next to the CUDA 13 libraries
# the base already provides. Installing those freezes with --no-deps minus
# an explicit exclusion list (deploy/image-exclude-*.txt) brings it to
# ~24GB unpacked. The two RUN blocks below carry the measurements and the
# reasoning; the assertions at the end of each are regression guards, so a
# future edit that reintroduces a cu12 stack fails the build instead of
# quietly adding 9GB.
#
# What remains is mostly irreducible without changing base images:
# ~22GB of that ~24GB is the base image itself (tensorrt_llm 3.5GB,
# flash_attn's single 1.7GB .so, torch, CUDA 13, Triton).
#
# UNVERIFIED for the slimmed install: actual GPU execution. Every check
# that does not need libcuda.so.1 was run against this base image on a
# Runpod CPU pod (see the two RUN blocks), but a CPU host cannot import
# tensorrt_llm or run a CUDA kernel, so the four-model GPU run that
# validated the earlier, fatter recipe has not been repeated against this
# one. Treat the next GPU deployment as that verification.

ARG TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.03-trtllm-python-py3
FROM ${TRITON_BASE_IMAGE}

# /opt/tritonserver/bin is ALREADY on this base image's PATH (verified via
# `skopeo inspect --config`) -- no PATH override needed here.
#
# Everything below was found by running this stack interactively on a GPU
# pod against this exact base image, not by guessing between rebuilds:
#
# 1. chatterbox_tts/1/model.py defaults CHATTERBOX_BACKEND to "vllm", which
#    needs two things this image originally lacked: a /venv/vllm, and
#    vllm_t3_model_dir (chatterbox's T3 GPT2 backbone re-exported for vLLM's
#    out-of-tree loader). Missing either produced
#    "FileNotFoundError: ... /venv/vllm/bin/python3" on a real pod, and the
#    export step had no recipe anywhere in this repo. Both now exist -- the
#    venv below, and scripts/export_chatterbox_t3_for_vllm.py wired into
#    scripts/build_models_from_scratch.sh -- and the whole path is verified
#    on a GPU (see the /venv/vllm block for the numbers). /venv/chatterbox
#    remains as the CHATTERBOX_BACKEND=pytorch fallback.
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
# interactively on this base image showed torch and librosa both import
# fine with the base image's own env, so neither override is needed and
# both risk shadowing the base image's settings. There are no
# nvidia-*-cu12 lib dirs to point at any more in any case -- see the
# install strategy below. The base image's own
# LD_LIBRARY_PATH=/usr/local/tensorrt/lib:... is load-bearing though:
# `import tensorrt` fails with "libnvonnxparser.so.10: cannot open shared
# object file" without it, which is easy to mistake for a broken install
# when testing over ssh (where the image's env isn't applied) rather than
# in the container.
# vllm is what chatterbox_tts/1/model.py has always defaulted to, and its
# prerequisites now exist in this image (/venv/vllm and the T3 export, see
# below). It is 2.4-3.9x faster on time-to-first-audio depending on
# concurrency -- 3.2x at the gateway's default of four sessions. Set
# CHATTERBOX_BACKEND=pytorch to fall back to /venv/chatterbox, which is
# kept precisely so that switch is real.
ENV CHATTERBOX_BACKEND="vllm"
ENV OPAL_PREFIX="/usr/local/mpi"

# INSTALL STRATEGY: --no-deps against a filtered freeze.
#
# deploy/requirements-main.txt is a `pip freeze` dump, which means it is
# already closed under dependencies -- every transitive dependency is in the
# file, pinned. The correct way to install a freeze is therefore --no-deps:
# pip installs exactly what is listed and resolves nothing. Doing that here
# is not a micro-optimization, it fixes three separate problems at once.
#
# 1. SIZE. Letting pip resolve dependencies meant torch==2.9.1+cu128 dragged
#    in a complete parallel CUDA 12.8 userspace as wheels -- onto a base
#    image that is CUDA 13.1 throughout. Measured on a real pod: this RUN
#    used to grow site-packages from 9.3GB to 19GB, of which ~4.2GB was
#    site-packages/nvidia/* (cu12 libs duplicating CUDA 13 system libs) and
#    ~3.9GB was site-packages/tensorrt_libs (duplicating
#    /usr/local/tensorrt/lib). With --no-deps plus the exclusion list, the
#    same RUN adds 379MB. See deploy/image-exclude-main.txt for the full
#    accounting and the reasoning behind every exclusion.
#
# 2. THE RESOLVER FIGHTS. Everything the previous revision of this file did
#    with grep and numpy_override.txt existed to work around pip's resolver:
#    librosa 1.0.0 declares numpy>=2.1.0 while this deployment pins
#    numpy==1.26.4 (ResolutionImpossible), and dropping both pins to escape
#    that made pip backtrack across ~200 other exact pins until it gave up
#    with "resolution-too-deep". With --no-deps there is no resolver, so
#    there is nothing to fight: numpy lands at exactly the 1.26.4 the freeze
#    pins and librosa at exactly 1.0.0, and numpy cannot drift, so
#    deploy/numpy_override.txt is no longer needed here at all. The
#    librosa/numpy combination itself is unchanged from the version that ran
#    the full pipeline end to end -- `pip check` still reports librosa's
#    numpy>=2.1.0 claim as unsatisfied, deliberately, exactly as before.
#
# 3. THE TENSORRT BOOBY TRAP. requirements-main.txt pins PyPI's
#    `tensorrt==10.14.1.48.post1`, which is a METADATA-ONLY wheel: dist-info
#    and nothing else. Installing it made pip uninstall the base image's
#    real, working tensorrt and leave nothing behind, so `import tensorrt`
#    -- and therefore tensorrt_llm, and therefore qwen_llm -- died with
#    ModuleNotFoundError. Earlier revisions repaired that afterwards by
#    force-reinstalling tensorrt-cu13 from pypi.nvidia.com, which is where
#    3.9GB of the duplicate tensorrt_libs came from. tensorrt* is now simply
#    excluded, so the damage never happens and the repair step is gone.
#
# awscli is not in the freeze (docker/triton/entrypoint.sh's optional S3
# restore needs it) so it is installed with its dependencies, after the
# frozen set -- this is the order that was actually verified, and the numpy
# assertion below covers the one thing that order could disturb.
#
# VERIFIED on a Runpod CPU pod against this exact base image: the filtered
# install returns 0, and numpy/tensorrt/torch/librosa/transformers/
# soundfile/datasets all import. `import tensorrt` gives the base image's
# real 10.14.1.48 rather than a metadata-only stub. tensorrt_llm is NOT
# import-checked here on purpose: it needs libcuda.so.1, which no GPU-less
# build host (CPU pod or CI runner) has, so asserting it would break the
# build everywhere it currently succeeds.
COPY deploy/requirements-main.txt /tmp/requirements-main.txt
COPY deploy/image-exclude-main.txt /tmp/image-exclude-main.txt
COPY deploy/filter_requirements.py /tmp/filter_requirements.py
RUN python3 /tmp/filter_requirements.py \
        /tmp/requirements-main.txt /tmp/image-exclude-main.txt \
        > /tmp/requirements-main-filtered.txt \
    && pip install --no-cache-dir --no-compile --no-deps -r /tmp/requirements-main-filtered.txt \
    && pip install --no-cache-dir --no-compile awscli \
    && python3 -c "import numpy; assert numpy.__version__.startswith('1.26'), f'numpy drifted to {numpy.__version__}, see README.md quirk #10'" \
    && python3 -c "import tensorrt; assert not tensorrt.__version__.endswith('post1'), 'tensorrt is PyPI metadata-only stub, see deploy/image-exclude-main.txt'; print('tensorrt', tensorrt.__version__)" \
    && python3 -c "import torch; assert torch.version.cuda.startswith('13'), f'expected the base image CUDA 13 torch, got cuda {torch.version.cuda}'; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && python3 -c "import librosa, transformers, soundfile, datasets, tritonclient; print('librosa', librosa.__version__, '| transformers', transformers.__version__)" \
    && python3 -c "import glob, os, site; sp = site.getsitepackages()[0]; cu12 = [os.path.basename(p) for p in glob.glob(os.path.join(sp, 'nvidia_*_cu12-*.dist-info'))]; assert not cu12, f'CUDA 12 wheels reappeared ({len(cu12)}, e.g. {cu12[:3]}) -- something bypassed deploy/image-exclude-main.txt'; assert not os.path.isdir(os.path.join(sp, 'tensorrt_libs')), 'tensorrt_libs reappeared: 3.9GB duplicate of /usr/local/tensorrt/lib'; print('no duplicate CUDA stack')"

# Chatterbox TTS runs in its own isolated venv, launched as a subprocess by
# triton_model_repo/chatterbox_tts/1/model.py at the hardcoded path
# /venv/chatterbox/bin/python3 -- its torch/torchaudio pins conflict with
# the main TensorRT-LLM stack above, same reason deploy/REBUILD.md keeps it
# isolated on bare metal. Do not `pip install chatterbox-tts` into the
# image's main environment instead of building this.
COPY deploy/requirements-chatterbox.txt /tmp/requirements-chatterbox.txt
COPY deploy/image-exclude-chatterbox.txt /tmp/image-exclude-chatterbox.txt
# Same --no-deps-against-a-filtered-freeze strategy as the main install
# above, and it collapses the same set of special cases: with no resolver,
# numpy lands at the pinned 1.26.4, librosa at 0.11.0 and chatterbox-tts at
# 0.1.7 straight from the freeze, so the grep exclusions and the separate
# --no-deps passes this RUN used to need are all gone.
#
# THE CUDA BUILD IS DELIBERATELY cu130, NOT THE PINNED cu128. This is the
# one place the version pins are not taken verbatim, and it is where ~6GB
# of the image went. requirements-chatterbox.txt pins torch==2.11.0+cu128,
# so this venv used to install a second complete CUDA 12.8 userspace
# (4.7GB of nvidia/* wheels) plus a second torch (1.6GB) and a second
# OpenAI-Triton (0.6GB) -- 6.9GB of an 8.0GB venv was a duplicate GPU
# stack, on an image whose base already provides CUDA 13.1 as system
# libraries. Installing the SAME pinned versions from PyTorch's cu130
# index instead, with --no-deps so no nvidia-* wheels come along, lets
# torch link the base image's CUDA 13 libs: the venv drops to 2.0GB.
#
# Why not share the main environment's torch instead (the obvious move,
# and tried first): the base image ships torch 2.10.0a0+...nv25.12, an
# NVIDIA pre-release build, and there is no torchaudio built against it --
# every released torchaudio fails to load libtorchaudio.so against it.
# chatterbox genuinely needs torchaudio's compiled ops (s3gen imports it,
# xvector.py uses torchaudio.compliance.kaldi, perth uses
# torchaudio.transforms), so the venv needs a matched torch+torchaudio
# pair, and a stock cu130 pair is the smallest one that exists.
#
# VERIFIED on a Runpod CPU pod against this exact base image:
#   - torch 2.11.0+cu130 (torch.version.cuda == 13.0), torchaudio
#     2.11.0+cu130, chatterbox, chatterbox.models.s3gen and perth all
#     import; Spectrogram and compliance.kaldi.fbank produce correct
#     shapes on real tensors
#   - `ldd` on libtorch_cuda.so, libtorch_global_deps.so and
#     libtorchaudio.so reports ZERO unresolved libraries -- the base
#     image's CUDA 13 libs satisfy them with no bundled wheels
#   - the cu13 libs `pip check` reports as "not installed"
#     (cusparseLt, nccl, nvshmem) are all present as system libraries
#     (/usr/local/cuda-13.1/targets/x86_64-linux/lib, /usr/lib/x86_64-linux-gnu)
#   - triton is excluded and stays excluded: nothing in chatterbox, perth
#     or s3tokenizer calls torch.compile or imports triton (grepped, not
#     assumed), and nothing uses torch.distributed
# Still GPU-unverifiable on a CPU host: actual CUDA kernel execution, which
# needs libcuda.so.1.
#
# KNOWN BEHAVIOR CHANGE, accepted: torchcodec is excluded, and in
# torchaudio 2.11 `torchaudio.load`/`torchaudio.save` are implemented via
# torchcodec, so they now raise ImportError in this venv. Nothing calls
# them -- not chatterbox, not perth, not triton_model_repo/chatterbox_tts
# (checked) -- and librosa.load/soundfile, which the code does use, work
# fine. Restoring them is not just a matter of adding the wheel back: a
# cu130 torchcodec still fails to load libtorchcodec_core4.so because the
# base image ships no FFmpeg shared libraries, so it would mean apt-getting
# FFmpeg to enable an API this deployment never calls.
#
# chatterbox-tts==0.1.7's own metadata hard-pins torch==2.6.0 and
# torchaudio==2.6.0. That pin has always been overridden here (installed
# --no-deps against a much newer torch) and still is; `pip check` reports
# it, deliberately. It also declares gradio, its optional demo UI, as a
# hard requirement -- excluded, since nothing in the package imports it.
RUN python3 -m venv /venv/chatterbox \
    && python3 /tmp/filter_requirements.py \
        /tmp/requirements-chatterbox.txt /tmp/image-exclude-chatterbox.txt \
        > /tmp/requirements-chatterbox-filtered.txt \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-compile --no-deps \
        -r /tmp/requirements-chatterbox-filtered.txt \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-compile --no-deps \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0+cu130 torchaudio==2.11.0+cu130 \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-compile --no-deps \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple nvidia-cudnn-cu13 \
    && /venv/chatterbox/bin/pip install --no-cache-dir --no-compile --no-deps \
        cuda-bindings==13.0.3 cuda-toolkit==13.0.2 cuda-pathfinder==1.6.1 \
    && /venv/chatterbox/bin/python3 -c "import torch, torchaudio; assert torch.version.cuda.startswith('13'), f'chatterbox venv got a non-CUDA-13 torch ({torch.version.cuda}); it would need its own 4.7GB cu12 wheel set'; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__)" \
    && /venv/chatterbox/bin/python3 -c "import torch, torchaudio.compliance.kaldi as kaldi; from torchaudio.transforms import Spectrogram, InverseSpectrogram, TimeStretch; assert Spectrogram()(torch.randn(16000)).shape[0] == 201; assert kaldi.fbank(torch.randn(1, 16000)).shape[1] == 23; print('torchaudio compiled ops OK')" \
    && /venv/chatterbox/bin/python3 -c "import chatterbox, perth; from chatterbox.models.s3gen import s3gen; print('chatterbox + perth import OK')" \
    && /venv/chatterbox/bin/python3 -c "import glob, os, site; sp = site.getsitepackages()[0]; cu12 = [os.path.basename(p) for p in glob.glob(os.path.join(sp, 'nvidia_*_cu12-*.dist-info'))]; assert not cu12, f'CUDA 12 wheels reappeared in the chatterbox venv ({len(cu12)}, e.g. {cu12[:3]}) -- see deploy/image-exclude-chatterbox.txt'; print('no duplicate CUDA stack')"

# /venv/vllm -- T3 decode through vLLM instead of eager PyTorch. This is what
# CHATTERBOX_BACKEND=vllm needs, and it is the default the model repo has
# always asked for; the pytorch worker stays as a fallback in
# /venv/chatterbox.
#
# WORTH IT? Measured on an RTX PRO 4500, same GPU, same sentences, same
# harness, comparing this venv's worker against /venv/chatterbox's:
#
#   batch   pytorch   vllm    speedup
#     1      0.62s    0.26s     2.4x
#     2      1.39s    0.50s     2.8x
#     4      2.66s    0.83s     3.2x     <- GATEWAY_MAX_SESSIONS default
#     8      6.08s    1.54s     3.9x
#
# The pytorch path scales linearly because chatterbox-tts==0.1.7 exposes only
# a single-item generate() and the worker loops; vLLM's continuous batching
# makes per-item cost FALL with batch size (0.26s -> 0.19s). Time-to-first-
# audio under four concurrent sessions is 2.66s vs 0.83s, which is the number
# that decides whether a voice agent feels broken. (chatterbox_worker_vllm.py
# claims 5-5.4x; that was against a batched+CUDA-graph PyTorch path this
# package does not expose. 2.4-3.9x is what this deployment actually gets.)
#
# vLLM owns the GPU stack here and chatterbox lives with its versions --
# numpy 2.3.5 and transformers 5.17.0 rather than the freeze's 1.26.4 and
# 5.2.0. Verified: chatterbox-tts imports and synthesizes fine against both.
#
# The uninstall is not an optimization afterthought, it is 2.1GB: vLLM pulls
# cuda-toolkit's cu13 wheels (cuBLAS, cuSOLVER, cuSPARSE, cuFFT, cuRAND,
# NVRTC...) which this base image already provides as CUDA 13.1 system
# libraries. Removing them left the benchmark above unchanged to within noise.
# cudnn/cusparselt/nccl/nvshmem are deliberately KEPT -- torch pins those and
# the base ships different versions, which is exactly the bug that took
# chatterbox_tts to UNAVAILABLE earlier (see image-exclude-chatterbox.txt).
COPY deploy/filter_missing_requirements.py /tmp/filter_missing_requirements.py
RUN python3 -m venv /venv/vllm \
    && /venv/vllm/bin/pip install --no-cache-dir --no-compile vllm==0.29.0 \
    && /venv/vllm/bin/pip uninstall -y -q \
        cuda-toolkit nvidia-cublas nvidia-cuda-cccl nvidia-cuda-crt \
        nvidia-cuda-cupti nvidia-cuda-nvcc nvidia-cuda-nvdisasm \
        nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cufft nvidia-curand \
        nvidia-cusolver nvidia-cusparse nvidia-nvjitlink nvidia-nvtx nvidia-nvvm \
    && python3 /tmp/filter_missing_requirements.py \
        /tmp/requirements-chatterbox.txt /venv/vllm/bin/python3 \
        --also-exclude /tmp/image-exclude-chatterbox.txt \
        > /tmp/requirements-vllm-extra.txt \
    && /venv/vllm/bin/pip install --no-cache-dir --no-compile --no-deps \
        -r /tmp/requirements-vllm-extra.txt \
    && /venv/vllm/bin/pip install --no-cache-dir --no-compile --no-deps chatterbox-tts==0.1.7 \
    && /venv/vllm/bin/python3 -c "import vllm, torch; assert torch.version.cuda.startswith('13'); print('vllm', vllm.__version__, '| torch', torch.__version__)" \
    && /venv/vllm/bin/python3 -c "import chatterbox, numpy, transformers; from chatterbox.tts_turbo import ChatterboxTurboTTS; print('chatterbox OK against numpy', numpy.__version__, 'transformers', transformers.__version__)" \
    && /venv/vllm/bin/python3 -c "import importlib.metadata as md; banned = {'cuda-toolkit','nvidia-cublas','nvidia-cuda-nvrtc','nvidia-cuda-runtime','nvidia-cufft','nvidia-curand','nvidia-cusolver','nvidia-cusparse','nvidia-nvjitlink'}; present = {(d.metadata['Name'] or '').lower().replace('_','-') for d in md.distributions()} & banned; assert not present, f'the cu13 wheel libs came back ({sorted(present)}) -- 2.1GB the base image already provides as CUDA 13.1 system libraries'; print('no duplicate CUDA 13 stack')"

# config.pbtxt files across triton_model_repo/ use absolute paths under
# /workspace/speech-cascade-inference/... (e.g. qwen_llm's engine_dir,
# whisper_asr's assets_dir, chatterbox_tts's reference-voice path) --
# matching that exact layout here means nothing in triton_model_repo/
# needs editing for Docker. Only models/ (weights + compiled engines) is
# expected to arrive via the volume docker-compose.yml mounts at that same
# path -- see docker/triton/entrypoint.sh for the optional S3 restore.
RUN mkdir -p /workspace/speech-cascade-inference
COPY triton_model_repo /workspace/speech-cascade-inference/triton_model_repo

# entrypoint.sh runs this when MODELS_DIR is empty and no S3_BUCKET_URI is
# set, which is what happens on a bare `docker run` or a Runpod pod created
# directly from this image. It pulls the checkpoints from the Hugging Face
# Hub and compiles whisper's TensorRT-LLM engines on whatever GPU the
# container landed on -- the engines are architecture-specific, so building
# them at first start is what lets one image serve different cards.
COPY scripts/build_models_from_scratch.sh /opt/speech-cascade/scripts/build_models_from_scratch.sh
# Needed by that script's T3 export step when CHATTERBOX_BACKEND=vllm.
COPY scripts/export_chatterbox_t3_for_vllm.py /opt/speech-cascade/scripts/export_chatterbox_t3_for_vllm.py

COPY docker/triton/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace/speech-cascade-inference

# Same three ports as the bare-metal deployment (README.md's "Ports"
# section) -- kept off the host's published ports in docker-compose.yml so
# they stay reachable only from other containers on the compose network,
# matching the "internal-only" property README.md describes.
EXPOSE 18000 18001 18002

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
