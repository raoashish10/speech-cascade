# docker/ — containerized deployment (no Vast.ai)

An alternative to `deploy/`'s bare-metal runbook: the same four-model
Triton voice pipeline + streaming gateway, packaged as two containers
(`docker/triton/`, `docker/gateway/`) run via the root `docker-compose.yml`,
instead of hand-installed onto a specific Vast.ai instance via supervisor.

**Why this exists**: `deploy/REBUILD.md` and the main `README.md`'s
"Environment quirks fixed to make this run bare-metal" section document ten
hand-fixed environment issues (MPI, CUDA runtime libs, `PYTHONHOME`/`PATH`
fights, missing `libssl.so.1.1`, missing DCGM, a cuBLAS version clash, numpy
pin drift, ...) and say outright none of it is needed inside NVIDIA's NGC
containers, which ship a matched toolchain. `docker/triton/Dockerfile`
builds on one of those images instead of reassembling the toolchain by
hand, which is what makes this portable to any GPU host with Docker +
`nvidia-container-toolkit` — not just this one Vast.ai instance.

## Build status — read this first

| image | built by | status |
|---|---|---|
| `speech-cascade-gateway` | CI (`.github/workflows/build-images.yml`) | **good** — real BuildKit, pushed to `ghcr.io/raoashish10/speech-cascade-gateway:latest` |
| `speech-cascade-triton` | `crane`, on a GPU pod (see below) | **good** — `ghcr.io/raoashish10/speech-cascade-triton:latest`, verified by running it |

`speech-cascade-triton:latest` was, until 2026-09-18, a kaniko build that was
**silently broken** — files written by `RUN pip install` were missing from its
layers, so it pulled and started cleanly and then died at model load with
`ModuleNotFoundError`. That tag now points at a verified image, so the trap is
gone, but it is worth knowing the failure mode existed: nothing about that
image looked wrong until the models failed to load.

**Verified end to end** by deploying the published image as a Runpod pod on an
RTX PRO 4500 Blackwell: all four models reach `READY`, and `voice_pipeline`
round-trips speech -> ASR -> LLM -> TTS -> speech.

```
input      : 3.17s @ 16000Hz
TRANSCRIPT : Testing 123.
LLM REPLY  : Testing 123. Let me know if you need anything else!
TTS AUDIO  : 3.56s @ 24000Hz
WALL TIME  : 3.25s
```

The triton image is deliberately not built in CI. Three approaches were
tried and all failed for reasons that are now understood and recorded:

- **Docker-in-Docker on a GPU pod** — impossible. Runpod's sandbox blocks
  the `unshare()`/`clone()` syscalls nested containers need.
- **kaniko** (daemonless, chroot-based, the DinD workaround) — produces the
  silently broken image above. Root-caused by running the identical pip
  sequence by hand on the same base image, where it yields a fully working
  environment, then comparing the two.
- **GitHub-hosted runners** — out of disk, twice, ~5 minutes in during the
  base image pull, before any pip install ran. The base image is 15.2 GB
  compressed / ~22 GB extracted against a runner that starts with ~20 GB
  free. The image was slimmed from ~40 GB to ~24 GB first (see "Image size")
  and it did not help, because ~22 GB of the ~24 GB *is* the base image.

Moving it back into CI needs a larger or self-hosted runner, not more
slimming. See `.github/workflows/build-images.yml` for the full write-up.

That work did find and fix real bugs in `docker/triton/Dockerfile` —
missing `--extra-index-url` for `+cu128` torch wheels, a `numpy`/`librosa`
pip resolver conflict, a `resolution-too-deep` error, a `chatterbox-tts`
pin conflict, a metadata-only `tensorrt` wheel that deleted the base
image's working one, and MPI's `OPAL_PREFIX`, among others — see the
Dockerfile's own inline comments for the evidence behind each.

## Building the triton image

If you have an amd64 Linux host with Docker and ~60 GB free, the ordinary
thing works and is what the Dockerfile is for:

```bash
docker build -f docker/triton/Dockerfile -t speech-cascade-triton:latest .
```

**If you don't** — which is the situation this project is actually in, since
Runpod pods can't run Docker and GitHub runners can't fit the base image —
the image can be assembled on a plain GPU pod with no Docker at all. This is
how the current `:latest` was built. `crane` (from google/go-containerregistry)
does nothing but registry HTTP: no daemon, no `unshare()`, no privileged
syscalls, so it runs where Docker and kaniko cannot.

The idea is to do by hand, verifiably, what a builder does: run the
Dockerfile's `RUN` steps natively, then package the filesystem delta as a
layer on top of the unmodified base.

```bash
# On a GPU pod running the SAME base image the Dockerfile builds FROM.
curl -sSL https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz \
  | tar xz -C /usr/local/bin crane

# 1. Run the Dockerfile's RUN steps natively (see its two RUN blocks), and
#    the COPY steps by hand: triton_model_repo ->
#    /workspace/speech-cascade-inference/, entrypoint.sh -> /usr/local/bin/,
#    scripts/build_models_from_scratch.sh -> /opt/speech-cascade/scripts/.

# 2. Compute the delta against the base image's own flattened file listing.
crane export <base-image> - | tar -tf - > /tmp/base-files.txt
python3 deploy/make_image_layer.py          # writes /tmp/layer.tgz

# 3. Append it to the base and set the image config in one shot.
crane mutate <base-image> --platform linux/amd64 \
  --append /tmp/layer.tgz \
  -e CHATTERBOX_BACKEND=pytorch -e OPAL_PREFIX=/usr/local/mpi \
  --entrypoint /usr/local/bin/entrypoint.sh \
  -w /workspace/speech-cascade-inference \
  --exposed-ports 18000/tcp,18001/tcp,18002/tcp \
  -t ghcr.io/<owner>/speech-cascade-triton:latest
```

Two things make this trustworthy rather than a second kaniko:

- **The delta is explicit.** It is not inferred by a builder that might quietly
  drop files; it is the set of paths under the touched roots whose mtime is
  newer than the base image build, plus `.wh.` whiteouts for the files `pip`
  *removed* when it replaced a package. Without those whiteouts the base
  layer's copy resurfaces and `importlib.metadata` sees two versions of the
  same package. The counts are printed before packaging (for the current
  image: 67,538 added/modified files, 1,263 whiteouts).
- **The result is tested by running it**, not by trusting the build. kaniko's
  output failed that test instantly.

**Check the layers before deploying anything:**

```bash
python3 deploy/check_image_layers.py ghcr.io/<owner>/speech-cascade-triton:latest
```

It streams just the layers this repo adds and fails on the two packaging
faults below. Both were hit for real, both stop the image from pulling at
all, and both are invisible until a pod tries to start -- so this takes
seconds and replaces a ~15 minute pod deploy as the way you find out.

**The two traps.** Loud failures, unlike kaniko's silent one, but each cost
a pod deploy to discover:

- **Hardlinks.** Package with `tar --hard-dereference`, and never put a
  directory *and* its own contents in the same file list. If tar emits a
  hardlink entry whose target isn't in the archive you get
  `failed to register layer: link ...: no such file or directory`. The cuDNN
  wheel triggered this.
- **Build the layer on Linux, not macOS.** BSD tar attaches macOS extended
  attributes, and the Linux runtime cannot set them:
  `failed to register layer: lsetxattr /workspace: xattr
  "com.apple.provenance": operation not supported`. Worse, the pull *retries*
  rather than failing fast, so the pod sits at `runtime: null` emitting
  "Downloading" lines that look like slow progress. If you must build on a
  Mac, use `COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs --no-acls`.

Measured: the delta is ~3.5 GB uncompressed / **1.3 GB compressed** on top of
the base's 15.2 GB. Iterating is fast — a second push that changed only
`entrypoint.sh` completed in about a second, because every other blob was
already in the registry.

The Dockerfile asserts its own critical invariants during the build, so a
successful build is meaningful rather than merely quiet: it checks that
numpy has not drifted off 1.26, that `tensorrt` is the base image's real
module and not PyPI's metadata-only stub, that torch is a CUDA 13 build,
that the chatterbox venv's torch/torchaudio are CUDA 13 and their compiled
ops actually run, and that no duplicate CUDA 12 wheel set has reappeared in
either environment. A kaniko-style layer-dropping failure cannot pass those.

To publish it:

```bash
docker tag speech-cascade-triton:latest ghcr.io/raoashish10/speech-cascade-triton:latest
docker push ghcr.io/raoashish10/speech-cascade-triton:latest
```

After the first GPU deployment of a hand-built image, update the "Still
unverified" note below — the GPU run of the *slimmed* recipe is the one
piece of verification still outstanding.

**Verified end-to-end on a real GPU** (RTX PRO 4500 Blackwell, sm_120,
Runpod): all four models reach Triton state `READY` and the full
`voice_pipeline` ensemble round-trips speech -> ASR -> LLM -> TTS -> speech.
A 11.0s input clip produced a correct transcript, a coherent LLM reply, and
7.0s of generated 24kHz audio in 13.2s wall time, with weights/engines built
from scratch (no S3) per `scripts/build_models_from_scratch.sh`.

Environment bugs found and fixed by that GPU run, none of which a
successful `docker build` would have caught:
- the original base image (`nvcr.io/nvidia/tensorrt-llm/release`) shipped no
  `tritonserver` binary at all -- switched to
  `nvcr.io/nvidia/tritonserver:<date>-trtllm-python-py3`
- `OPAL_PREFIX` (this image's OpenMPI can't find its own runtime data, which
  kills `tensorrt_llm` and therefore `qwen_llm`)
- PyPI's pinned `tensorrt`/`tensorrt_cu13` wheels are metadata-only and
  *delete* the base image's working `tensorrt` module -- restored from
  NVIDIA's index
- `chatterbox_tts` defaults to a `CHATTERBOX_BACKEND=vllm` path needing a
  `/venv/vllm` and a model export that don't exist in this repo; pinned to
  the `pytorch` backend instead

**Image size**: the triton image went from ~40GB unpacked (~27GB
compressed) to ~24GB unpacked. ~19GB of it was duplicate CUDA userspace:
`deploy/requirements-main.txt` and `deploy/requirements-chatterbox.txt` are
`pip freeze` dumps from the old CUDA 12.8 bare-metal box, and replaying them
verbatim onto a CUDA 13.1 base image made each of the two environments
install its own complete `nvidia-*-cu12` wheel set beside the CUDA 13
libraries the base already ships as system libraries. Measured on a pod:

| | before | after |
|---|---|---|
| base image | 22 GB | 22 GB |
| main pip install | +9.7 GB | **+0.4 GB** |
| `/venv/chatterbox` | +8.0 GB | **+2.1 GB** |

Both installs now use `--no-deps` against the freeze (which is already
dependency-closed, so there is nothing to resolve) minus an explicit
exclusion list -- `deploy/image-exclude-main.txt` and
`deploy/image-exclude-chatterbox.txt`, applied by
`deploy/filter_requirements.py`. Those files carry the per-package reasoning;
the short version is that the base image already provides torch, tensorrt,
tensorrt_llm, triton and all of CUDA 13. Dropping the resolver also removed
the `numpy`/`librosa` `ResolutionImpossible` and `resolution-too-deep`
workarounds and the `tensorrt` metadata-only-wheel repair step, since none of
those failure modes can occur without dependency resolution.

What's left is mostly irreducible without changing base images: ~22GB of the
~24GB is the NGC base itself (`tensorrt_llm` 3.5GB, `flash_attn`'s single
1.7GB `.so`, torch, CUDA 13, Triton).

**The slimmed install is GPU-verified.** On an RTX PRO 4500 Blackwell:
`tensorrt_llm 1.2.0` imports (it needs `libcuda.so.1` *and* working MPI, so a
CPU host cannot check it at all), `tensorrt` resolves to the base image's real
10.14.1.48 rather than PyPI's metadata-only stub, both environments run CUDA
matmuls, `torchaudio`'s compiled ops run on-device, `build_models_from_scratch.sh`
downloads the Hub checkpoints and compiles the whisper engines, and all four
models load and serve a request.

**One bug the slimming introduced, caught only by running it.** Excluding the
bundled cuDNN from `/venv/chatterbox` on the grounds that the base image
provides one was wrong: the base ships cuDNN 9.17.0 and `torch 2.11.0+cu130`
is compiled against 9.19.0, so torch refuses to run RNN kernels
(`cuDNN version incompatibility`) and `chatterbox_tts` goes `UNAVAILABLE`,
since its s3gen voice encoder uses LSTMs. `pip check` had warned about exactly
this and the warning was dismissed because `ldd` showed nothing unresolved --
bad evidence, since cuDNN is `dlopen`'d lazily and never appears in `ldd`
output. Fixed by installing `nvidia-cudnn-cu13` (+519MB, so the venv is
~3.0GB rather than 2.1GB). The lesson is recorded in
`deploy/image-exclude-chatterbox.txt`: static inspection cannot establish that
a lazily-loaded library is unused.

Also still unverified, unchanged from before: the `vllm` chatterbox backend
(~5x faster per its own docstring) -- see
`scripts/export_chatterbox_t3_for_vllm.py`, a from-scratch reconstruction of
the missing T3 export step, never run. Also the `gateway` container against a
live `triton` (only the triton half was exercised directly), and
sustained/concurrent load.

**One accepted behavior change** from the slimming: `torchcodec` is no longer
installed in `/venv/chatterbox`, and `torchaudio` 2.11 implements
`torchaudio.load`/`save` via torchcodec, so those two functions now raise
`ImportError` there. Nothing in `chatterbox`, `perth` or
`triton_model_repo/chatterbox_tts/` calls them, and `librosa.load`/`soundfile`
(which the code does use) work normally. Adding the wheel back isn't
sufficient either -- torchcodec needs FFmpeg shared libraries the base image
doesn't ship.

## Quick start

Weights and compiled engines are **not** in the image -- they live at
`/workspace/speech-cascade-inference/models`, which `docker-compose.yml`
mounts as the `inference-data` volume.

**You don't have to populate it yourself.** `docker/triton/entrypoint.sh`
bootstraps on first start when that directory is empty: it pulls the
checkpoints from the Hugging Face Hub and compiles whisper's TensorRT-LLM
engines on whatever GPU the container landed on. No S3 bucket and no
pre-built artifacts are required, which is what makes the image usable
directly as a Runpod pod template with nothing mounted at all.

That is the default rather than a fallback, because the engines are
GPU-architecture-specific (see "GPU portability caveat"). Compiling them at
first start on the card that will serve them is what lets one image work
across different GPUs; baking them in would tie the image to one card.

**Verified** by deploying the published image as a Runpod pod with nothing
mounted, on an RTX PRO 4500 Blackwell:

```
models/ is empty and S3_BUCKET_URI is unset -- building from
scratch: Hugging Face checkpoints + whisper engines for this GPU.
==> qwen_llm: downloading Qwen3-8B-NVFP4 (~6GB)
Fetching 12 files: 100% [00:32]
==> whisper_asr: building TensorRT-LLM encoder/decoder engines
    encoder engine generation completed in 38.3s
    decoder: total time of building all engines 00:00:07
==> done. models/ now contains: Qwen3-8B-NVFP4, whisper-base-trtllm
```

**2 min 43 s** from container start to a populated `models/`, of which ~32s
was the Hub download and ~45s the two engine builds. Mounting a volume at
that path makes it a one-time cost per volume instead of per start.

### Startup cost, measured

| | cold (new pod) | warm (stop -> start) |
|---|---|---|
| image pull + extract | 4-11 min | **skipped** |
| HF bootstrap | 2m43s | **skipped** |
| four models to `READY` | ~4 min | ~4 min |
| **total** | **11-18 min** | **4m07s** |

Two things worth knowing about those numbers:

- **The pull dominates cold start and is almost entirely the base image.**
  It varies by datacenter: the same image pulled in 4m14s in `EU-RO-1` and
  was still retrying after 25 minutes in `EUR-IS-1`. Pin
  `dataCenterIds` to somewhere you have pulled before; that is worth more
  than any further image slimming, since ~22 of the ~24GB is the NGC base.
- **A stopped pod keeps its container disk**, at least across a
  `stop` -> `start` on the same machine: the warm restart above did *not*
  re-run the bootstrap (Triton began loading 115s in, less than the 163s the
  bootstrap alone takes) and `models/` was still populated. Runpod's API
  describes container disk as "ephemeral, wiped on restart", so do not rely
  on this surviving a reschedule onto a different host -- but stopping
  rather than terminating clearly does avoid both the pull and the bootstrap.

The remaining ~4 min is the models themselves and is not container overhead:
TensorRT-LLM's import, qwen's engine build (~32s) and 6.1GB engine load,
whisper's two engines, and chatterbox's ~3GB model. The bare-metal
deployment pays the same cost -- `README.md` documents "~5-6 min" for a
supervisor restart. `entrypoint.sh` loads the four serially on purpose (see
its own comment about an OOM-killed stub); parallelising is the obvious
lever if this ever needs to be faster, and has not been tried against this
image.

Set `S3_BUCKET_URI` instead to restore a prebuilt tree, which is faster but
needs somewhere to have built it first.

`chatterbox_tts` additionally needs its reference voice clip
(`ref_audio_path`) -- a deployment artifact you supply yourself, and it must
be **longer than 5 seconds** or the model refuses to load.

**Verified**: a pod from the published image with no clip and nothing
mounted reaches all four models `READY`, including `chatterbox_tts`, and
round-trips the pipeline (`"Testing 123."` -> LLM reply -> 3.40s of 24kHz
audio). The voice is chatterbox's built-in one.

**You no longer need to supply one just to get the model up.**
`chatterbox-turbo`'s checkpoint includes `conds.pt`, a built-in voice that
`ChatterboxTurboTTS.from_pretrained()` loads automatically, so the model has
a working voice with no external artifact at all. `ref_audio_path` now
defaults to empty, meaning "use that built-in voice".

This was a real defect rather than a missing file. `ref_audio_path` was a
required config parameter pointing at `scripts/pipeline_output.wav` -- which
is in neither this repo nor the image -- so a container started without it
loaded the entire model and then died:

```
chatterbox_tts load request FAILED
  TritonModelException: chatterbox worker failed to start: None
```

(observed on a Runpod pod from the published image, where the other three
models reached `READY`). That message is also unhelpfully opaque: the
worker's real error is swallowed, which is worth fixing separately.

**Set `ref_audio_path` when you want a specific cloned voice.** Reproducible
ways to get the clip to the container, in the order I'd pick them:

1. **Ship it with the weights.** Put the clip in the same Hugging Face repo
   the checkpoints come from and fetch it in
   `scripts/build_models_from_scratch.sh`. This is the most consistent
   option -- it arrives exactly the way every other model artifact already
   does, versioned with them, and needs no extra infrastructure.
2. **Commit it to the repo** and `COPY` it in the Dockerfile. A 41s mono wav
   is ~2MB, which git handles fine. Simplest and fully hermetic -- but only
   if you have the rights to redistribute that recording, which is the usual
   reason a voice clip is kept out of a repo.
3. **`S3_BUCKET_URI`**, which `entrypoint.sh` already restores from. Reuses
   existing machinery, at the cost of credentials.

Whichever you choose, the clip must be **longer than 5 seconds** (chatterbox
asserts this), and a configured-but-missing path is now a startup error
rather than a silent fallback -- a deployment that means to use a particular
voice should not quietly end up on the default one.

```bash
cp .env.example .env   # fill in S3 creds (or skip and populate the volume yourself) + GATEWAY_AUTH_TOKEN
docker compose up --build
```

Requires a host with an NVIDIA GPU and `nvidia-container-toolkit` installed
(`nvidia-ctk runtime configure --runtime=docker`, or whatever your provider's
own Docker-native image/template already sets up — RunPod and SaladCloud
both do, per the "which GPU provider" discussion this came out of).

`triton` takes ~5-6 minutes to become healthy (TensorRT-LLM import + engine
load, same cost `README.md`'s "Managing the service" section documents for
the bare-metal deployment) — `gateway` waits on `service_healthy` before
starting.

## What's different from the bare-metal deployment

| | Bare metal (`deploy/`) | Docker (`docker/`) |
|---|---|---|
| Toolchain | Hand-assembled to match the driver/CUDA on one specific instance | Comes matched from the NGC base image |
| Process manager | `supervisor` | `docker compose` (`restart: unless-stopped`) |
| Triton <-> gateway | both bind `127.0.0.1` on the same box | separate containers, `TRITON_URL=triton:18001` over the compose network |
| External auth | Vast.ai's Caddy edge checks `$OPEN_BUTTON_TOKEN` in front of the gateway | `GATEWAY_AUTH_TOKEN` checked by the gateway itself (see `streaming_gateway/server.py`) — no Vast.ai-specific edge to depend on |
| Weights/engines | restored via `aws s3 sync` into `/workspace` directly | same `aws s3 sync`, run by `docker/triton/entrypoint.sh` into a named volume |
| Monitoring stack (Prometheus/Grafana/exporters) | supervisor services, see `monitoring/` | **not containerized yet** — see "Not yet done" below |

Everything under `triton_model_repo/*/config.pbtxt` still points at
absolute paths like `/workspace/speech-cascade-inference/models/...` —
deliberately unchanged, since `docker/triton/Dockerfile` reproduces that
same layout inside the image/volume rather than editing every config file.

## Auth

The bare-metal deployment relies on Vast.ai's own Caddy reverse-proxy edge
to check a shared token in front of the gateway (see
`streaming_gateway/README.md`). That edge doesn't exist here, so
`streaming_gateway/server.py` now has its own optional shared-token check
(`GATEWAY_AUTH_TOKEN` env var, `?token=` query param or `Authorization:
Bearer` header — same client-facing convention). Leave it unset only if
something else in front of the gateway (a reverse proxy you add, a VPN,
your provider's own auth) is already handling this — an empty token means
the check is skipped entirely, same as today's bare-metal behavior with no
Caddy in front of it.

## GPU portability caveat

Moving off the current Vast.ai RTX 5070 Ti instance doesn't make the
*compiled TensorRT engines* portable — `README.md`'s "Testing each stage"
and `scripts/build_engine.sh`'s own header both say the `.engine` output is
tied to the exact GPU architecture + TensorRT-LLM version it was built
with. A different GPU (even another Blackwell card — RTX 5080/5090 instead
of 5070 Ti) needs `scripts/build_engine.sh` / `scripts/quantize_*.py` rerun
against it (see `deploy/REBUILD.md` section 4, or `deploy/ansible`'s
`build_source: scratch`), not the `.engine` files copied over. Docker
solves the toolchain-matching problem, not the engine-portability one.

## Not yet done

- Monitoring stack (`monitoring/`, `prometheus.yml`, Grafana dashboards,
  the alert-notifier and Triton-state-exporter supervisor services) isn't
  containerized here yet — still only documented for the bare-metal/
  supervisor deployment.
- **The triton image build is manual.** It is no longer *unverified* -- the
  published image was built with `crane` on a pod and confirmed by running
  it -- but it is a procedure a person follows, not a pipeline. See
  "Building the triton image". Automating it would mean running that same
  procedure on a scheduled GPU pod, which is a real option and simply
  hasn't been done.
- **`chatterbox_tts` needs a reference voice clip you supply.** It is read
  from `ref_audio_path` (`scripts/pipeline_output.wav`) and is a deployment
  artifact, not something reproducible from public sources, so it is not in
  the image or the repo. Concretely: chatterbox asserts the clip is
  **longer than 5 seconds** (`Audio prompt must be longer than 5 seconds!`)
  and the model goes `UNAVAILABLE` without a usable one. The end-to-end run
  above used a tiled copy of `tests/fixtures/vad_sample_16k.wav` as a
  stand-in purely to exercise the path -- it is not a real voice and the
  output quality means nothing.
- **`gateway` has still never been run against a live `triton`.** Only the
  triton half has been exercised directly.
- `docker/triton/entrypoint.sh`'s sequential model-load order carries over
  the bare-metal OOM workaround (see its own comments) without
  re-verifying the memory ceiling that caused it against this image/host.
