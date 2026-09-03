# Ansible playbook — fresh instance → working deployment

This automates `deploy/REBUILD.md`. Read that runbook first if you haven't —
this playbook is a direct translation of its steps 1, 2, 3, and 6 (system
packages, S3 restore, three venvs, supervisor services), plus optionally
step 4 (building the LLM and Whisper TensorRT-LLM engines from scratch).

**Two ways to get engines onto the box**, picked via the `build_source` var
in `group_vars/all/vars.yml`:

- **`build_source: s3` (default, unchanged from before)** — restores
  pre-built checkpoints/engines from the private S3 bucket. Fast (minutes,
  no GPU compute), but only usable by someone with access to that bucket,
  and the restored engines are only valid on hardware matching the exact
  GPU architecture + TensorRT-LLM version they were originally built with
  — see `scripts/build_engine.sh`'s own header on why compiled engines
  aren't portable.
- **`build_source: scratch`** — runs the quantization/build commands from
  REBUILD.md section 4 on the target instance itself: NVFP4-quantizes the
  LLM (`scripts/quantize_nvfp4.py`) and compiles Whisper's encoder/decoder
  TensorRT-LLM engines from the upstream example. This is the path for
  someone replicating this project **without** access to the private
  bucket — real GPU time (quantization + calibration + two `trtllm-build`
  calls), and **unverified end-to-end**: REBUILD.md's own 4b/4c sections
  are marked "reconstructed from source/prose, not re-run on this
  instance," and these tasks are a direct translation of that — not a
  confirmed-working automation. Treat a first real run of this as the
  actual verification, not this playbook's existence.

Either way, the Triton server binary itself (`triton_server/extracted/`)
and the `libssl1.1` compat shim always come from S3 (or the compat_libs
extraction, which this playbook does unconditionally) — there's no
from-scratch path documented anywhere for the server binary itself, since
it isn't a model.

## Layout

| File | What it does |
|---|---|
| `playbook.yml` | The whole thing, tagged by phase (see below). |
| `inventory.example.ini` | Copy to `inventory.ini`, point it at the fresh instance. |
| `group_vars/all/vars.yml` | Non-secret vars (paths, package names, version pins). |
| `group_vars/all/vault.yml.example` | Template for secrets. Copy to `group_vars/all/vault.yml`, fill in real values, then `ansible-vault encrypt group_vars/all/vault.yml`. **Never commit the unencrypted file.** |

## Secrets this needs

- AWS credentials with read access to `s3://ashish-s3-coding-bucket/` — only needed when `build_source: s3` (the default).
- A GitHub token with read access to this repo (to clone it onto the fresh instance) — skip this if you're running the playbook from a checkout that's already there, or if the instance already has SSH deploy keys set up.
- An `HF_TOKEN` (`vault_hf_token`) — needed when `build_source: scratch`. Must be from an account that has accepted the terms of the gated `nvidia/Nemotron-Post-Training-Dataset-v2` dataset used for NVFP4 calibration, not just any token. Not needed when `build_source: s3`.

None of these live in this repo. Put them in `group_vars/all/vault.yml` (vault-encrypted) or pass as `-e` extra-vars / environment on the `ansible-playbook` command line.

## Usage

```bash
cp inventory.example.ini inventory.ini        # edit: point at the fresh instance's IP + mapped SSH port
cp group_vars/all/vault.yml.example group_vars/all/vault.yml
$EDITOR group_vars/all/vault.yml              # fill in real credentials
ansible-vault encrypt group_vars/all/vault.yml

# full rebuild, restoring pre-built engines from S3 (default, needs bucket access):
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass

# full rebuild, building engines from scratch instead (no bucket access needed,
# but real GPU time and an HF token that's accepted the gated calibration dataset):
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass -e build_source=scratch

# or run one phase at a time:
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags packages
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags s3_restore        # honors build_source
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags venvs
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags build_from_scratch -e build_source=scratch  # needs venvs first
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags supervisor
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags verify
```

Targeting the box you're already on (no SSH hop)? Use the bundled
`inventory.local.ini` instead — `ansible-playbook -i inventory.local.ini playbook.yml ...`.

## Walkthrough: replicating from scratch, step by step

For someone new to this project, without access to the private S3 bucket,
end to end:

1. **Provision a fresh GPU instance.** Needs to be a GPU box (this project
   targets an RTX 5070 Ti / SM120-class card) — the playbook itself doesn't
   provision the machine, that's outside its scope, same as REBUILD.md.
2. **Clone this repo** onto whichever machine you're running
   `ansible-playbook` from. The playbook separately clones it *onto the
   target instance* in the `repo` task below, but you need a local
   checkout first to have `deploy/ansible/` to run at all.
3. **Set up the inventory**: copy `inventory.example.ini` to
   `inventory.ini` and point it at the fresh instance's IP + SSH port, or
   use the bundled `inventory.local.ini` if running directly on the box
   (no SSH hop).
4. **Set up secrets**: copy `group_vars/all/vault.yml.example` to
   `group_vars/all/vault.yml`, fill in real values, then
   `ansible-vault encrypt group_vars/all/vault.yml`. The one that actually
   matters for a from-scratch replication is `vault_hf_token` — it must
   come from an HF account that's accepted the terms of the gated
   `nvidia/Nemotron-Post-Training-Dataset-v2` dataset (used for NVFP4
   calibration), not just any token. (`vault_aws_access_key_id`/`secret`
   are only needed for the S3 path, which a new person without bucket
   access wouldn't be using anyway.)
5. **Set `build_source: scratch`** — either edit `group_vars/all/vars.yml`
   directly, or just pass `-e build_source=scratch` on the command line.
   This is the switch that builds everything locally instead of trying to
   `aws s3 sync` from a bucket you don't have access to.
6. **Run it**:
   `ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass -e build_source=scratch`.
   From here the playbook runs through its tagged phases in order:
   - **`repo`** — clones this git repo onto the target instance.
   - **`packages`** — installs the system packages this project needs
     bare-metal (MPI, CUDA 13.2 toolkit-only libs, DCGM), plus extracts the
     `libssl1.1` compat shim Ubuntu 24.04 doesn't ship. Identical regardless
     of `build_source`.
   - **`s3_restore`** — skipped (since `build_source != s3`) except for a
     debug message flagging one honest gap: the Triton server binary
     itself has no from-scratch build path documented anywhere, so it has
     to come from somewhere else — it isn't a model, so this playbook
     can't conjure it.
   - **`venvs`** — creates the three pinned Python virtual environments
     (`main`, `gateway`, `chatterbox`) from the exact `pip freeze`
     snapshots checked into the repo, then verifies numpy didn't silently
     get upgraded (which would break TensorRT-LLM's compiled bindings).
   - **`build_from_scratch`** (only runs because `build_source: scratch`):
     1. Fails fast with a clear message if `vault_hf_token` is empty.
     2. Installs `nvidia-modelopt[hf]` into `/venv/main`.
     3. Runs `scripts/quantize_nvfp4.py` to produce the NVFP4-quantized LLM
        checkpoint.
     4. Clones the TensorRT-LLM GitHub repo (just for its Whisper example
        scripts, which aren't shipped in the pip package).
     5. Downloads Whisper-base's weights and assets.
     6. Converts the Whisper checkpoint and runs the two `trtllm-build`
        calls (encoder, then decoder) to produce its TensorRT-LLM engines.
   - **`supervisor`** — installs and starts the gateway and Triton as
     supervisor-managed services (refuses to touch Triton if it's already
     running live, unless explicitly confirmed).
   - **`verify`** — polls Triton's `/v2/repository/index` until all four
     models report `READY`, and optionally runs the integration test suite
     against the live server.
7. **Wait.** This path takes real time and GPU compute (quantization +
   calibration + two `trtllm-build` compiles), unlike the S3 path, which is
   just a fast download.
8. **Know it's genuinely unverified end-to-end** — the one honest caveat
   worth repeating: these tasks are a direct translation of REBUILD.md's
   documented commands, not a rerun-and-confirmed automation. If it breaks
   on a first real run, that's useful signal for closing a real gap, not a
   sign you did something wrong.

## The one dangerous tag: `supervisor`

Installing and (re)starting `speech-cascade-triton` on a box where Triton is
**already serving live traffic** duplicates GPU model loads and fights over
ports 18000-18002 — REBUILD.md flags this explicitly, which is why the
upstream repo hadn't installed the service live as of this writing. The
`supervisor` tag therefore **refuses to touch `speech-cascade-triton` if
it's already running**, unless you pass `-e confirm_triton_restart=true`.
The gateway service (cheap to restart) isn't guarded the same way.

## What's intentionally out of scope

- **The Triton server binary itself and the libssl1.1 compat shim** always
  come from S3/the Ubuntu package respectively, regardless of
  `build_source` — there's no from-scratch build documented anywhere for
  these, since they aren't models.
- **`chatterbox_tts`'s weights** — not restored by either path.
  `ChatterboxTurboTTS.from_pretrained()` pulls them from the HF cache the
  first time it actually runs, same as any other `from_pretrained()`-based
  model — nothing for this playbook to do.
- **Monitoring stack** (Prometheus/Grafana/alert-notifier/triton-state-exporter
  in `deploy/supervisor/`) — not covered by REBUILD.md, not part of this pass.
- **The Vast.ai base image itself** (Caddy, the portal, Jupyter, syncthing,
  tensorboard) — that's the image, not something this project deploys;
  nothing to automate there beyond what's already baked in.
- **Non-Blackwell GPUs on `build_source: scratch`.** The `build_from_scratch`
  tasks assume Blackwell-class hardware (NVFP4 needs native FP4 tensor
  cores, which only exist from Blackwell onward — this isn't a config
  choice, older architectures can't run it at all). If you're replicating
  this on an older GPU, use `scripts/quantize_fp8.py` + `build_engine.sh`
  by hand instead (REBUILD.md §4a — already scripted, GPU-generation-
  agnostic) and point `triton_model_repo/nemotron_llm/config.pbtxt` at the
  result yourself. Not automated here — no evidence yet that anyone's
  actually hit this, so it isn't worth the added complexity (see the
  git history around this doc for the reasoning) until someone does.

## The `build_from_scratch` tag is unverified end-to-end

Same honesty caveat REBUILD.md gives its own §4b/§4c: these tasks are a
direct translation of those sections' documented commands, not a
rerun-and-confirmed automation. If you run `build_source: scratch` and hit
a failure, that's genuinely useful signal for closing the gap REBUILD.md
already flags — please report back what broke rather than silently
patching around it, the same way this project's other "reconstructed, not
verified" gaps got closed (`scripts/quantize_nvfp4.py` itself started this
way).
