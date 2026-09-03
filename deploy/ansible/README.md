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
| `group_vars/all.yml` | Non-secret vars (paths, package names, version pins). |
| `group_vars/vault.yml.example` | Template for secrets. Copy to `group_vars/vault.yml`, fill in real values, then `ansible-vault encrypt group_vars/vault.yml`. **Never commit the unencrypted file.** |

## Secrets this needs

- AWS credentials with read access to `s3://ashish-s3-coding-bucket/` — only needed when `build_source: s3` (the default).
- A GitHub token with read access to this repo (to clone it onto the fresh instance) — skip this if you're running the playbook from a checkout that's already there, or if the instance already has SSH deploy keys set up.
- An `HF_TOKEN` (`vault_hf_token`) — needed when `build_source: scratch`. Must be from an account that has accepted the terms of the gated `nvidia/Nemotron-Post-Training-Dataset-v2` dataset used for NVFP4 calibration, not just any token. Not needed when `build_source: s3`.

None of these live in this repo. Put them in `group_vars/vault.yml` (vault-encrypted) or pass as `-e` extra-vars / environment on the `ansible-playbook` command line.

## Usage

```bash
cp inventory.example.ini inventory.ini        # edit: point at the fresh instance's IP + mapped SSH port
cp group_vars/vault.yml.example group_vars/vault.yml
$EDITOR group_vars/vault.yml                  # fill in real credentials
ansible-vault encrypt group_vars/vault.yml

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

## The `build_from_scratch` tag is unverified end-to-end

Same honesty caveat REBUILD.md gives its own §4b/§4c: these tasks are a
direct translation of those sections' documented commands, not a
rerun-and-confirmed automation. If you run `build_source: scratch` and hit
a failure, that's genuinely useful signal for closing the gap REBUILD.md
already flags — please report back what broke rather than silently
patching around it, the same way this project's other "reconstructed, not
verified" gaps got closed (`scripts/quantize_nvfp4.py` itself started this
way).
