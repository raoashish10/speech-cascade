# Ansible playbook — fresh instance → working deployment

This automates `deploy/REBUILD.md`. Read that runbook first if you haven't —
this playbook is a direct translation of its steps 1, 2, 3, and 6 (system
packages, S3 restore, three venvs, supervisor services). It deliberately
does **not** automate step 4 (rebuilding compiled TensorRT engines from
scratch) — REBUILD.md is explicit that step 2's S3 restore already gives
you every engine this deployment needs, and step 4's own paths are marked
"reconstructed, not verified end-to-end" in REBUILD.md's "Known gaps". If
you actually need to regenerate an engine, follow REBUILD.md section 4 by
hand rather than trusting an unverified automated version of it.

## Layout

| File | What it does |
|---|---|
| `playbook.yml` | The whole thing, tagged by phase (see below). |
| `inventory.example.ini` | Copy to `inventory.ini`, point it at the fresh instance. |
| `group_vars/all.yml` | Non-secret vars (paths, package names, version pins). |
| `group_vars/vault.yml.example` | Template for secrets. Copy to `group_vars/vault.yml`, fill in real values, then `ansible-vault encrypt group_vars/vault.yml`. **Never commit the unencrypted file.** |

## Secrets this needs

- AWS credentials with read access to `s3://ashish-s3-coding-bucket/` (to restore weights/engines).
- A GitHub token with read access to this repo (to clone it onto the fresh instance) — skip this if you're running the playbook from a checkout that's already there, or if the instance already has SSH deploy keys set up.
- An `HF_TOKEN` — only needed if you also run the (separate, unautomated) NVFP4 requantization in REBUILD.md §4b; not needed for a normal restore.

None of these live in this repo. Put them in `group_vars/vault.yml` (vault-encrypted) or pass as `-e` extra-vars / environment on the `ansible-playbook` command line.

## Usage

```bash
cp inventory.example.ini inventory.ini        # edit: point at the fresh instance's IP + mapped SSH port
cp group_vars/vault.yml.example group_vars/vault.yml
$EDITOR group_vars/vault.yml                  # fill in real credentials
ansible-vault encrypt group_vars/vault.yml

# full rebuild, in REBUILD.md order:
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass

# or run one phase at a time:
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags packages
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags s3_restore
ansible-playbook -i inventory.ini playbook.yml --ask-vault-pass --tags venvs
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

- **Engine rebuilds** (REBUILD.md §4) — restore from S3 instead; see above.
- **Monitoring stack** (Prometheus/Grafana/alert-notifier/triton-state-exporter
  in `deploy/supervisor/`) — not covered by REBUILD.md, not part of this pass.
- **The Vast.ai base image itself** (Caddy, the portal, Jupyter, syncthing,
  tensorboard) — that's the image, not something this project deploys;
  nothing to automate there beyond what's already baked in.
