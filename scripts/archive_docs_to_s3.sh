#!/bin/bash
# Uploads the 13 investigation docs that used to live in docs/ (see
# docs/README.md) to S3, extracting them directly from git history rather
# than assuming they're still present in the working tree -- safe to run
# before or after the move-docs-to-s3 branch merges, as long as this repo's
# git history still has the commit below.
#
# Needs real AWS credentials for the destination bucket -- this was written
# from a session with no access to it, so it's untested against the real
# bucket. Run it once, then spot-check with `aws s3 ls` before trusting it.
#
# Usage:
#   scripts/archive_docs_to_s3.sh

set -euo pipefail

# Last commit where all 13 docs still existed in git, before the
# move-docs-to-s3 branch removed them (see that branch's first commit's
# parent). Pin this rather than using HEAD so the script keeps working
# even from a checkout that's long since moved past this point.
SOURCE_REF="${SOURCE_REF:-1b6395febc6019ea8cf0a422d4a30c494a21bd69}"

S3_DEST="${S3_DEST:-s3://ashish-s3-coding-bucket/speech-cascade-inference/docs/}"

DOCS=(
  kokoro-tts-capacity-fix.md
  kokoro-tts-vram-headroom.md
  kv-cache-investigation.md
  monitoring.md
  nemotron-batch-size-scaling.md
  nemotron-response-quality.md
  nemotron-token-cap-investigation.md
  nvfp4-candidate-investigation.md
  nvfp4-classic-backend-collapse.md
  qwen-llm-migration.md
  qwen-nvfp4-serving-backend-comparison.md
  tts-replacement-investigation.md
  voice-pipeline-queueing.md
)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

for doc in "${DOCS[@]}"; do
  git show "${SOURCE_REF}:docs/${doc}" > "${TMPDIR}/${doc}"
done

aws s3 sync "$TMPDIR" "$S3_DEST"

echo "Uploaded ${#DOCS[@]} docs to $S3_DEST"
