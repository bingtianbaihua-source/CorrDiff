#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${BUNDLE_DIR}/../../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${BUNDLE_DIR}"
mkdir -p inputs outputs logs

# (Re)create checklist input for deterministic validation.
cat > inputs/workflow_checklist.txt <<'TXT'
stage_1_vae_pretraining
stage_2_latent_diffusion
disenmood_mode_switch
legacy_3d_path_disabled
TXT

python verify_workflow_docs.py

SUMMARY="outputs/workflow_summary.txt"
test -f "${SUMMARY}"

for key in \
  stage_1_vae_pretraining \
  stage_2_latent_diffusion \
  disenmood_mode_switch \
  legacy_3d_path_disabled
do
  grep -q "${key}" "${SUMMARY}"
done

grep -q "Stage 1: VAE pretraining" "${SUMMARY}"
grep -q "Stage 2: BranchDiffusion latent diffusion training" "${SUMMARY}"
grep -q "disenmood_mode: true" "${SUMMARY}"
grep -q "DISABLED (no fallback route)" "${SUMMARY}"

