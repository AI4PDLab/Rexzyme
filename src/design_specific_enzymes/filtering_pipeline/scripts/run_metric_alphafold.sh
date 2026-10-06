#!/usr/bin/env bash
set -euo pipefail

# Template: run AlphaFold and export a CSV with:
# sequence_md5,plddt,iptm
#
# This script executes the AlphaFold command; parsing to CSV is left to the
# environment-specific post-processing step.

INPUT_FASTA="${INPUT_FASTA:-results/dpo_generated.fasta}"
OUTPUT_DIR="${OUTPUT_DIR:-results/alphafold}"
ALPHAFOLD_CMD="${ALPHAFOLD_CMD:-python run_alphafold.py}"
OLIGOMER_STATE="${OLIGOMER_STATE:-dimer}"

mkdir -p "${OUTPUT_DIR}"
echo "Running AlphaFold (${OLIGOMER_STATE}) on ${INPUT_FASTA}"
echo "Command: ${ALPHAFOLD_CMD}"

# Example invocation; replace flags with local AlphaFold setup.
${ALPHAFOLD_CMD} \
  --input_fasta "${INPUT_FASTA}" \
  --output_dir "${OUTPUT_DIR}" \
  --multimer_mode "${OLIGOMER_STATE}"

echo "AlphaFold finished. Export sequence_md5,plddt,iptm to a CSV for merge."
