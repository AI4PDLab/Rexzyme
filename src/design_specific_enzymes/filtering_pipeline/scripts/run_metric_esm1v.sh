#!/usr/bin/env bash
set -euo pipefail

# Run ESM1v scoring.
# Expected downstream CSV columns: sequence_md5,esm1v_score

INPUT_FASTA="${INPUT_FASTA:-results/dpo_generated.fasta}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-results/esm1v/output_file_esm1v}"
ESM1V_CMD="${ESM1V_CMD:-python3 esm1v/protein_gibbs_sampler/src/pgen/likelihood_esm.py}"
DEVICE="${DEVICE:-gpu}"

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

${ESM1V_CMD} \
  -i "${INPUT_FASTA}" \
  -o "${OUTPUT_PREFIX}" \
  --model esm1v \
  --device "${DEVICE}"

echo "ESM1v completed. Convert output to sequence_md5,esm1v_score CSV."
