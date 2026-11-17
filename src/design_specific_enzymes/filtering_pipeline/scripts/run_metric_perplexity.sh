#!/usr/bin/env bash
set -euo pipefail

# Build the base candidate table from seq_gen FASTA.

INPUT_FASTA="${INPUT_FASTA:-results/dpo_generated.fasta}"
OUTPUT_CSV="${OUTPUT_CSV:-results/filtering/candidates_base.csv}"
TARGET_EC="${TARGET_EC:-}"
REACTION="${REACTION:-}"

python src/design_specific_enzymes/filtering_pipeline/scripts/build_candidates_from_fasta.py \
  --input_fasta "${INPUT_FASTA}" \
  --output_csv "${OUTPUT_CSV}" \
  --target_ec "${TARGET_EC}" \
  --reaction "${REACTION}"
