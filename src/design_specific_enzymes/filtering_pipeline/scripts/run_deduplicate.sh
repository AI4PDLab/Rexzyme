#!/usr/bin/env bash
set -euo pipefail

# Step 1: deduplicate the generated FASTA and build the candidate table used by
# all later CSV-based filter steps.
#
# Outputs:
#   OUTPUT_FASTA  deduplicated FASTA
#   OUTPUT_CSV    candidate table (candidate_id, sequence, sequence_md5, perplexity, ...)

INPUT_FASTA="${INPUT_FASTA:-results/dpo_generated.fasta}"
OUTPUT_FASTA="${OUTPUT_FASTA:-results/step01_deduplicated.fasta}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step01_deduplicated.csv}"
TARGET_EC="${TARGET_EC:-}"
REACTION="${REACTION:-}"

python src/design_specific_enzymes/filtering_pipeline/scripts/deduplicate_fasta.py \
  --input_fasta "${INPUT_FASTA}" \
  --output_fasta "${OUTPUT_FASTA}"

python src/design_specific_enzymes/filtering_pipeline/scripts/build_candidates_from_fasta.py \
  --input_fasta "${OUTPUT_FASTA}" \
  --output_csv "${OUTPUT_CSV}" \
  --target_ec "${TARGET_EC}" \
  --reaction "${REACTION}"
