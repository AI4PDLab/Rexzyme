#!/usr/bin/env bash
set -euo pipefail

# Run CLEAN inference, then keep EC-matching rows.
#
# Inputs expected in score table:
# - predicted_ec
# - target_ec

INPUT_CSV="${INPUT_CSV:-results/step02_perplexity_top25.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step03_clean_ec_match.csv}"
LEVEL="${LEVEL:-4}"

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_clean_ec_match.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --predicted_ec_column predicted_ec \
  --target_ec_column target_ec \
  --level "${LEVEL}"
