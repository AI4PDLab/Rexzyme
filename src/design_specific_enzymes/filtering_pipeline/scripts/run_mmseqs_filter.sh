#!/usr/bin/env bash
set -euo pipefail

# Template: run MMseqs externally and merge identity into input CSV as
# identity_to_natural, then apply identity filter.

INPUT_CSV="${INPUT_CSV:-results/step04_plddt_gt75.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step05_mmseqs_nonidentical.csv}"
MAX_IDENTITY="${MAX_IDENTITY:-0.9999}"

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_mmseqs_identity.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --identity_column identity_to_natural \
  --max_identity "${MAX_IDENTITY}"
