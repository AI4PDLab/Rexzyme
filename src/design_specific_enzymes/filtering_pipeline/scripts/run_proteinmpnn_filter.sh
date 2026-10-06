#!/usr/bin/env bash
set -euo pipefail

# Template: run ProteinMPNN externally and merge scores into CSV as
# proteinmpnn_score.

INPUT_CSV="${INPUT_CSV:-results/step08_esm1v_top500.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step09_mpnn_top200.csv}"
TOP_N="${TOP_N:-200}"

python design_specific_enzymes/filtering_pipeline/scripts/filter_top_n.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --column proteinmpnn_score \
  --top_n "${TOP_N}" \
  --direction higher_is_better
