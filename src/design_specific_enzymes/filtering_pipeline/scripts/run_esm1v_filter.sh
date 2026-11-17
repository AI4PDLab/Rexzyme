#!/usr/bin/env bash
set -euo pipefail

# Template: run ESM1v externally and merge scores into CSV as esm1v_score.

INPUT_CSV="${INPUT_CSV:-results/step07_foldseek_top1000.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step08_esm1v_top500.csv}"
TOP_N="${TOP_N:-500}"

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_n.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --column esm1v_score \
  --top_n "${TOP_N}" \
  --direction higher_is_better
