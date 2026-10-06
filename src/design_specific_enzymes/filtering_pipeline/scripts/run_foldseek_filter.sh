#!/usr/bin/env bash
set -euo pipefail

# Template: run Foldseek externally and merge TM-scores into CSV as tmscore.

INPUT_CSV="${INPUT_CSV:-results/step06_iptm_top25.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step07_foldseek_top1000.csv}"
TOP_N="${TOP_N:-1000}"

python design_specific_enzymes/filtering_pipeline/scripts/filter_top_n.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --column tmscore \
  --top_n "${TOP_N}" \
  --direction higher_is_better
