#!/usr/bin/env bash
set -euo pipefail

# Keep the best fraction by perplexity (lower is better).

INPUT_CSV="${INPUT_CSV:-results/step01_deduplicated.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-results/step02_perplexity_top25.csv}"
FRACTION="${FRACTION:-0.25}"

python src/design_specific_enzymes/filtering_pipeline/scripts/filter_top_fraction.py \
  --input_csv "${INPUT_CSV}" \
  --output_csv "${OUTPUT_CSV}" \
  --column perplexity \
  --fraction "${FRACTION}" \
  --direction lower_is_better
