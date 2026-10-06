#!/usr/bin/env bash
set -euo pipefail

# Deduplicate FASTA sequences.

INPUT_FASTA="${INPUT_FASTA:-results/generated.fasta}"
OUTPUT_FASTA="${OUTPUT_FASTA:-results/step01_deduplicated.fasta}"

python design_specific_enzymes/filtering_pipeline/scripts/deduplicate_fasta.py \
  --input_fasta "${INPUT_FASTA}" \
  --output_fasta "${OUTPUT_FASTA}"
