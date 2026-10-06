#!/usr/bin/env bash
# Generate enzyme sequences with Rexzyme (paper sampling settings).
#
#   INPUT_FILE            reaction SMILES, one per line (default: data/generation_most.txt)
#   OUTPUT_FOLDER         FASTA output directory (default: generation_output)
#   MODEL_PATH            checkpoint (default: AI4PD/REXzyme/checkpoint-90000)
#   TOKENIZER_AA          amino-acid tokenizer (default: AI4PD/REXzyme/tokenizer_aa)
#   TOKENIZER_MOL         SMILES tokenizer (default: AI4PD/REXzyme/tokenizer_smiles)
#   TOP_K                 top-k sampling (default: 100)
#   TOP_P                 nucleus sampling (default: 1.0)
#   REPETITION_PENALTY    generate() repetition penalty (default: 1.0)
#   NUM_RETURN_SEQUENCES  sequences per reaction (default: 25)
#   SEED                  random seed (default: 0)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

INPUT_FILE=${INPUT_FILE:-$REPO_ROOT/data/generation_most.txt}
OUTPUT_FOLDER=${OUTPUT_FOLDER:-generation_output}
MODEL_PATH=${MODEL_PATH:-AI4PD/REXzyme/checkpoint-90000}
TOKENIZER_AA=${TOKENIZER_AA:-AI4PD/REXzyme/tokenizer_aa}
TOKENIZER_MOL=${TOKENIZER_MOL:-AI4PD/REXzyme/tokenizer_smiles}
TOP_K=${TOP_K:-100}
TOP_P=${TOP_P:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
NUM_RETURN_SEQUENCES=${NUM_RETURN_SEQUENCES:-25}
SEED=${SEED:-0}

(cd "$REPO_ROOT" && python -m src.inference \
    --input_file "$INPUT_FILE" \
    --model_path "$MODEL_PATH" \
    --tokenizer_aa "$TOKENIZER_AA" \
    --tokenizer_mol "$TOKENIZER_MOL" \
    --output_folder "$OUTPUT_FOLDER" \
    --top_k "$TOP_K" \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --num_return_sequences "$NUM_RETURN_SEQUENCES" \
    --seed "$SEED")
