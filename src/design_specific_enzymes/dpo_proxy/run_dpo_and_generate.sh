#!/usr/bin/env bash
set -euo pipefail

# Launcher for weighted DPO + generation.
# Defaults match the original DPO_pLM.py + seq_gen.py hyperparameters.
# Adjust MODEL_PATH, ACTIVITY_FILE, REACTION, PROXY_COLUMN, and MODE for your project.

MODEL_PATH="${MODEL_PATH:-/path/to/finetuned_model}"
OUTPUT_MODEL_DIR="${OUTPUT_MODEL_DIR:-results/dpo_model}"
TOKENIZER_AA="${TOKENIZER_AA:-AI4PD/REXzyme/tokenizer_aa}"
TOKENIZER_MOL="${TOKENIZER_MOL:-AI4PD/REXzyme/tokenizer_smiles}"
ACTIVITY_FILE="${ACTIVITY_FILE:-/path/to/activity.csv}"
SEQUENCE_COLUMN="${SEQUENCE_COLUMN:-Seq}"
PROXY_COLUMN="${PROXY_COLUMN:-Kcat/Km}"
REACTION="${REACTION:-SMILES_REACTION_HERE}"
MODE="${MODE:-weighted}"            # weighted is the only supported mode
OBJECTIVE="${OBJECTIVE:-maximize}"  # maximize | minimize
if [ "${MODE}" != "weighted" ]; then
  echo "ERROR: MODE=${MODE} is not supported; 'weighted' is the only DPO mode ('ranked' was removed)." >&2
  exit 1
fi
# Default matches the input expected by filtering_pipeline/scripts/run_*.sh
FASTA_OUTPUT="${FASTA_OUTPUT:-results/dpo_generated.fasta}"

# Hyperparameters (matching original DPO_pLM.py CONFIG)
BETA="${BETA:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-1e-7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
SEED="${SEED:-1998}"

# Generation (matching seq_gen.py defaults)
GENERATION_ROUNDS="${GENERATION_ROUNDS:-500}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-20}"
TOP_K="${TOP_K:-18}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"

python src/design_specific_enzymes/dpo_proxy/dpo_optimize_and_generate.py \
  --model_path "${MODEL_PATH}" \
  --output_model_dir "${OUTPUT_MODEL_DIR}" \
  --tokenizer_aa "${TOKENIZER_AA}" \
  --tokenizer_mol "${TOKENIZER_MOL}" \
  --activity_file "${ACTIVITY_FILE}" \
  --sequence_column "${SEQUENCE_COLUMN}" \
  --proxy_column "${PROXY_COLUMN}" \
  --reaction "${REACTION}" \
  --mode "${MODE}" \
  --objective "${OBJECTIVE}" \
  --beta "${BETA}" \
  --learning_rate "${LEARNING_RATE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --seed "${SEED}" \
  --generate_after_dpo \
  --generation_reaction "${REACTION}" \
  --fasta_output "${FASTA_OUTPUT}" \
  --generation_rounds "${GENERATION_ROUNDS}" \
  --num_return_sequences "${NUM_RETURN_SEQUENCES}" \
  --top_k "${TOP_K}" \
  --top_p "${TOP_P}" \
  --repetition_penalty "${REPETITION_PENALTY}"
