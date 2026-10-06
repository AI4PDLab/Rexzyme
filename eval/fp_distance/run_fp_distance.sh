#!/usr/bin/env bash
# Minimum fingerprint distance from held-out reactions (0-1255) to training.
#
# Usage:
#   run_fp_distance.sh --smiles-map FILE --train-json FILE --output FILE
#
# Named flags (preferred):
#   --smiles-map FILE   held-out SMILES map
#   --train-json FILE   local JSON/JSONL or Hub repo id
#   --output FILE       output JSON
#   --cache-dir DIR     fingerprint pickle directory
#
# Environment fallbacks: TRAIN_JSON, OUT, SMILES_MAP, CACHE_DIR

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

TRAIN_JSON=${TRAIN_JSON:-AI4PD/REXzyme}
OUT=${OUT:-fingerprint_distances.json}
SMILES_MAP=${SMILES_MAP:-$REPO_ROOT/data/generated/generation_heldout/smiles_map.json}
CACHE_DIR=${CACHE_DIR:-}

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --smiles-map|--input) SMILES_MAP="$2"; shift 2 ;;
        --train-json) TRAIN_JSON="$2"; shift 2 ;;
        --output|--output-json|--out) OUT="$2"; shift 2 ;;
        --cache-dir) CACHE_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

CACHE_DIR=${CACHE_DIR:-$(dirname "$OUT")}

python "$SCRIPT_DIR/fingerprint_dataset_distance.py" \
    --smiles-map "$SMILES_MAP" \
    --train-json "$TRAIN_JSON" \
    --output-json "$OUT" \
    --cache-dir "$CACHE_DIR"
