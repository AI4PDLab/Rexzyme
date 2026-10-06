#!/usr/bin/env bash
# Search held-out designs against the training sequences with MMseqs.
#
#   query   : data/generated/generation_heldout/all_sequences_with_filenames.fasta
#             (31,400 sequences; reactions 0-1255, 25 designs each)
#   target  : training FASTA (build from Hugging Face training JSON with
#             eval/train_identity/mmseq_file_preparation.py --json_file AI4PD/REXzyme)
#   results : $OUT/designs_vs_train.m8
#             $OUT/designs_vs_train.best.tsv   (best hit per design)
#
# Usage:
#   run_mmseqs_designs_vs_train.sh --query FILE --target FILE --output DIR
#
# Named flags (preferred):
#   --query FILE     query FASTA (default: repo held-out FASTA)
#   --target FILE    training FASTA (required)
#   --output DIR     output directory (default: ./designs_vs_train)
#   --mmseqs EXE     mmseqs executable
#   --threads N      --threads (default: 12)
#
# Environment fallbacks: QUERY, TARGET, OUT, MMSEQS, THREADS, SENS, MAX_SEQS

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

MMSEQS=${MMSEQS:-mmseqs}
QUERY=${QUERY:-$REPO_ROOT/data/generated/generation_heldout/all_sequences_with_filenames.fasta}
TARGET=${TARGET:-}
OUT=${OUT:-./designs_vs_train}
THREADS=${THREADS:-12}
SENS=${SENS:-5.7}
MAX_SEQS=${MAX_SEQS:-300}

usage() {
    sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --query|--input) QUERY="$2"; shift 2 ;;
        --target) TARGET="$2"; shift 2 ;;
        --out|--output) OUT="$2"; shift 2 ;;
        --mmseqs) MMSEQS="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        --sens) SENS="$2"; shift 2 ;;
        --max-seqs) MAX_SEQS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$TARGET" ]; then
    echo "Set --target (or TARGET) to the training-set FASTA" >&2
    usage >&2
    exit 1
fi

if ! command -v "$MMSEQS" >/dev/null 2>&1; then
    echo "MMseqs not found: $MMSEQS (set MMSEQS or add mmseqs to PATH)" >&2
    exit 1
fi
if [ ! -s "$QUERY" ]; then
    echo "Query FASTA not found: $QUERY" >&2
    exit 1
fi
if [ ! -s "$TARGET" ]; then
    echo "Target FASTA not found: $TARGET" >&2
    exit 1
fi

mkdir -p "$OUT"
TMP=$OUT/mmseqs_tmp
mkdir -p "$TMP"

M8=$OUT/designs_vs_train.m8
BEST=$OUT/designs_vs_train.best.tsv
FMT="query,target,pident,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen,qcov,tcov"

n_query=$(grep -c '^>' "$QUERY" || true)
echo "[$(date +%T)] searching $n_query designs against $TARGET"

"$MMSEQS" easy-search "$QUERY" "$TARGET" "$M8" "$TMP" \
    --format-output "$FMT" \
    -s "$SENS" --max-seqs "$MAX_SEQS" --threads "$THREADS" \
    --split-memory-limit "${SPLIT_MEMORY_LIMIT:-20G}" \
    --remove-tmp-files 1 -v 3

echo "[$(date +%T)] search done, hits: $(wc -l < "$M8")"

# Best hit per design (highest bit score). Columns: query=1 pident=3 evalue=12 bits=13 qcov=16 tcov=17
printf 'design\tbest_target\tpident\tfident\talnlen\tevalue\tbits\tqcov\ttcov\n' > "$BEST"
sort -k1,1 -k13,13gr "$M8" | awk -F'\t' '
    !seen[$1]++ { print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$12"\t"$13"\t"$16"\t"$17 }
' >> "$BEST"

n_best=$(($(wc -l < "$BEST") - 1))
echo "[$(date +%T)] best-hit summary rows: $n_best / $n_query designs"
echo "Done. Results: $M8 and $BEST"
