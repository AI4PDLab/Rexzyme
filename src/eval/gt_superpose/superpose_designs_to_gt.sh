#!/usr/bin/env bash
# Superpose each held-out design against its ground-truth structures with
# Foldseek (TM-align). Processes reactions 0-1255 by default.
#
#   designs : $DESIGNS/output_file_<N>/pdbs/*.pdb
#             (also accepts $DESIGNS/output_file_<N>/*.pdb
#              and the older output_file-<N> folder names)
#   ground  : $GT/gt_output_file_<N>/*.{pdb,cif}
#   output  : $OUT/output_file_<N>/tmscores.tsv
#             $OUT/all_tmscores.tsv
#             $OUT/all_tmscores_unfiltered.tsv
#
# A pair is kept when:
#   tcov >= MIN_TCOV  (default 0.5)
#   tlen >= MIN_TLEN  (default 50)
#   qcov >= MIN_QCOV  (default 0.5)
#
# Usage:
#   superpose_designs_to_gt.sh --designs DIR --gt DIR --out DIR [--start N] [--end N]
#   superpose_designs_to_gt.sh [START] [END] [MIN_TCOV]
#
# Named flags (preferred):
#   --designs DIR    Root of per-reaction design PDBs
#   --gt DIR         Root of gt_output_file_* folders
#   --out DIR        Output directory
#   --start N        First reaction index (default: 0)
#   --end N          Last reaction index (default: 1255)
#   --min-tcov F     Minimum target coverage (default: 0.5)
#   --foldseek EXE   Foldseek executable
#
# Environment fallbacks: FOLDSEEK, DESIGNS, GT, OUT

set -uo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

FS=${FOLDSEEK:-foldseek}
DESROOT=${DESIGNS:-./designs}
GTROOT=${GT:-./gt_structures}
OUT=${OUT:-./superposition_ground_truth_designs}
START=""
END=""
MIN_TCOV=""
MIN_TLEN=${MIN_TLEN:-50}
MIN_QCOV=${MIN_QCOV:-0.5}

usage() {
    sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --designs|--input) DESROOT="$2"; shift 2 ;;
        --gt) GTROOT="$2"; shift 2 ;;
        --out|--output) OUT="$2"; shift 2 ;;
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --min-tcov) MIN_TCOV="$2"; shift 2 ;;
        --foldseek) FS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *) break ;;
    esac
done

START=${START:-${1:-0}}
END=${END:-${2:-1255}}
MIN_TCOV=${MIN_TCOV:-${3:-0.5}}

if ! command -v "$FS" >/dev/null 2>&1; then
    echo "Foldseek not found: $FS (set FOLDSEEK or add foldseek to PATH)" >&2
    exit 1
fi

mkdir -p "$OUT"
MASTER=$OUT/all_tmscores.tsv
MASTER_ALL=$OUT/all_tmscores_unfiltered.tsv

FMT="query,target,alntmscore,qtmscore,ttmscore,lddt,rmsd,prob,fident,nident,alnlen,qlen,tlen,qstart,qend,tstart,tend,qcov,tcov,mismatch,gapopen,evalue,bits"
HEADER=$'design\tground_truth\talntmscore\tqtmscore\tttmscore\tlddt\trmsd\tprob\tfident\tnident\talnlen\tqlen\ttlen\tqstart\tqend\ttstart\ttend\tqcov\ttcov\tmismatch\tgapopen\tevalue\tbits'

if [ ! -s "$MASTER" ]; then
    printf 'reaction\t%s\n' "$HEADER" > "$MASTER"
fi
if [ ! -s "$MASTER_ALL" ]; then
    printf 'reaction\t%s\n' "$HEADER" > "$MASTER_ALL"
fi

collect_designs() {
    local n=$1
    local files=()
    files=("$DESROOT/output_file_$n/pdbs/"*.pdb)
    if [ ${#files[@]} -eq 0 ]; then
        files=("$DESROOT/output_file_$n/"*.pdb)
    fi
    if [ ${#files[@]} -eq 0 ]; then
        files=("$DESROOT/output_file-$n/pdbs/"*.pdb)
    fi
    if [ ${#files[@]} -eq 0 ]; then
        files=("$DESROOT/output_file-$n/"*.pdb)
    fi
    if [ ${#files[@]} -gt 0 ]; then
        printf '%s\n' "${files[@]}"
    fi
}

processed=0
skipped=0
kept_total=0
dropped_total=0

for N in $(seq "$START" "$END"); do
    mapfile -t despdbs < <(collect_designs "$N")
    GT=$GTROOT/gt_output_file_$N
    gtpdbs=("$GT"/*.pdb "$GT"/*.cif)

    if [ ${#despdbs[@]} -eq 0 ] || [ -z "${despdbs[0]:-}" ] || [ ${#gtpdbs[@]} -eq 0 ]; then
        skipped=$((skipped + 1))
        continue
    fi

    odir=$OUT/output_file_$N
    mkdir -p "$odir"
    tmp=$odir/_tmp
    rm -rf "$tmp"
    mkdir -p "$tmp"

    if ! "$FS" createdb "${gtpdbs[@]}" "$tmp/gtDB" > "$tmp/createdb.log" 2>&1; then
        echo "  ! foldseek createdb failed for output_file-$N (see $tmp/createdb.log)" >&2
        skipped=$((skipped + 1))
        continue
    fi

    if ! "$FS" easy-search "${despdbs[@]}" "$tmp/gtDB" "$odir/_raw.tsv" "$tmp/fstmp" \
        --alignment-type 1 --exhaustive-search 1 -e inf --tmscore-threshold 0.0 \
        --max-seqs 10000 --format-output "$FMT" -v 1 > "$tmp/search.log" 2>&1; then
        echo "  ! foldseek search failed for output_file-$N (see $tmp/search.log)" >&2
        skipped=$((skipped + 1))
        continue
    fi

    if [ ! -s "$odir/_raw.tsv" ]; then
        echo "  ! no output for output_file-$N (see $tmp)" >&2
        skipped=$((skipped + 1))
        continue
    fi

    # Raw columns: alnlen=11 qlen=12 tlen=13 qcov=18 tcov=19
    awk -v tcov=19 -v tlen=13 -v qcov=18 \
        -v tct="$MIN_TCOV" -v tlt="$MIN_TLEN" -v qct="$MIN_QCOV" \
        'BEGIN{OFS="\t"} ($tcov+0)>=tct && ($tlen+0)>=tlt && ($qcov+0)>=qct' \
        "$odir/_raw.tsv" > "$odir/_kept.tsv"
    awk -v tcov=19 -v tlen=13 -v qcov=18 \
        -v tct="$MIN_TCOV" -v tlt="$MIN_TLEN" -v qct="$MIN_QCOV" \
        'BEGIN{OFS="\t"} !(($tcov+0)>=tct && ($tlen+0)>=tlt && ($qcov+0)>=qct)' \
        "$odir/_raw.tsv" > "$odir/_dropped.tsv"

    nk=$(wc -l < "$odir/_kept.tsv")
    nd=$(wc -l < "$odir/_dropped.tsv")
    kept_total=$((kept_total + nk))
    dropped_total=$((dropped_total + nd))

    { printf '%s\n' "$HEADER"; cat "$odir/_kept.tsv"; } > "$odir/tmscores.tsv"
    { printf '%s\n' "$HEADER"; cat "$odir/_raw.tsv"; } > "$odir/tmscores_all.tsv"

    awk -v r="output_file_$N" 'BEGIN{OFS="\t"}{print r,$0}' "$odir/_kept.tsv" >> "$MASTER"
    awk -v r="output_file_$N" 'BEGIN{OFS="\t"}{print r,$0}' "$odir/_raw.tsv" >> "$MASTER_ALL"

    rm -rf "$tmp" "$odir/_raw.tsv" "$odir/_kept.tsv" "$odir/_dropped.tsv"
    processed=$((processed + 1))
    if [ $((processed % 25)) -eq 0 ]; then
        echo "[processed=$processed skipped=$skipped kept=$kept_total dropped=$dropped_total] last N=$N" >&2
    fi
done

echo "Done. processed=$processed skipped=$skipped"
echo "Kept (tcov>=$MIN_TCOV, tlen>=$MIN_TLEN, qcov>=$MIN_QCOV)=$kept_total  Dropped=$dropped_total"
echo "Filtered aggregate:   $MASTER"
echo "Unfiltered aggregate: $MASTER_ALL"
echo "Re-apply the same filter later with: python $SCRIPT_DIR/apply_reliability_filter.py --root $OUT"
