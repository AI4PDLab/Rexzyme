#!/usr/bin/env python3
"""Nearest-neighbour cosine distance from held-out embeddings to the training set.

Writes a JSON keyed by reaction SMILES. If ``--smiles-map`` is given, also writes
``embedding_distance_by_file.tsv`` with one row per ``output_file_N.fasta``
(reactions 0–1255).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMILES_MAP = (
    REPO_ROOT / "data" / "generated" / "generation_heldout" / "smiles_map.json"
)
EPS = 1e-8


def load_embeddings(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-embeddings",
        required=True,
        type=Path,
        help="Pickle of train embeddings (reaction -> ndarray)",
    )
    parser.add_argument(
        "--heldout-embeddings",
        required=True,
        type=Path,
        help="Pickle of held-out embeddings (reaction -> ndarray)",
    )
    parser.add_argument(
        "--output-json",
        "--output",
        dest="output_json",
        required=True,
        type=Path,
        help="JSON mapping held-out SMILES -> nearest training neighbour",
    )
    parser.add_argument(
        "--fingerprints-json",
        type=Path,
        default=None,
        help="Optional fingerprint_distances.json to merge min_distance",
    )
    parser.add_argument(
        "--smiles-map",
        type=Path,
        default=DEFAULT_SMILES_MAP,
        help="Held-out smiles_map.json used to write the per-file TSV",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=None,
        help="Per-file TSV (default: embedding_distance_by_file.tsv next to --output-json)",
    )
    args = parser.parse_args()

    print("Loading train embeddings…")
    train_embs_dict = load_embeddings(args.train_embeddings)
    print("Loading held-out embeddings…")
    val_embs_dict = load_embeddings(args.heldout_embeddings)

    train_reactions = list(train_embs_dict.keys())
    train_matrix = np.stack([train_embs_dict[rxn] for rxn in train_reactions], axis=0)
    train_matrix = train_matrix / (np.linalg.norm(train_matrix, axis=1, keepdims=True) + EPS)

    fp_map = {}
    if args.fingerprints_json:
        fp_list = json.loads(args.fingerprints_json.read_text(encoding="utf-8"))
        fp_map = {entry["reaction"]: entry for entry in fp_list}

    results = {}
    missing_fingerprint = 0
    for rxn, vec in tqdm(val_embs_dict.items(), desc="Finding nearest neighbours"):
        if vec.shape[0] != train_matrix.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch for {rxn}: {vec.shape[0]} vs {train_matrix.shape[1]}"
            )
        vec_norm = vec / (np.linalg.norm(vec) + EPS)
        dists = 1.0 - (train_matrix @ vec_norm)
        idx_min = int(np.argmin(dists))
        if rxn in fp_map:
            fasta_path = fp_map[rxn].get("fasta_path", "")
            min_fp_dist = fp_map[rxn].get("min_distance")
        else:
            fasta_path = ""
            min_fp_dist = None
            missing_fingerprint += 1
        results[rxn] = {
            "closest_reaction": train_reactions[idx_min],
            "min_emb_dist": float(dists[idx_min]),
            "fasta_path": fasta_path,
            "min_fingerprint_dist": min_fp_dist,
        }

    if args.fingerprints_json:
        print(
            f"{missing_fingerprint}/{len(val_embs_dict)} reactions not found "
            "in the provided fingerprints JSON"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json}")

    if args.smiles_map and args.smiles_map.is_file():
        meta = json.loads(args.smiles_map.read_text(encoding="utf-8"))
        output_tsv = args.output_tsv or args.output_json.with_name(
            "embedding_distance_by_file.tsv"
        )
        rows = []
        missing = []
        for fasta_path, reaction in meta.items():
            file_name = os.path.basename(str(fasta_path))
            if reaction not in results:
                missing.append((file_name, reaction))
                continue
            rows.append((file_name, results[reaction]["min_emb_dist"], reaction))

        if missing:
            preview = "\n".join(f"  {name}\t{rxn[:80]}" for name, rxn in missing[:20])
            raise SystemExit(
                f"{len(missing)} smiles_map reactions have no embedding distance.\n{preview}"
            )

        def file_number(row):
            try:
                return int(
                    row[0]
                    .replace("output_file_", "")
                    .replace("output_file-", "")
                    .replace(".fasta", "")
                )
            except ValueError:
                return float("inf")

        rows.sort(key=file_number)
        with output_tsv.open("w", encoding="utf-8") as handle:
            handle.write("file_name\tmin_emb_distance\tsmiles\n")
            for file_name, dist, reaction in rows:
                handle.write(f"{file_name}\t{dist}\t{reaction}\n")
        print(f"Wrote {output_tsv} ({len(rows)} reactions, including 0–1255)")


if __name__ == "__main__":
    main()
