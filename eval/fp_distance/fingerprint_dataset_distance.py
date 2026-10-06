#!/usr/bin/env python3
"""Minimum Tanimoto distance from held-out reactions to the training set.

Each reaction SMILES is encoded as the concatenation of an RDKit structural
reaction fingerprint and a difference fingerprint. The reported distance is
``1 - max Tanimoto similarity`` to any training reaction.

Held-out reactions default to
``data/generated/generation_heldout/smiles_map.json`` (reactions 0–1255).
Training reactions default to ``training_dataset.json`` in
https://huggingface.co/AI4PD/REXzyme (or a local JSON/JSONL path).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hf_assets import DEFAULT_HF_REPO, resolve_train_json

from rdkit import DataStructs
from rdkit.Chem import rdChemReactions
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMILES_MAP = (
    REPO_ROOT / "data" / "generated" / "generation_heldout" / "smiles_map.json"
)


def load_heldout_reactions(smiles_map_path: Path) -> dict[str, list[str]]:
    """Return reaction SMILES -> list of fasta names (``output_file_N.fasta``)."""
    meta = json.loads(smiles_map_path.read_text(encoding="utf-8"))
    reac2paths: dict[str, list[str]] = {}
    for fasta_name, smiles in meta.items():
        reac2paths.setdefault(smiles, []).append(os.path.basename(str(fasta_name)))
    return reac2paths


def load_training_reactions(train_json_path: Path) -> set[str]:
    reactions: set[str] = set()
    with train_json_path.open(encoding="utf-8") as handle:
        first = handle.readline()
        stripped = first.lstrip()
        if stripped.startswith("["):
            handle.seek(0)
            for entry in json.load(handle):
                reactions.add(entry["translation"]["Reaction"])
            return reactions

        def add_line(line: str) -> None:
            line = line.strip()
            if not line:
                return
            entry = json.loads(line)
            reactions.add(entry["translation"]["Reaction"])

        add_line(first)
        for line in handle:
            add_line(line)
    return reactions


def sparse_to_explicit(sparse_fp, n_bits: int = 2048) -> ExplicitBitVect:
    ebv = DataStructs.ExplicitBitVect(n_bits)
    for bit_id in sparse_fp.GetNonzeroElements().keys():
        if bit_id < n_bits:
            ebv.SetBit(bit_id)
    return ebv


def concatenate_fingerprints(fp1: ExplicitBitVect, fp2: ExplicitBitVect) -> ExplicitBitVect:
    len1 = fp1.GetNumBits()
    len2 = fp2.GetNumBits()
    combined = ExplicitBitVect(len1 + len2)
    for i in range(len1):
        if fp1.GetBit(i):
            combined.SetBit(i)
    for i in range(len2):
        if fp2.GetBit(i):
            combined.SetBit(len1 + i)
    return combined


def fp_concat_from_reaction(rxn_smiles: str, radius: int = 2, fp_size: int = 2048):
    try:
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        if rxn is None:
            raise ValueError(f"Could not parse reaction SMILES: {rxn_smiles}")

        struct_params = rdChemReactions.ReactionFingerprintParams()
        struct_params.fpSize = fp_size
        struct_params.radius = radius
        struct_params.includeAgents = False
        struct_params.useBondOrder = True
        struct_params.atomInvariants = 0
        struct_params.countSimulation = False

        diff_params = rdChemReactions.ReactionFingerprintParams()
        diff_params.fpSize = fp_size
        diff_params.radius = radius
        diff_params.includeAgents = False
        diff_params.useBondOrder = True
        diff_params.atomInvariants = 0
        diff_params.bitRatioProd = 1.0
        diff_params.bitRatioReact = -1.0
        diff_params.countSimulation = False

        diff_fp_sparse = rdChemReactions.CreateDifferenceFingerprintForReaction(
            rxn, diff_params
        )
        diff_fp = sparse_to_explicit(diff_fp_sparse, n_bits=fp_size)
        struct_fp = rdChemReactions.CreateStructuralFingerprintForReaction(
            rxn, struct_params
        )
        return concatenate_fingerprints(struct_fp, diff_fp)
    except Exception as error:
        print(f"Error parsing reaction: {rxn_smiles}\n{error}")
        return None


def load_or_compute_fps(
    smiles_list: list[str],
    cache_path: Path,
    label: str,
) -> dict:
    if cache_path.is_file():
        print(f"Reloading {label} fingerprints from {cache_path}")
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    print(f"Computing fingerprints for {label} reactions…")
    fps = {
        rxn: fp
        for rxn in smiles_list
        if (fp := fp_concat_from_reaction(rxn)) is not None
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {label} fingerprints to {cache_path}")
    with cache_path.open("wb") as handle:
        pickle.dump(fps, handle)
    return fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smiles-map",
        "--input",
        dest="smiles_map",
        type=Path,
        default=DEFAULT_SMILES_MAP,
        help="JSON map of output_file_N.fasta -> held-out reaction SMILES",
    )
    parser.add_argument(
        "--train-json",
        "-t",
        default=DEFAULT_HF_REPO,
        help=(
            "Local training JSON/JSONL, or Hugging Face repo id "
            f"(default: {DEFAULT_HF_REPO}, file training_dataset.json)"
        ),
    )
    parser.add_argument(
        "--output-json",
        "--output",
        "-o",
        dest="output_json",
        type=Path,
        required=True,
        help="Where to write the per-reaction distance JSON",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for fingerprint pickles (default: next to --output-json)",
    )
    args = parser.parse_args()

    if not args.smiles_map.is_file():
        raise SystemExit(f"SMILES map not found: {args.smiles_map}")
    train_json = resolve_train_json(args.train_json)

    cache_dir = args.cache_dir or args.output_json.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading held-out reactions…")
    inf_reac2paths = load_heldout_reactions(args.smiles_map)
    inf_smiles = list(inf_reac2paths.keys())
    print(f"Unique held-out reactions: {len(inf_smiles)}")

    train_cache = train_json.with_name("train_fingerprints.pkl")
    if not train_cache.is_file():
        train_cache = cache_dir / "train_fingerprints.pkl"
    if train_cache.is_file():
        train_fps = load_or_compute_fps([], train_cache, "training")
    else:
        print("Loading training reactions…")
        train_smiles = list(load_training_reactions(train_json))
        print(f"Unique training reactions: {len(train_smiles)}")
        train_fps = load_or_compute_fps(train_smiles, train_cache, "training")

    inf_fps = load_or_compute_fps(
        inf_smiles, cache_dir / "heldout_fingerprints.pkl", "held-out"
    )

    print("Calculating minimum distances…")
    train_rxn_list = list(train_fps.keys())
    train_fp_list = [train_fps[rxn] for rxn in train_rxn_list]

    results = []
    for rxn, fp_i in inf_fps.items():
        min_dist = None
        closest_train_rxn = None
        if train_fp_list:
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, train_fp_list)
            best_idx = max(range(len(sims)), key=sims.__getitem__)
            min_dist = 1.0 - sims[best_idx]
            closest_train_rxn = train_rxn_list[best_idx]
            if min_dist == 0.0:
                print(
                    "[Match] Held-out reaction matches training exactly:\n"
                    f"  Held-out: {rxn}\n"
                    f"  Training: {closest_train_rxn}",
                    flush=True,
                )

        for fasta_path in inf_reac2paths[rxn]:
            results.append(
                {
                    "reaction": rxn,
                    "fasta_path": fasta_path,
                    "min_distance": min_dist,
                    "closest_train_reaction": closest_train_rxn,
                }
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {args.output_json}")
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Done. {len(results)} held-out entries.")


if __name__ == "__main__":
    main()
