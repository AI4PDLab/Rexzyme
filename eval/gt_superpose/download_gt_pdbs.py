#!/usr/bin/env python3
"""Download ground-truth structures for held-out reactions.

For every reaction in the held-out SMILES map, create ``gt_output_file_<N>``
and download up to 5 structures (best first):

  * best_criterion == PDB      -> experimental structures from RCSB (.pdb, then .cif)
  * best_criterion == score:*  -> AlphaFold models for the curated UniProt accessions
  * best_criterion == none     -> empty folder

Reaction numbers come from ``output_file_N.fasta`` keys in the SMILES map
(0 through 1255 for the paper held-out set). Rhea / EC / UniProt accessions
come from ``rhea_uniprot_sequence_check_best.tsv``.

Resumable: a folder with a DONE marker is skipped; existing non-empty files
are not re-downloaded. Only the standard library is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMILES_MAP = (
    REPO_ROOT / "data" / "generated" / "generation_heldout" / "smiles_map.json"
)
DEFAULT_TSV = (
    REPO_ROOT
    / "data"
    / "extended_reaction_information"
    / "generation_heldout"
    / "rhea_uniprot_sequence_check_best.tsv"
)

UA = {"User-Agent": "rexzyme-gt-pdb-downloader/1.0"}
UNIPROT = "https://rest.uniprot.org/uniprotkb/search"
RCSB_PDB = "https://files.rcsb.org/download/{}.pdb"
RCSB_CIF = "https://files.rcsb.org/download/{}.cif"
AF_API = "https://alphafold.ebi.ac.uk/api/prediction/{}"

MAX_STRUCT = 5


def http_get(url: str, retries: int = 4, timeout: int = 60, binary: bool = False):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as handle:
                data = handle.read()
                return data if binary else data.decode("utf-8", "replace")
        except Exception as error:
            last = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed ({last}): {url}")


def download_file(url: str, dest: str, retries: int = 4, timeout: int = 120) -> bool:
    """Download url -> dest. Returns True on success. Skips if dest is non-empty."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as handle:
                data = handle.read()
            if not data:
                return False
            tmp = dest + ".part"
            with open(tmp, "wb") as handle:
                handle.write(data)
            os.replace(tmp, dest)
            return True
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return False
            last = error
            time.sleep(1.5 * (attempt + 1))
        except Exception as error:
            last = error
            time.sleep(1.5 * (attempt + 1))
    print(f"    ! failed {url}: {last}", file=sys.stderr)
    return False


def download_rcsb_structure(pdb_id: str, folder: str) -> tuple[str, str] | None:
    """Download a PDB, falling back to mmCIF for large assemblies."""
    pdb_dest = os.path.join(folder, f"{pdb_id}.pdb")
    if download_file(RCSB_PDB.format(pdb_id), pdb_dest):
        return f"{pdb_id}.pdb", RCSB_PDB.format(pdb_id)
    cif_dest = os.path.join(folder, f"{pdb_id}.cif")
    if download_file(RCSB_CIF.format(pdb_id), cif_dest):
        return f"{pdb_id}.cif", RCSB_CIF.format(pdb_id)
    return None


def pdb_ids_from_query(base_query: str, sleep: float) -> list[str]:
    """Return ordered unique PDB IDs for entries matching base_query with a PDB xref."""
    query = urllib.parse.urlencode(
        {
            "query": f"{base_query} AND (database:pdb)",
            "fields": "accession,xref_pdb",
            "format": "tsv",
            "size": 50,
        }
    )
    text = http_get(f"{UNIPROT}?{query}")
    time.sleep(sleep)
    ids: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        for pdb_id in cols[1].split(";"):
            pdb_id = pdb_id.strip()
            if pdb_id and pdb_id not in seen:
                seen.add(pdb_id)
                ids.append(pdb_id)
    return ids


def pdb_ids_from_accessions(accessions: list[str], sleep: float) -> list[str]:
    """Collect PDB IDs in curated accession order (first accession first)."""
    ids: list[str] = []
    seen: set[str] = set()
    for accession in accessions:
        for pdb_id in pdb_ids_from_query(f"accession:{accession}", sleep):
            if pdb_id not in seen:
                seen.add(pdb_id)
                ids.append(pdb_id)
    return ids


def get_alphafold_url(accession: str) -> str | None:
    try:
        text = http_get(AF_API.format(accession), retries=2, timeout=30)
        data = json.loads(text)
        if data and isinstance(data, list):
            return data[0].get("pdbUrl")
    except Exception:
        return None
    return None


def parse_reaction_number(path_or_name: str) -> int | None:
    match = re.search(r"output_file[_-](\d+)\.fasta", path_or_name)
    return int(match.group(1)) if match else None


def load_smiles_map(path: Path) -> list[tuple[int, str]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for key, smiles in meta.items():
        number = parse_reaction_number(key)
        if number is not None:
            items.append((number, smiles))
    items.sort()
    return items


def load_uniprot_tsv(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            rows[parts[index["Reaction"]]] = {
                "rhea": parts[index["RheaID"]].strip(),
                "ec": parts[index["EC"]].strip(),
                "method": parts[index["Method"]].strip(),
                "has_seq": parts[index["Has_sequence"]].strip(),
                "criterion": parts[index["best_criterion"]].strip(),
                "best_acc": [
                    acc
                    for acc in parts[index["best_accessions"]].strip().split(",")
                    if acc
                ],
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smiles-map",
        type=Path,
        default=DEFAULT_SMILES_MAP,
        help="JSON map of output_file_N.fasta -> reaction SMILES",
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=DEFAULT_TSV,
        help="UniProt best-accession TSV for held-out reactions",
    )
    parser.add_argument("--outdir", "--output", dest="outdir", type=Path, default=Path("gt_structures"),
                        help="Directory for gt_output_file_<N> folders")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First reaction index to process (inclusive). Default: all.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last reaction index to process (inclusive). Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a DONE marker already exists.",
    )
    args = parser.parse_args()

    if not args.smiles_map.is_file():
        raise SystemExit(f"SMILES map not found: {args.smiles_map}")
    if not args.tsv.is_file():
        raise SystemExit(f"UniProt TSV not found: {args.tsv}")

    items = load_smiles_map(args.smiles_map)
    if args.start is not None:
        items = [item for item in items if item[0] >= args.start]
    if args.end is not None:
        items = [item for item in items if item[0] <= args.end]
    if not items:
        raise SystemExit("No held-out reactions in the requested range")

    tsv = load_uniprot_tsv(args.tsv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    n_done = n_pdb = n_af = n_empty = n_skip = 0
    total = len(items)

    for i, (num, smiles) in enumerate(items, 1):
        folder = os.path.join(args.outdir, f"gt_output_file_{num}")
        os.makedirs(folder, exist_ok=True)
        marker = os.path.join(folder, "DONE")
        if os.path.exists(marker) and not args.force:
            n_skip += 1
            continue

        info = tsv.get(smiles)
        manifest: list[tuple[str, str, str]] = []

        if not info or info["has_seq"] != "Yes" or info["criterion"] == "none":
            n_empty += 1
        else:
            if info["method"] == "rhea":
                base = f"rhea:{info['rhea']}"
            else:
                base = f"EC {info['ec']}"

            if info["criterion"] == "PDB":
                # Prefer PDB IDs linked to the curated accessions (so reaction
                # 1255 keeps 2FSS / 6T8Y ahead of the generic UniProt order).
                pdb_ids = pdb_ids_from_accessions(info["best_acc"][:20], args.sleep)
                for pdb_id in pdb_ids_from_query(base, args.sleep):
                    if pdb_id not in pdb_ids:
                        pdb_ids.append(pdb_id)

                got = 0
                for pdb_id in pdb_ids:
                    if got >= MAX_STRUCT:
                        break
                    downloaded = download_rcsb_structure(pdb_id, folder)
                    if downloaded:
                        filename, url = downloaded
                        manifest.append((filename, url, "experimental_PDB"))
                        got += 1
                        time.sleep(args.sleep)
                if got:
                    n_pdb += 1
                else:
                    n_empty += 1

            elif str(info["criterion"]).startswith("score:"):
                got = 0
                for accession in info["best_acc"]:
                    if got >= MAX_STRUCT:
                        break
                    url = get_alphafold_url(str(accession))
                    time.sleep(args.sleep)
                    if not url:
                        continue
                    dest = os.path.join(folder, f"AF-{accession}.pdb")
                    if download_file(url, dest):
                        manifest.append(
                            (
                                f"AF-{accession}.pdb",
                                url,
                                "alphafold_" + str(info["criterion"]),
                            )
                        )
                        got += 1
                        time.sleep(args.sleep)
                if got:
                    n_af += 1
                else:
                    n_empty += 1

        with open(os.path.join(folder, "manifest.tsv"), "w", encoding="utf-8") as handle:
            handle.write("filename\tsource_url\ttype\n")
            for row in manifest:
                handle.write("\t".join(row) + "\n")
        with open(
            os.path.join(folder, "reaction_info.tsv"), "w", encoding="utf-8"
        ) as handle:
            handle.write(
                "Reaction\tRheaID\tEC\tMethod\tbest_criterion\tn_downloaded\n"
            )
            rhea = info["rhea"] if info else "NA"
            ec = info["ec"] if info else "NA"
            method = info["method"] if info else "none"
            criterion = info["criterion"] if info else "none"
            handle.write(
                f"{smiles}\t{rhea}\t{ec}\t{method}\t{criterion}\t{len(manifest)}\n"
            )

        open(marker, "w", encoding="utf-8").close()
        n_done += 1

        if i % 25 == 0:
            print(
                f"[{i}/{total}] processed={n_done} pdb={n_pdb} af={n_af} "
                f"empty={n_empty} skip={n_skip}",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"Done. total={total} processed={n_done} skipped={n_skip} | "
        f"folders with PDB={n_pdb}, AlphaFold={n_af}, empty={n_empty}"
    )


if __name__ == "__main__":
    main()
