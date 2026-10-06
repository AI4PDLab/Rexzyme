# Held-out fingerprint distance

Minimum Tanimoto distance from each held-out reaction (0–1255) to the closest
training reaction. Each SMILES is encoded as the concatenation of an RDKit
structural reaction fingerprint and a difference fingerprint.

Training reactions are read from `training_dataset.json` on
[AI4PD/REXzyme](https://huggingface.co/AI4PD/REXzyme) unless you pass a local file.
That file is JSONL: `translation.Reaction` is the reaction SMILES.

```bash
bash eval/fp_distance/run_fp_distance.sh \
  --smiles-map data/generated/generation_heldout/smiles_map.json \
  --train-json AI4PD/REXzyme \
  --output fingerprint_distances.json
```

Or the Python script directly:

```bash
python eval/fp_distance/fingerprint_dataset_distance.py \
  --smiles-map data/generated/generation_heldout/smiles_map.json \
  --train-json AI4PD/REXzyme \
  --output fingerprint_distances.json
```

Requires RDKit (`pip install -r requirements-eval.txt`). Fingerprint pickles
are cached next to the output JSON (or `--cache-dir`) so a rerun does not
recompute the training set.

The paper TM-score versus fingerprint-distance table is
[`../tmscore_plots/tmscore_vs_fp_distance.tsv`](../tmscore_plots/tmscore_vs_fp_distance.tsv).
