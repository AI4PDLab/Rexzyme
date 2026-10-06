#!/usr/bin/env python3
"""Extract Rexzyme encoder embeddings for training and held-out reactions.

Defaults match the paper analysis: encoder embeddings (``--layer-index 0``),
attention-weighted pooling on the training set, and mean/max/std pooling on
the held-out set.

Held-out reactions can be a SMILES map
(``data/generated/generation_heldout/smiles_map.json``), a one-SMILES-per-line
text file (``data/generation_heldout.txt``), or a training-style JSON/JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from hf_assets import DEFAULT_HF_REPO, DEFAULT_TOKENIZER_MOL, resolve_train_json

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HELDOUT = (
    REPO_ROOT / "data" / "generated" / "generation_heldout" / "smiles_map.json"
)


class ReactionDataset(Dataset):
    def __init__(self, reactions):
        self.reactions = list(reactions)

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        return self.reactions[idx]


def read_entry(entry, smiles_set, gt_dict):
    rxn = entry["translation"]["Reaction"]
    gt_prot = entry["translation"].get("Sequence") or entry["translation"].get(
        "Protein sequence"
    )
    smiles_set.add(rxn)
    if gt_prot is None:
        return
    if isinstance(gt_prot, list):
        gt_dict[rxn].extend(gt_prot)
    else:
        gt_dict[rxn].append(gt_prot)


def load_json_reactions(input_file: Path) -> set[str]:
    smiles_set: set[str] = set()
    gt_dict: dict[str, list] = defaultdict(list)
    with input_file.open(encoding="utf-8") as handle:
        text = handle.read().strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                parsed_entries = [parsed]
            elif isinstance(parsed, list):
                parsed_entries = parsed
            else:
                raise ValueError(
                    f"Unsupported JSON top-level type in {input_file}: "
                    f"{type(parsed).__name__}"
                )
            for entry in parsed_entries:
                read_entry(entry, smiles_set, gt_dict)
        except json.JSONDecodeError:
            handle.seek(0)
            for line in handle:
                if line.strip():
                    read_entry(json.loads(line), smiles_set, gt_dict)
    return smiles_set


def load_reactions(path: Path) -> list[str]:
    """Load unique reaction SMILES from JSON/JSONL, smiles_map.json, or a txt file."""
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8").lstrip()
        if text.startswith("{") and '"translation"' not in text[:200]:
            meta = json.loads(path.read_text(encoding="utf-8"))
            return list(dict.fromkeys(str(value) for value in meta.values()))
        return sorted(load_json_reactions(path))
    return list(
        dict.fromkeys(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    )


def get_encoder_layer_hidden_state(encoder_outputs, layer_index):
    if layer_index is None:
        layer_index = -1
    hidden_states = getattr(encoder_outputs, "hidden_states", None)
    if hidden_states is None:
        if layer_index == -1:
            return encoder_outputs.last_hidden_state
        raise ValueError("Hidden states are unavailable; cannot select encoder layer.")
    num_states = len(hidden_states)
    if num_states == 0:
        raise ValueError("No encoder hidden states available.")
    if layer_index >= 0:
        if layer_index >= num_states:
            raise ValueError(
                f"layer_index {layer_index} out of range for encoder with {num_states} states."
            )
        target_idx = layer_index
    else:
        if abs(layer_index) > num_states:
            raise ValueError(
                f"layer_index {layer_index} out of range for encoder with {num_states} states."
            )
        target_idx = num_states + layer_index
    return hidden_states[target_idx]


def describe_embedding_source(layer_index):
    if layer_index is None or layer_index == -1:
        return "the last encoder hidden state"
    if layer_index == 0:
        return "the encoder embeddings (hidden_states[0])"
    if layer_index == 1:
        return "the first encoder hidden state (hidden_states[1])"
    if layer_index > 1:
        return f"encoder hidden state at layer index {layer_index} (hidden_states[{layer_index}])"
    return f"encoder hidden state {layer_index} from the end (hidden_states[{layer_index}])"


def pool_embedding(encoder_embedding, mask, avg_weighting=None):
    seq_len = mask.sum()
    masked_embeddings = encoder_embedding * mask
    sum_embeddings = masked_embeddings.sum(dim=0)
    mean_embedding = sum_embeddings / seq_len

    masked_for_max = encoder_embedding * mask + (1 - mask) * (-1e9)
    max_embedding, _ = masked_for_max.max(dim=0)

    mean_broadcasted = mean_embedding.unsqueeze(0).expand_as(encoder_embedding)
    centered_embeddings = (encoder_embedding - mean_broadcasted) * mask
    squared_centered = (centered_embeddings ** 2) * mask
    variance = squared_centered.sum(dim=0) / seq_len
    std_embedding = torch.sqrt(variance + 1e-8)

    if avg_weighting is not None:
        attn_weights = avg_weighting * mask.squeeze(-1)
        attn_weights = attn_weights / (attn_weights.sum() + 1e-8)
        mean_embedding = (encoder_embedding * attn_weights.unsqueeze(-1)).sum(dim=0)

    mean_embedding = mean_embedding / (mean_embedding.norm(p=2) + 1e-8)
    max_embedding = max_embedding / (max_embedding.norm(p=2) + 1e-8)
    std_embedding = std_embedding / (std_embedding.norm(p=2) + 1e-8)
    return torch.cat([mean_embedding, max_embedding, std_embedding], dim=0)


def compute_attention_rollout(cross_attentions):
    if not cross_attentions or len(cross_attentions) == 0:
        return None

    attn = torch.stack(
        [
            torch.stack(
                [layer_attn.mean(dim=1).squeeze(-2) for layer_attn in timestep_attns]
            )
            for timestep_attns in cross_attentions
        ]
    )
    attn = attn.permute(1, 0, 2, 3)
    rollout = attn[0]
    rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8)
    for layer_idx in range(1, attn.shape[0]):
        current_layer = attn[layer_idx]
        current_layer = current_layer / (current_layer.sum(dim=-1, keepdim=True) + 1e-8)
        rollout = rollout * current_layer + 0.1 * rollout
        rollout = rollout / (rollout.sum(dim=-1, keepdim=True) + 1e-8)
    return rollout.mean(dim=0)


def extract_attention_weighted_embeddings(
    model, tokenizer, reactions, device, batch_size=8, layer_index=-1
):
    model.eval()
    embeddings_dict = {}
    dataloader = DataLoader(ReactionDataset(reactions), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_reactions in tqdm(dataloader, desc="Extracting attention-weighted embeddings"):
            inputs = tokenizer(
                batch_reactions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                max_length=250,
            )
            avg_attention = compute_attention_rollout(outputs.cross_attentions)
            encoder_outputs = model.encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
            encoder_hidden_states = get_encoder_layer_hidden_state(
                encoder_outputs, layer_index
            )
            for i, reaction in enumerate(batch_reactions):
                attn_weights = avg_attention[i]
                mask = inputs["attention_mask"][i].unsqueeze(-1).float()
                embeddings_dict[reaction] = (
                    pool_embedding(
                        encoder_hidden_states[i], mask, avg_weighting=attn_weights
                    )
                    .cpu()
                    .numpy()
                )
    return embeddings_dict


def extract_embeddings(model, tokenizer, reactions, device, batch_size=8, layer_index=-1):
    model.eval()
    embeddings_dict = {}
    dataloader = DataLoader(ReactionDataset(reactions), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_reactions in tqdm(dataloader, desc="Extracting embeddings"):
            inputs = tokenizer(
                batch_reactions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            encoder_outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            encoder_embeddings = get_encoder_layer_hidden_state(
                encoder_outputs, layer_index
            )
            for i, reaction in enumerate(batch_reactions):
                mask = attention_mask[i].unsqueeze(-1).float()
                embeddings_dict[reaction] = (
                    pool_embedding(encoder_embeddings[i], mask).cpu().numpy()
                )
    return embeddings_dict


def extract_set(model, tokenizer, reactions, device, batch_size, layer_index, pooling: str):
    if pooling == "attention":
        return extract_attention_weighted_embeddings(
            model, tokenizer, reactions, device, batch_size, layer_index
        )
    return extract_embeddings(
        model, tokenizer, reactions, device, batch_size, layer_index
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Trained seq2seq checkpoint")
    parser.add_argument(
        "--tokenizer-mol",
        default=DEFAULT_TOKENIZER_MOL,
        help="SMILES tokenizer path or Hugging Face id",
    )
    parser.add_argument(
        "--train-file",
        default=DEFAULT_HF_REPO,
        help=(
            "Local training JSON/JSONL, or Hugging Face repo id "
            f"(default: {DEFAULT_HF_REPO}, file training_dataset.json)"
        ),
    )
    parser.add_argument(
        "--heldout-file",
        type=Path,
        default=DEFAULT_HELDOUT,
        help="Held-out reactions: smiles_map.json, SMILES txt, or JSON/JSONL",
    )
    parser.add_argument(
        "--output-folder",
        "--output",
        dest="output_folder",
        required=True,
        type=Path,
        help="Directory for train_embeddings.pkl and heldout_embeddings.pkl",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Encoder layer (-1 last, 0 embeddings). Paper analysis used 0.",
    )
    parser.add_argument(
        "--train-pooling",
        choices=["attention", "pooled"],
        default="attention",
        help="Pooling for training reactions (paper: attention)",
    )
    parser.add_argument(
        "--heldout-pooling",
        choices=["attention", "pooled"],
        default="pooled",
        help="Pooling for held-out reactions (paper: pooled)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Embedding values will be extracted from {describe_embedding_source(args.layer_index)}.")

    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    tokenizer_mol = AutoTokenizer.from_pretrained(args.tokenizer_mol)

    print("Loading training reactions...")
    train_file = resolve_train_json(args.train_file)
    train_reactions = load_reactions(train_file)
    print(f"Found {len(train_reactions)} unique training reactions")

    print("Loading held-out reactions...")
    heldout_reactions = load_reactions(args.heldout_file)
    print(f"Found {len(heldout_reactions)} unique held-out reactions")

    print(f"Extracting training embeddings ({args.train_pooling})...")
    train_embeddings = extract_set(
        model,
        tokenizer_mol,
        train_reactions,
        device,
        args.batch_size,
        args.layer_index,
        args.train_pooling,
    )

    print(f"Extracting held-out embeddings ({args.heldout_pooling})...")
    heldout_embeddings = extract_set(
        model,
        tokenizer_mol,
        heldout_reactions,
        device,
        args.batch_size,
        args.layer_index,
        args.heldout_pooling,
    )

    train_output_path = args.output_folder / "train_embeddings.pkl"
    heldout_output_path = args.output_folder / "heldout_embeddings.pkl"
    with train_output_path.open("wb") as handle:
        pickle.dump(train_embeddings, handle)
    with heldout_output_path.open("wb") as handle:
        pickle.dump(heldout_embeddings, handle)

    print(f"Training embeddings saved to: {train_output_path}")
    print(f"Held-out embeddings saved to: {heldout_output_path}")
    print(f"Training embeddings: {len(train_embeddings)} reactions")
    print(f"Held-out embeddings: {len(heldout_embeddings)} reactions")
    if train_embeddings:
        print(f"Embedding dimension: {next(iter(train_embeddings.values())).shape}")


if __name__ == "__main__":
    main()
