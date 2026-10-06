#!/usr/bin/env python3
"""DPO optimization + sequence generation for Rexzyme encoder-decoder models.

Port of the original DPO_pLM.py + seq_gen.py pipeline to the current repo.

Adaptations vs. the original scripts:
- No "translation" prefix prepended to reaction SMILES (current model was not
  trained with it).
- Tokenizer loading supports owner/repo/subfolder Hub paths
  (e.g. AI4PD/REXzyme/tokenizer_aa).
- Paths and hyperparameters are CLI arguments instead of being hardcoded.
- Perplexity during generation is conditioned on the reaction (encoder input),
  which is the correct definition for an encoder-decoder model.

Everything else matches the original logic exactly:
- weighted mode: F.cross_entropy(beta * pi_ll, softmax(scores))
- ranked mode:   F.mse_loss(beta * pi_ll, ones) after sorting by score
- Log-likelihood is computed per sequence using model.loss (averaged NLL).
- The reference model is loaded but not used in the loss, matching the
  original DPO_pLM.py behaviour where ref_log_probs was set to None.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def split_hf_subfolder_ref(spec: str) -> tuple[str, str | None]:
    """Support owner/repo/subfolder shorthand for Hub references."""
    if Path(spec).exists():
        return spec, None
    parts = spec.strip("/").split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return spec, None


def load_tokenizer(path: str) -> AutoTokenizer:
    ref, subfolder = split_hf_subfolder_ref(path)
    kwargs = {"subfolder": subfolder} if subfolder else {}
    return AutoTokenizer.from_pretrained(ref, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO optimize and generate with Rexzyme (original DPO_pLM logic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True, help="Finetuned model checkpoint.")
    parser.add_argument("--output_model_dir", required=True, help="Directory to save best DPO model.")
    parser.add_argument("--tokenizer_aa", required=True, help="AA tokenizer (local path or Hub id).")
    parser.add_argument("--tokenizer_mol", required=True, help="SMILES tokenizer (local path or Hub id).")

    parser.add_argument("--activity_file", required=True,
                        help="CSV/TSV/XLSX with sequences and proxy values.")
    parser.add_argument("--sequence_column", default="Seq",
                        help="Column name for protein sequences.")
    parser.add_argument("--proxy_column", required=True,
                        help="Numeric proxy column (e.g. Kcat, Km, Kcat/Km, plddt).")
    parser.add_argument("--reaction", default=None,
                        help="Reaction SMILES used for all rows (if --reaction_column is not set).")
    parser.add_argument("--reaction_column", default=None,
                        help="Optional per-row reaction SMILES column.")
    parser.add_argument("--mode", choices=["weighted", "ranked"], required=True,
                        help="DPO objective mode.")
    parser.add_argument("--objective", choices=["maximize", "minimize"], default="maximize",
                        help="Whether larger proxy values are better (maximize) or worse (minimize).")

    # Training hyperparameters — defaults match the original DPO_pLM.py CONFIG
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data used for training (rest goes to eval).")
    parser.add_argument("--seed", type=int, default=1998)
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=1024)

    # Generation — defaults match seq_gen.py
    parser.add_argument("--generate_after_dpo", action="store_true",
                        help="Run sequence generation after optimization.")
    parser.add_argument("--generation_reaction", default=None,
                        help="Reaction SMILES to generate from (defaults to --reaction).")
    parser.add_argument("--fasta_output", default="dpo_generated.fasta",
                        help="Output FASTA path.")
    parser.add_argument("--generation_rounds", type=int, default=500,
                        help="Number of generation rounds (original: 500).")
    parser.add_argument("--num_return_sequences", type=int, default=20,
                        help="Sequences generated per round (original: 20).")
    parser.add_argument("--top_k", type=int, default=18,
                        help="top-k sampling (original: 18).")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty (original: 1.1).")
    parser.add_argument("--generation_max_length", type=int, default=1024)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported activity file extension: {suffix}")


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(" ", "", regex=False), errors="coerce")


def split_frame(
    frame: pd.DataFrame, train_split: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = list(range(len(frame)))
    random.Random(seed).shuffle(indices)
    pivot = int(len(indices) * train_split)
    return (
        frame.iloc[indices[:pivot]].reset_index(drop=True),
        frame.iloc[indices[pivot:]].reset_index(drop=True),
    )


class PreferenceDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]
        return {
            "sequence": row["sequence"],
            "reaction": row["reaction"],
            "score": float(row["score"]),
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "sequences": [x["sequence"] for x in batch],
        "reactions": [x["reaction"] for x in batch],
        "scores": torch.tensor([x["score"] for x in batch], dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Log-likelihood (original DPO_pLM.py approach: per-sequence, via model.loss)
# ---------------------------------------------------------------------------

def log_likelihood(
    sequences: list[str],
    reactions: list[str],
    device: torch.device,
    model: AutoModelForSeq2SeqLM,
    tokenizer_aa: AutoTokenizer,
    tokenizer_mol: AutoTokenizer,
    max_source_length: int,
    max_target_length: int,
) -> torch.Tensor:
    """Return per-sequence mean log-likelihood using model.loss (averaged NLL negated).

    Iterates one sequence at a time, matching the original DPO_pLM.py log_likelihood.
    No "translation" prefix is added to reactions.
    """
    all_ll: list[torch.Tensor] = []
    for seq, reac in zip(sequences, reactions):
        inputs = tokenizer_mol(
            reac,
            max_length=max_source_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        label_ids = tokenizer_aa(
            seq,
            max_length=max_target_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        outputs = model(**inputs, labels=label_ids)
        # outputs.loss is mean NLL; negate to get mean log-likelihood
        all_ll.append(-outputs.loss.unsqueeze(0))
    return torch.cat(all_ll)


# ---------------------------------------------------------------------------
# Loss functions (original DPO_pLM.py)
# ---------------------------------------------------------------------------

def weighted_dpo_loss(pi_ll: torch.Tensor, scores: torch.Tensor, beta: float) -> torch.Tensor:
    """Original weighted loss: F.cross_entropy(beta * pi_ll, softmax(scores)).

    Encourages the model's sequence-level likelihood distribution to match the
    score ranking. Reference model is not used (matching original behaviour).
    """
    pi_ratio = beta * pi_ll
    weights = torch.softmax(scores, dim=0)
    return F.cross_entropy(pi_ratio, weights)


def ranked_dpo_loss(pi_ll: torch.Tensor, scores: torch.Tensor, beta: float) -> torch.Tensor:
    """Original ranked loss: sort by softmax score, then MSE(beta * pi_ll, ones).

    Sorts sequences by their score weight (descending) and pushes all
    log-likelihoods toward 1/beta via MSE. Reference model is not used.
    """
    weights = torch.softmax(scores, dim=0)
    sorted_indices = torch.argsort(weights, descending=True)
    pi_ll_sorted = pi_ll[sorted_indices]
    pi_ratio = beta * pi_ll_sorted
    return F.mse_loss(pi_ratio, torch.ones_like(pi_ratio))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: AutoModelForSeq2SeqLM,
    data_loader: DataLoader,
    optimizer: AdamW,
    tokenizer_aa: AutoTokenizer,
    tokenizer_mol: AutoTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in data_loader:
        optimizer.zero_grad()
        pi_ll = log_likelihood(
            batch["sequences"], batch["reactions"], device, model,
            tokenizer_aa, tokenizer_mol, args.max_source_length, args.max_target_length,
        )
        scores = batch["scores"].to(device)
        if args.mode == "weighted":
            loss = weighted_dpo_loss(pi_ll, scores, args.beta)
        else:
            loss = ranked_dpo_loss(pi_ll, scores, args.beta)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def eval_epoch(
    model: AutoModelForSeq2SeqLM,
    data_loader: DataLoader,
    tokenizer_aa: AutoTokenizer,
    tokenizer_mol: AutoTokenizer,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            pi_ll = log_likelihood(
                batch["sequences"], batch["reactions"], device, model,
                tokenizer_aa, tokenizer_mol, args.max_source_length, args.max_target_length,
            )
            scores = batch["scores"].to(device)
            if args.mode == "weighted":
                loss = weighted_dpo_loss(pi_ll, scores, args.beta)
            else:
                loss = ranked_dpo_loss(pi_ll, scores, args.beta)
            losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


# ---------------------------------------------------------------------------
# Generation (original seq_gen.py logic, adapted)
# ---------------------------------------------------------------------------

def compute_perplexity(
    model: AutoModelForSeq2SeqLM,
    tokenizer_aa: AutoTokenizer,
    tokenizer_mol: AutoTokenizer,
    reaction: str,
    seq_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Conditional perplexity P(seq | reaction) via model.loss."""
    src = tokenizer_mol(reaction, return_tensors="pt").to(device)
    labels = seq_ids.clone()
    pad_id = tokenizer_aa.pad_token_id
    if pad_id is not None:
        labels = labels.masked_fill(labels == pad_id, -100)
    with torch.no_grad():
        outputs = model(**src, labels=labels)
    return math.exp(outputs.loss.item())


def generate_fasta(
    model: AutoModelForSeq2SeqLM,
    tokenizer_aa: AutoTokenizer,
    tokenizer_mol: AutoTokenizer,
    reaction: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Generate sequences over multiple rounds, sort by perplexity, save FASTA.

    Matches seq_gen.py structure: generation_rounds × num_return_sequences
    sequences, sorted ascending by perplexity.
    """
    model.eval()
    input_ids = tokenizer_mol(reaction, return_tensors="pt").input_ids.to(device)
    all_records: list[tuple[str, float]] = []

    for round_idx in range(args.generation_rounds):
        outputs = model.generate(
            input_ids,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_length=args.generation_max_length,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )
        for out_ids in outputs:
            seq = tokenizer_aa.decode(out_ids, skip_special_tokens=True)
            ppl = compute_perplexity(
                model, tokenizer_aa, tokenizer_mol, reaction, out_ids.unsqueeze(0), device
            )
            all_records.append((seq, ppl))
        if (round_idx + 1) % 50 == 0:
            print(f"  Generation round {round_idx + 1}/{args.generation_rounds} "
                  f"({len(all_records)} sequences so far)")

    all_records.sort(key=lambda x: x[1])

    out_path = Path(args.fasta_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, (seq, ppl) in enumerate(all_records):
            fh.write(f">{idx}_ppl_{ppl:.4f}\n{seq}\n")
    print(f"Saved {len(all_records)} sequences to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and validate activity table
    frame = read_table(Path(args.activity_file))
    for col in (args.sequence_column, args.proxy_column):
        if col not in frame.columns:
            raise ValueError(f"Column {col!r} not found in {args.activity_file}. "
                             f"Available: {list(frame.columns)}")

    if args.reaction_column:
        if args.reaction_column not in frame.columns:
            raise ValueError(f"Column {args.reaction_column!r} not found.")
        reactions = frame[args.reaction_column].astype(str)
    else:
        if not args.reaction:
            raise ValueError("Provide either --reaction or --reaction_column.")
        reactions = pd.Series([args.reaction] * len(frame))

    work = pd.DataFrame({
        "sequence": frame[args.sequence_column].astype(str).str.strip(),
        "reaction": reactions.astype(str).str.strip(),
        "score": clean_numeric(frame[args.proxy_column]),
    })
    work = work[(work["sequence"] != "") & (work["reaction"] != "")].dropna(subset=["score"])
    if len(work) < 4:
        raise ValueError(f"Need at least 4 valid rows after filtering; got {len(work)}.")

    if args.objective == "minimize":
        work["score"] = -work["score"]

    train_df, eval_df = split_frame(work, train_split=args.train_split, seed=args.seed)
    print(f"Loaded {len(work)} rows ({len(train_df)} train / {len(eval_df)} eval).")

    train_loader = DataLoader(
        PreferenceDataset(train_df), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        PreferenceDataset(eval_df), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_batch,
    )

    tokenizer_mol = load_tokenizer(args.tokenizer_mol)
    tokenizer_aa = load_tokenizer(args.tokenizer_aa)

    policy_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    # Reference model is loaded but not used in the loss, matching original DPO_pLM.py.
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        policy_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    best_eval = float("inf")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train_epoch(
            policy_model, train_loader, optimizer,
            tokenizer_aa, tokenizer_mol, args, device,
        )
        eval_loss = eval_epoch(
            policy_model, eval_loader,
            tokenizer_aa, tokenizer_mol, args, device,
        )
        print(f"Epoch {epoch}/{args.num_epochs} - "
              f"train_loss={train_loss:.6f}  eval_loss={eval_loss:.6f}")

        if eval_loss < best_eval:
            best_eval = eval_loss
            out_dir = Path(args.output_model_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            policy_model.save_pretrained(out_dir)
            tokenizer_aa.save_pretrained(out_dir / "tokenizer_aa")
            tokenizer_mol.save_pretrained(out_dir / "tokenizer_smiles")
            print(f"  → Saved best model to {out_dir}")

    if args.generate_after_dpo:
        generation_reaction = args.generation_reaction or args.reaction
        if not generation_reaction:
            raise ValueError(
                "Set --generation_reaction (or --reaction) when using --generate_after_dpo."
            )
        print(f"\nGenerating {args.generation_rounds} rounds × "
              f"{args.num_return_sequences} sequences from best model …")
        reloaded = AutoModelForSeq2SeqLM.from_pretrained(args.output_model_dir).to(device)
        generate_fasta(reloaded, tokenizer_aa, tokenizer_mol, generation_reaction, args, device)


if __name__ == "__main__":
    main()
