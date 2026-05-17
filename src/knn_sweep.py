from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from baselines import TransformerCE
from dataset import NpyTimeSeriesDataset, split_shape
from model import TransformerEncoder


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KNN inference sweep for trained checkpoints.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-type", choices=["supcon", "ce"], required=True)
    parser.add_argument("--embedding", choices=["cls", "projection"], default="cls")
    parser.add_argument("--memory-sizes", type=parse_int_list, default=parse_int_list("10000,50000,100000"))
    parser.add_argument("--ks", type=parse_int_list, default=parse_int_list("5,10,20,50,100"))
    parser.add_argument("--votes", type=parse_str_list, default=parse_str_list("majority,exp0.05,exp0.1"))
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(args: argparse.Namespace, seq_len: int, n_features: int) -> torch.nn.Module:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    common = {
        "seq_len": seq_len,
        "n_features": n_features,
        "d_model": saved_args.get("d_model", 128),
        "n_heads": saved_args.get("n_heads", 4),
        "n_layers": saved_args.get("n_layers", 3),
        "dim_feedforward": saved_args.get("dim_feedforward", 256),
        "embed_dim": saved_args.get("embed_dim", 64),
        "dropout": saved_args.get("dropout", 0.1),
    }
    if args.checkpoint_type == "supcon":
        model = TransformerEncoder(**common)
    else:
        model = TransformerCE(**common)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def encode(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    checkpoint_type: str,
    embedding: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embs = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            if embedding == "projection":
                if checkpoint_type != "supcon":
                    raise ValueError("projection embedding is only meaningful for SupCon checkpoints")
                z = model(x)
            else:
                z = model(x, return_embedding=True)
                z = F.normalize(z, dim=1)
            embs.append(z.detach().cpu())
            labels.append(y.cpu())
    return torch.cat(embs, dim=0), torch.cat(labels, dim=0)


def vote_predictions(top_labels: torch.Tensor, top_sims: torch.Tensor, k: int, vote: str) -> torch.Tensor:
    labels = top_labels[:, :k]
    sims = top_sims[:, :k]
    if vote == "majority":
        return torch.mode(labels, dim=1).values

    scores = torch.zeros((labels.shape[0], 3), dtype=torch.float32)
    if vote.startswith("exp"):
        tau = float(vote.replace("exp", ""))
        weights = torch.exp((sims - sims.max(dim=1, keepdim=True).values) / tau)
    elif vote == "sim":
        weights = sims - sims.min(dim=1, keepdim=True).values + 1e-6
    else:
        raise ValueError(f"unknown vote mode: {vote}")

    for cls in range(3):
        scores[:, cls] = (weights * (labels == cls)).sum(dim=1)
    return scores.argmax(dim=1)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    train_shape = split_shape(args.data_dir, "train")
    seq_len, n_features = train_shape[1], train_shape[2]
    max_memory = min(max(args.memory_sizes), train_shape[0])
    args.memory_sizes = [min(m, train_shape[0]) for m in args.memory_sizes]
    max_k = max(args.ks)

    model = build_model(args, seq_len, n_features).to(device)
    mem_start = train_shape[0] - max_memory
    memory_ds = NpyTimeSeriesDataset(args.data_dir, "train", mem_start, train_shape[0])
    val_ds = NpyTimeSeriesDataset(args.data_dir, "val")
    memory_loader = DataLoader(
        memory_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"checkpoint={args.checkpoint} type={args.checkpoint_type} embedding={args.embedding} "
        f"device={device} memory_max={max_memory} val={len(val_ds)}",
        flush=True,
    )
    memory_embeddings, memory_labels = encode(model, memory_loader, device, args.checkpoint_type, args.embedding)
    query_embeddings, y_true = encode(model, val_loader, device, args.checkpoint_type, args.embedding)
    y_true_np = y_true.numpy()

    results = []
    for memory_size in args.memory_sizes:
        mem_emb = memory_embeddings[-memory_size:].to(device)
        mem_lab = memory_labels[-memory_size:].to(device)
        combo_preds = {(k, vote): [] for k in args.ks for vote in args.votes}

        for start in range(0, query_embeddings.shape[0], args.eval_batch_size):
            q = query_embeddings[start : start + args.eval_batch_size].to(device)
            sim = q @ mem_emb.T
            top_sims, top_idx = sim.topk(min(max_k, memory_size), dim=1)
            top_labels = mem_lab[top_idx]
            for k in args.ks:
                if k > memory_size:
                    continue
                for vote in args.votes:
                    combo_preds[(k, vote)].append(vote_predictions(top_labels, top_sims, k, vote).cpu())

        for (k, vote), parts in combo_preds.items():
            if not parts:
                continue
            y_pred = torch.cat(parts).numpy()
            row = {
                "checkpoint_type": args.checkpoint_type,
                "embedding": args.embedding,
                "memory_size": memory_size,
                "k": k,
                "vote": vote,
                "macro_f1": float(f1_score(y_true_np, y_pred, average="macro")),
                "mcc": float(matthews_corrcoef(y_true_np, y_pred)),
            }
            results.append(row)
            print(
                f"memory={memory_size} k={k} vote={vote} "
                f"f1={row['macro_f1']:.4f} mcc={row['mcc']:.4f}",
                flush=True,
            )

    best_f1 = max(results, key=lambda r: r["macro_f1"])
    best_mcc = max(results, key=lambda r: r["mcc"])
    payload = {"results": results, "best_f1": best_f1, "best_mcc": best_mcc}
    print(f"BEST_F1 {best_f1}", flush=True)
    print(f"BEST_MCC {best_mcc}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"saved results={args.out}", flush=True)


if __name__ == "__main__":
    main()
