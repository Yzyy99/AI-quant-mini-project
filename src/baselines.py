from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from dataset import NpyTimeSeriesDataset, split_shape
from evaluate import evaluate_knn
from model import TransformerEncoder


class TransformerCE(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        embed_dim: int = 64,
        dropout: float = 0.1,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(
            seq_len=seq_len,
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        emb = self.encoder(x, return_embedding=True)
        if return_embedding:
            return emb
        return self.classifier(emb)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw KNN and CE baselines.")
    parser.add_argument("--data-dir", type=Path, default=Path("binance_btcusdt_1m_singlefeature_T60_H10"))
    parser.add_argument("--mode", choices=["raw-knn", "ce", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tail_slice(total: int, size: int) -> tuple[int, int]:
    size = min(size, total)
    return total - size, total


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Neutral", "Up", "Down"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def load_flat_window(data_dir: Path, split: str, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
    y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
    X_part = np.asarray(X[start:stop], dtype=np.float32).reshape(stop - start, -1)
    y_part = np.asarray(y[start:stop], dtype=np.int64)
    return X_part, y_part


def raw_knn(
    data_dir: Path,
    device: torch.device,
    memory_size: int,
    k: int,
    eval_batch_size: int,
) -> dict:
    train_n = split_shape(data_dir, "train")[0]
    val_n = split_shape(data_dir, "val")[0]
    mem_start, mem_stop = tail_slice(train_n, memory_size)
    t0 = time.time()

    X_mem, y_mem = load_flat_window(data_dir, "train", mem_start, mem_stop)
    X_val, y_val = load_flat_window(data_dir, "val", 0, val_n)

    memory = torch.from_numpy(X_mem).to(device)
    memory_labels = torch.from_numpy(y_mem).to(device)
    memory_norm = (memory * memory).sum(dim=1).view(1, -1)
    k = min(k, len(memory_labels))

    preds = []
    with torch.no_grad():
        for start in range(0, len(X_val), eval_batch_size):
            q = torch.from_numpy(X_val[start : start + eval_batch_size]).to(device)
            q_norm = (q * q).sum(dim=1).view(-1, 1)
            dist = q_norm + memory_norm - 2.0 * (q @ memory.T)
            topk_idx = dist.topk(k, dim=1, largest=False).indices
            topk_labels = memory_labels[topk_idx]
            pred = torch.mode(topk_labels, dim=1).values
            preds.append(pred.cpu())
    y_pred = torch.cat(preds).numpy()
    result = class_metrics(y_val, y_pred)
    result.update(
        {
            "baseline": "raw_knn",
            "split": "val",
            "memory_size": int(memory_size),
            "k": int(k),
            "seconds": time.time() - t0,
        }
    )
    return result


def evaluate_ce_softmax(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1).cpu()
            y_true.append(y.cpu())
            y_pred.append(pred)
    yt = torch.cat(y_true).numpy()
    yp = torch.cat(y_pred).numpy()
    return class_metrics(yt, yp)


def run_ce(args: argparse.Namespace, device: torch.device) -> dict:
    train_shape = split_shape(args.data_dir, "train")
    val_shape = split_shape(args.data_dir, "val")
    seq_len, n_features = train_shape[1], train_shape[2]
    train_ds = NpyTimeSeriesDataset(args.data_dir, "train")
    val_ds = NpyTimeSeriesDataset(args.data_dir, "val")
    mem_start, mem_stop = tail_slice(len(train_ds), args.memory_size)
    memory_ds = NpyTimeSeriesDataset(args.data_dir, "train", mem_start, mem_stop)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    memory_loader = DataLoader(
        memory_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = TransformerCE(
        seq_len=seq_len,
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss()

    best = {"val_macro_f1": -1.0, "epoch": 0}
    best_path = args.checkpoint_dir / f"ce_transformer_F{n_features}.pt"
    history = []

    print(
        f"CE baseline seq_len={seq_len} n_features={n_features} train={len(train_ds)} "
        f"val={len(val_ds)} memory={len(memory_ds)} device={device}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        softmax_metrics = evaluate_ce_softmax(model, val_loader, device)
        knn_metrics = evaluate_knn(model, val_loader, memory_loader, device, k=args.k)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": softmax_metrics["macro_f1"],
            "val_mcc": softmax_metrics["mcc"],
            "val_knn_macro_f1": knn_metrics.macro_f1,
            "val_knn_mcc": knn_metrics.mcc,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(row)

        if row["val_macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": row["val_macro_f1"], "epoch": epoch}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "seq_len": seq_len,
                    "n_features": n_features,
                    "best": best,
                },
                best_path,
            )

        print(
            f"epoch={epoch:03d} ce_loss={row['train_loss']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} val_mcc={row['val_mcc']:.4f} "
            f"ce_knn_f1={row['val_knn_macro_f1']:.4f} ce_knn_mcc={row['val_knn_mcc']:.4f} "
            f"lr={row['lr']:.2e} seconds={row['seconds']:.1f}",
            flush=True,
        )

    return {
        "baseline": "transformer_ce",
        "history": history,
        "best": best,
        "best_checkpoint": str(best_path),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)

    with open(args.data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"data_dir={args.data_dir} model_features={meta.get('model_features')}", flush=True)

    results: dict[str, object] = {}
    if args.mode in {"raw-knn", "all"}:
        raw = raw_knn(args.data_dir, device, args.memory_size, args.k, args.eval_batch_size)
        results["raw_knn"] = raw
        print(
            f"raw_knn val_macro_f1={raw['macro_f1']:.4f} val_mcc={raw['mcc']:.4f} "
            f"seconds={raw['seconds']:.1f}",
            flush=True,
        )

    if args.mode in {"ce", "all"}:
        results["transformer_ce"] = run_ce(args, device)

    n_features = split_shape(args.data_dir, "train")[2]
    out_path = args.results_dir / f"baselines_F{n_features}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"saved results={out_path}", flush=True)


if __name__ == "__main__":
    main()
