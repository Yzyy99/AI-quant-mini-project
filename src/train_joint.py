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
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader

from dataset import NpyTimeSeriesDataset, split_shape
from evaluate import evaluate_knn
from loss import supcon_loss
from model import TransformerEncoder


class JointTransformer(nn.Module):
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

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        cls = self.encoder.encode(x)
        if return_embedding:
            return F.normalize(cls, dim=1)
        logits = self.classifier(cls)
        z = self.encoder.projection_head(cls)
        z = F.normalize(z, dim=1)
        return logits, z


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CE + SupCon joint model.")
    parser.add_argument("--data-dir", type=Path, default=Path("binance_btcusdt_1m_multifeature_selected_T60_H10"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--supcon-weight", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
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


def evaluate_softmax(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    true_parts = []
    pred_parts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits, _ = model(x)
            true_parts.append(y.cpu())
            pred_parts.append(logits.argmax(dim=1).cpu())
    y_true = torch.cat(true_parts).numpy()
    y_pred = torch.cat(pred_parts).numpy()
    return float(f1_score(y_true, y_pred, average="macro")), float(matthews_corrcoef(y_true, y_pred))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_shape = split_shape(args.data_dir, "train")
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
        drop_last=True,
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

    device = choose_device(args.device)
    model = JointTransformer(
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
    ce_loss_fn = nn.CrossEntropyLoss()

    print(
        f"Joint CE+SupCon seq_len={seq_len} n_features={n_features} train={len(train_ds)} "
        f"val={len(val_ds)} memory={len(memory_ds)} lambda={args.supcon_weight} device={device}",
        flush=True,
    )

    history = []
    best = {"val_macro_f1": -1.0, "val_mcc": -1.0, "epoch": 0}
    tag = f"F{n_features}_lam{args.supcon_weight:g}".replace(".", "p")
    best_path = args.checkpoint_dir / f"joint_ce_supcon_{tag}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ce_losses = []
        sup_losses = []
        total_losses = []
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, z = model(x)
            ce = ce_loss_fn(logits, y)
            sup = supcon_loss(z, y, temperature=args.temperature)
            loss = ce + args.supcon_weight * sup
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ce_losses.append(float(ce.detach().cpu()))
            sup_losses.append(float(sup.detach().cpu()))
            total_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_f1, val_mcc = evaluate_softmax(model, val_loader, device)
        knn = evaluate_knn(model, val_loader, memory_loader, device, k=args.k)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_losses)),
            "ce_loss": float(np.mean(ce_losses)),
            "supcon_loss": float(np.mean(sup_losses)),
            "val_macro_f1": val_f1,
            "val_mcc": val_mcc,
            "val_knn_macro_f1": knn.macro_f1,
            "val_knn_mcc": knn.mcc,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(row)
        if row["val_macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": row["val_macro_f1"], "val_mcc": row["val_mcc"], "epoch": epoch}
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
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} ce={row['ce_loss']:.4f} "
            f"supcon={row['supcon_loss']:.4f} val_f1={row['val_macro_f1']:.4f} "
            f"val_mcc={row['val_mcc']:.4f} knn_f1={row['val_knn_macro_f1']:.4f} "
            f"knn_mcc={row['val_knn_mcc']:.4f} lr={row['lr']:.2e} seconds={row['seconds']:.1f}",
            flush=True,
        )

    out_path = args.results_dir / f"joint_ce_supcon_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "best": best, "best_checkpoint": str(best_path)}, f, indent=2)
    print(f"saved best_checkpoint={best_path}", flush=True)
    print(f"saved results={out_path}", flush=True)


if __name__ == "__main__":
    main()
