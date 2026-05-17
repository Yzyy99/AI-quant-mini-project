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
from torch.utils.data import DataLoader

from baselines import TransformerCE, class_metrics, tail_slice
from dataset import NpyTimeSeriesDataset, split_shape
from evaluate import evaluate_knn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CE variants for noisy financial labels.")
    parser.add_argument("--data-dir", type=Path, default=Path("binance_btcusdt_1m_multifeature_selected_T60_H10"))
    parser.add_argument("--variant-name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight", choices=["none", "balanced", "inverse_sqrt"], default="none")
    parser.add_argument("--focal-gamma", type=float, default=0.0)
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


def make_class_weight(data_dir: Path, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none":
        return None
    y = np.load(data_dir / "y_train.npy", mmap_mode="r")
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=3).astype(np.float64)
    if mode == "balanced":
        weights = counts.sum() / (len(counts) * counts)
    elif mode == "inverse_sqrt":
        weights = np.sqrt(counts.mean() / counts)
    else:
        raise ValueError(f"unknown class weight mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def json_safe_args(args: argparse.Namespace) -> dict:
    safe = vars(args).copy()
    for key, value in list(safe.items()):
        if isinstance(value, Path):
            safe[key] = str(value)
    return safe


def regularized_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weight: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
) -> torch.Tensor:
    per_sample = F.cross_entropy(
        logits,
        target,
        weight=class_weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if focal_gamma <= 0:
        return per_sample.mean()
    pt = torch.softmax(logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1).clamp_min(1e-6)
    return ((1.0 - pt) ** focal_gamma * per_sample).mean()


def evaluate_softmax(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
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
    return class_metrics(torch.cat(y_true).numpy(), torch.cat(y_pred).numpy())


def run(args: argparse.Namespace, device: torch.device) -> dict:
    train_shape = split_shape(args.data_dir, "train")
    seq_len, n_features = train_shape[1], train_shape[2]
    train_ds = NpyTimeSeriesDataset(args.data_dir, "train")
    val_ds = NpyTimeSeriesDataset(args.data_dir, "val")
    mem_start, mem_stop = tail_slice(len(train_ds), args.memory_size)
    memory_ds = NpyTimeSeriesDataset(args.data_dir, "train", mem_start, mem_stop)

    pin_memory = device.type == "cuda"
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
    class_weight = make_class_weight(args.data_dir, args.class_weight, device)

    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.variant_name)
    best_f1 = {"val_macro_f1": -1.0, "epoch": 0}
    best_mcc = {"val_mcc": -1.0, "epoch": 0}
    best_path = args.checkpoint_dir / f"ce_regularized_F{n_features}_{safe_name}.pt"
    history = []

    print(
        f"CE regularized variant={args.variant_name} seq_len={seq_len} n_features={n_features} "
        f"train={len(train_ds)} val={len(val_ds)} memory={len(memory_ds)} "
        f"label_smoothing={args.label_smoothing} class_weight={args.class_weight} "
        f"focal_gamma={args.focal_gamma} dropout={args.dropout} weight_decay={args.weight_decay} "
        f"device={device}",
        flush=True,
    )
    if class_weight is not None:
        print(f"class_weight={class_weight.detach().cpu().tolist()}", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        losses = []
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = regularized_ce_loss(logits, y, class_weight, args.label_smoothing, args.focal_gamma)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        softmax_metrics = evaluate_softmax(model, val_loader, device)
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

        improved = False
        if row["val_macro_f1"] > best_f1["val_macro_f1"]:
            best_f1 = row.copy()
            improved = True
        if row["val_mcc"] > best_mcc["val_mcc"]:
            best_mcc = row.copy()
            improved = True
        if improved:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": json_safe_args(args),
                    "seq_len": seq_len,
                    "n_features": n_features,
                    "best_f1": best_f1,
                    "best_mcc": best_mcc,
                },
                best_path,
            )

        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} val_mcc={row['val_mcc']:.4f} "
            f"knn_f1={row['val_knn_macro_f1']:.4f} knn_mcc={row['val_knn_mcc']:.4f} "
            f"lr={row['lr']:.2e} seconds={row['seconds']:.1f}",
            flush=True,
        )

    return {
        "experiment": "ce_regularized",
        "variant": args.variant_name,
        "args": json_safe_args(args),
        "history": history,
        "best_f1": best_f1,
        "best_mcc": best_mcc,
        "best_checkpoint": str(best_path),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    result = run(args, device)
    n_features = split_shape(args.data_dir, "train")[2]
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in args.variant_name)
    out_path = args.results_dir / f"ce_regularized_F{n_features}_{safe_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved results={out_path}", flush=True)


if __name__ == "__main__":
    main()
