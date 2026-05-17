from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NpyTimeSeriesDataset, split_shape
from evaluate import evaluate_knn
from loss import supcon_loss
from model import TransformerEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SupCon Transformer on processed npy data.")
    parser.add_argument("--data-dir", type=Path, default=Path("binance_btcusdt_1m_singlefeature_T60_H10"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--memory-size", type=int, default=8192)
    parser.add_argument("--max-train-samples", type=int, default=65536)
    parser.add_argument("--max-val-samples", type=int, default=16384)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    train_shape = split_shape(args.data_dir, "train")
    val_shape = split_shape(args.data_dir, "val")
    seq_len, n_features = train_shape[1], train_shape[2]
    if val_shape[1:] != train_shape[1:]:
        raise ValueError(f"train shape {train_shape} and val shape {val_shape} are incompatible")

    with open(args.data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    train_stop = min(train_shape[0], args.max_train_samples) if args.max_train_samples else train_shape[0]
    val_stop = min(val_shape[0], args.max_val_samples) if args.max_val_samples else val_shape[0]
    mem_start, mem_stop = tail_slice(train_stop, args.memory_size)

    train_ds = NpyTimeSeriesDataset(args.data_dir, "train", 0, train_stop)
    memory_ds = NpyTimeSeriesDataset(args.data_dir, "train", mem_start, mem_stop)
    val_ds = NpyTimeSeriesDataset(args.data_dir, "val", 0, val_stop)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    memory_loader = DataLoader(memory_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    device = choose_device(args.device)
    model = TransformerEncoder(
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

    print(
        f"data_dir={args.data_dir} seq_len={seq_len} n_features={n_features} "
        f"train={len(train_ds)} val={len(val_ds)} memory={len(memory_ds)} "
        f"device={device} torch={torch.__version__}",
        flush=True,
    )
    print(f"meta model_features={meta.get('model_features')}", flush=True)

    history = []
    best_f1 = -1.0
    best_path = args.checkpoint_dir / f"supcon_transformer_F{n_features}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = model(x)
            loss = supcon_loss(z, y, temperature=args.temperature)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        scheduler.step()
        eval_result = evaluate_knn(model, val_loader, memory_loader, device, k=args.k)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_macro_f1": eval_result.macro_f1,
            "val_mcc": eval_result.mcc,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.time() - t0,
        }
        history.append(row)

        if eval_result.macro_f1 > best_f1:
            best_f1 = eval_result.macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "seq_len": seq_len,
                    "n_features": n_features,
                    "best_val_macro_f1": best_f1,
                    "epoch": epoch,
                },
                best_path,
            )

        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} val_mcc={row['val_mcc']:.4f} "
            f"lr={row['lr']:.2e} seconds={row['seconds']:.1f}",
            flush=True,
        )

    results_path = args.results_dir / f"supcon_train_F{n_features}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"history": history, "best_checkpoint": str(best_path)}, f, indent=2)
    print(f"saved best_checkpoint={best_path}", flush=True)
    print(f"saved results={results_path}", flush=True)


if __name__ == "__main__":
    main()
