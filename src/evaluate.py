from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    macro_f1: float
    mcc: float
    y_true: np.ndarray
    y_pred: np.ndarray


def _encode_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = []
    labels = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            emb = model(x, return_embedding=True)
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb.cpu())
            labels.append(y.cpu())
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def evaluate_knn(
    model: torch.nn.Module,
    query_loader: DataLoader,
    memory_loader: DataLoader,
    device: torch.device,
    k: int = 20,
    sim_batch_size: int = 512,
) -> EvalResult:
    memory_embeddings, memory_labels = _encode_loader(model, memory_loader, device)
    k = min(k, len(memory_labels))

    y_true_batches = []
    y_pred_batches = []
    model.eval()
    with torch.no_grad():
        for x, y in query_loader:
            x = x.to(device, non_blocking=True)
            query = model(x, return_embedding=True)
            query = F.normalize(query, dim=1).cpu()

            preds = []
            for start in range(0, query.shape[0], sim_batch_size):
                q = query[start : start + sim_batch_size]
                sim = q @ memory_embeddings.T
                topk_idx = sim.topk(k, dim=1).indices
                topk_labels = memory_labels[topk_idx]
                pred = torch.mode(topk_labels, dim=1).values
                preds.append(pred)

            y_true_batches.append(y.cpu())
            y_pred_batches.append(torch.cat(preds, dim=0))

    y_true = torch.cat(y_true_batches).numpy()
    y_pred = torch.cat(y_pred_batches).numpy()
    return EvalResult(
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        mcc=float(matthews_corrcoef(y_true, y_pred)),
        y_true=y_true,
        y_pred=y_pred,
    )

