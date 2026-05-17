import torch


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss for L2-normalized features."""

    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.ndim}D")
    if labels.ndim != 1:
        labels = labels.view(-1)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels batch sizes differ")

    device = features.device
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)

    sim = torch.matmul(features, features.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
    positive_mask = torch.eq(labels, labels.T) & self_mask

    exp_sim = torch.exp(sim) * self_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return features.new_tensor(0.0, requires_grad=True)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()

