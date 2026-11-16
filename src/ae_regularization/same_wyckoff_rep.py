import torch
import torch.nn.functional as F

def duplicate_same_wyckoff_rep_loss_per_graph(
    reps_frac: torch.Tensor,          # [N, 3]
    masked_logits_w: torch.Tensor,    # [N, V]
    batch_index: torch.Tensor,        # [N]
    num_graphs: int,
    tol: float = 1e-3,
    lengths_per_graph: torch.Tensor | None = None,   # [B, 3]
    beta: float = 20.0,
) -> torch.Tensor:
    """
    Penalize pairs of representatives in the same graph that:
      1) are very close in coordinate space
      2) are predicted to have the same Wyckoff type

    Returns per-graph loss: [B]
    """
    device = reps_frac.device
    probs = F.softmax(masked_logits_w, dim=1)   # [N, V]

    losses = torch.zeros(num_graphs, device=device)

    for g in range(num_graphs):
        idx = (batch_index == g).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            continue

        x = reps_frac[idx]      # [M, 3]
        p = probs[idx]          # [M, V]

        # pairwise wrapped deltas
        delta = x[:, None, :] - x[None, :, :]   # [M, M, 3]
        delta = (delta + 0.5) % 1.0 - 0.5

        if lengths_per_graph is not None:
            lengths = lengths_per_graph[g]      # [3]
            delta = delta * lengths.view(1, 1, 3)

        dist = torch.linalg.norm(delta, dim=-1)   # [M, M]

        M = dist.shape[0]
        eye = torch.eye(M, dtype=torch.bool, device=device)

        # soft closeness penalty (smooth version of "same key")
        # large when distance ~ 0, decays smoothly after tol
        closeness = F.softplus(tol - dist, beta=beta) ** 2   # [M, M]

        # soft probability that two reps have same Wyckoff
        # if p_i and p_j are one-hot on same class -> high
        same_wp = p @ p.T   # [M, M]

        # ignore diagonal
        closeness = closeness.masked_fill(eye, 0.0)
        same_wp = same_wp.masked_fill(eye, 0.0)

        penalty = closeness * same_wp

        # average over off-diagonal pairs
        num_pairs = M * (M - 1)
        losses[g] = penalty.sum() / max(num_pairs, 1)

    return losses
