import torch
import torch.nn.functional as F



def duplicate_zero_dof_wyckoff_loss_per_graph(
    masked_logits_w: torch.Tensor,   # [N, V]
    batch_index: torch.Tensor,       # [N]
    num_graphs: int,
    zero_dof_mask_vocab: torch.Tensor,   # [V] bool, True for zero-DOF Wyckoff types
) -> torch.Tensor:
    probs = F.softmax(masked_logits_w, dim=-1)   # [N, V]
    probs_zero = probs * zero_dof_mask_vocab.float().unsqueeze(0)  # [N, V]

    losses = probs.new_zeros(num_graphs)

    for g in range(num_graphs):
        idx = (batch_index == g).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            continue

        p = probs_zero[idx]   # [M, V]

        # pairwise same-Wyckoff affinity
        same = p @ p.T   # [M, M]

        M = same.shape[0]
        eye = torch.eye(M, dtype=torch.bool, device=same.device)
        same = same.masked_fill(eye, 0.0)

        num_pairs = M * (M - 1)
        losses[g] = same.sum() / max(num_pairs, 1)

    return losses
