import torch
import torch.nn.functional as F


def make_free_vector_from_full_pos(
    pos_pred: torch.Tensor,
    free_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Convert full xyz positional predictions [N, 3] into free-variable vectors [N, 3].

    pos_pred:
        Full xyz predictions, e.g. [x_pred, y_pred, z_pred]

    free_mask:
        Positional free-coordinate mask, e.g.
        [1, 0, 0] for x,2x,1/4
        [1, 0, 1] for x,0,z
        [1, 1, 0] for x,y,0

    Returns:
        f_pred [N, 3], where non-free entries are zeroed.
    """
    return pos_pred * free_mask.float()


def lattice_matrix_from_params(cell_full: torch.Tensor, angles_in_degrees: bool = False) -> torch.Tensor:
    """
    cell_full: [B, 6] = [a, b, c, alpha, beta, gamma]
    returns:  [B, 3, 3]
    """
    a, b, c, alpha, beta, gamma = cell_full.unbind(dim=-1)

    if angles_in_degrees:
        deg2rad = torch.pi / 180.0
        alpha = alpha * deg2rad
        beta = beta * deg2rad
        gamma = gamma * deg2rad

    cos_alpha = torch.cos(alpha)
    cos_beta = torch.cos(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma).clamp_min(1e-8)

    ax = a
    ay = torch.zeros_like(a)
    az = torch.zeros_like(a)

    bx = b * cos_gamma
    by = b * sin_gamma
    bz = torch.zeros_like(b)

    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

    cz_sq = c**2 - cx**2 - cy**2
    cz = torch.sqrt(torch.clamp(cz_sq, min=1e-12))

    A = torch.stack([ax, ay, az], dim=-1)
    B = torch.stack([bx, by, bz], dim=-1)
    C = torch.stack([cx, cy, cz], dim=-1)

    lattice = torch.stack([A, B, C], dim=1)   # [B, 3, 3]
    return lattice

def minimum_image_distances_torch(frac_coords: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """
    frac_coords: [M, 3]
    lattice:     [3, 3]
    returns:     [M, M]
    """
    diff = frac_coords[:, None, :] - frac_coords[None, :, :]   # [M, M, 3]
    diff = diff - torch.round(diff)                            # minimum image in fractional space
    cart_diff = diff @ lattice                                 # [M, M, 3]
    return torch.linalg.norm(cart_diff, dim=-1)                # [M, M]


def repulsion_loss_per_graph(
    expanded_pred: torch.Tensor,      # [N, K, 3] fractional coords after symmetry expansion
    op_mask: torch.Tensor,            # [N, K]
    batch_index: torch.Tensor,        # [N]
    lattice_mats: torch.Tensor,       # [B, 3, 3]
    act_func: str,
    min_dist: float = 0.5,
) -> torch.Tensor:
    """
    Returns per-graph loss: [B]

    Penalizes atom pairs whose minimum-image Cartesian distance is below min_dist.
    This is intentionally aligned with a struct-validity rule like:
        invalid if pairwise distance < 0.5
    """
    num_graphs = lattice_mats.shape[0]
    losses = expanded_pred.new_zeros(num_graphs)

    for g in range(num_graphs):
        orbit_mask = (batch_index == g)
        if not orbit_mask.any():
            continue

        coords_g = expanded_pred[orbit_mask]   # [n_orbits_g, K, 3]
        mask_g = op_mask[orbit_mask]           # [n_orbits_g, K]

        # Flatten valid expanded atoms
        frac_coords = coords_g[mask_g]         # [M, 3]

        if frac_coords.shape[0] < 2:
            losses[g] = 0.0
            continue

        lattice = lattice_mats[g]              # [3, 3]
        dist_matrix = minimum_image_distances_torch(frac_coords, lattice)

        M = dist_matrix.shape[0]
        eye = torch.eye(M, dtype=torch.bool, device=dist_matrix.device)

        # Ignore self-distances
        dist_matrix = dist_matrix.masked_fill(eye, float("inf"))

        
        
        # Penalize only pairs that are too close
        
        if act_func == "relu":
            violation = F.relu(min_dist - dist_matrix)
        elif act_func == "softplus":
            beta = 30.0 # NOTE Move later
            violation = F.softplus(min_dist - dist_matrix, beta=beta)

        # Remove diagonal and keep only actual violations
        viol = violation[~eye]
        positive_viol = viol[viol > 0]

        if positive_viol.numel() > 0:
            losses[g] = positive_viol.mean()
        else:
            losses[g] = 0.0

    return losses