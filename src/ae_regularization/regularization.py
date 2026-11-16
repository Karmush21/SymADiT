import torch
import torch.nn.functional as F


def pairwise_pbc_distances(frac_coords: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """
    frac_coords: [M, 3]
    lattice:     [3, 3]
    returns:     [M, M]
    """
    diff_frac = frac_coords[:, None, :] - frac_coords[None, :, :]    # [M, M, 3]
    diff_frac = torch.remainder(diff_frac + 0.5, 1.0) - 0.5          # minimum image

    diff_cart = diff_frac @ lattice                                  # [M, M, 3]
    dist = torch.linalg.norm(diff_cart, dim=-1)                      # [M, M]
    return dist






