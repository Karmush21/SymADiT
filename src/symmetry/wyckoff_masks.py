import torch
import re
import numpy as np
from pyxtal.symmetry import Group




# Use in vae_module.py creates the mask with same shape, so we just pred now and not gt. better
def extract_masks(names, mask_dict, device):
    return torch.tensor([mask_dict[name][1] for name in names], dtype=torch.long, device=device)

_VAR_RE = re.compile(r'(?<![A-Za-z])([xyz])(?![A-Za-z])')

def wyckoff_mask(s: str):
    """
    Counts independent DOF from a Wyckoff string like:
      "x, y, z"        -> dof=3
      "x, 0, z"        -> dof=2
      "x, -x, z"       -> dof=2
      "x, y-x, 1/2"    -> dof=2  (depends on x and y)
      "0, 0, 0"        -> dof=0
    """
    parts = [p.strip() for p in s.split(",")]

    used = set()
    indep_mask = []

    for p in parts:
        vars_in_part = set(_VAR_RE.findall(p))
        new_vars = vars_in_part - used

        # 1 if this coordinate introduces at least one new independent symbol
        indep_mask.append(1 if new_vars else 0)

        used |= vars_in_part

    dof = len(used)
    return dof, np.array(indep_mask, dtype=int)


def build_wyckoff_pos_mask_dict(max_spg: int = 230):
    """
    Build a dict mapping 'sg_multletter' → mask, e.g. '225_4a' → [0,0,0].
    Uses the first representative op for each Wyckoff position.
    """
    wp_mask_dict = {}

    for sg in range(1, max_spg + 1):
        group = Group(sg)

        # group.wp (or group.Wyckoff_positions, depending on version)
        for wp in group.Wyckoff_positions:  # or group.wp
            mult = wp.multiplicity    # e.g. 4
            letter = wp.letter        # e.g. 'a'
            key = f"{sg}_{mult}{letter}"

            # Take first representative operation
            rep_str = wp.ops[0].as_xyz_str()   # e.g. 'x, 0, z'
            mask = wyckoff_mask(rep_str)

            wp_mask_dict[key] = mask

    return wp_mask_dict



def build_wyckoff_pos_mask_dict(max_spg: int = 230):
    """
    Build a dict mapping 'sg_multletter' → mask, e.g. '225_4a' → [0,0,0].
    Uses the first representative op for each Wyckoff position.
    """
    wp_mask_dict = {}

    for sg in range(1, max_spg + 1):
        group = Group(sg)

        # group.wp (or group.Wyckoff_positions, depending on version)
        for wp in group.Wyckoff_positions:  # or group.wp
            mult = wp.multiplicity    # e.g. 4
            letter = wp.letter        # e.g. 'a'
            key = f"{sg}_{mult}{letter}"

            # Take first representative operation
            rep_str = wp.ops[0].as_xyz_str()   # e.g. 'x, 0, z'
            mask = wyckoff_mask(rep_str)

            wp_mask_dict[key] = mask

    return wp_mask_dict


