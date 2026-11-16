import torch
from src.symmetry.atom import z_to_symbol
from typing import Dict



def build_exact_orbit_buckets(
    num_orbits: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    Group sample indices by exact number of orbits.

    Parameters
    ----------
    num_orbits
        Long tensor of shape [N]. Entry i is the number of
        orbits sampled for crystal i.

    Returns
    -------
    orbit_buckets
        Dictionary mapping:

            orbit_count -> indices of samples with that count

        Example:

            num_orbits = tensor([2, 4, 2, 3, 4])

            orbit_buckets = {
                2: tensor([0, 2]),
                3: tensor([3]),
                4: tensor([1, 4]),
            }
    """

    num_orbits = torch.as_tensor(num_orbits)

    if num_orbits.ndim != 1:
        raise ValueError(
            "num_orbits must have shape [N], but got "
            f"{tuple(num_orbits.shape)}."
        )

    if num_orbits.numel() == 0:
        raise ValueError("num_orbits cannot be empty.")

    if torch.any(num_orbits <= 0):
        raise ValueError(
            "All num_orbits values must be positive."
        )

    orbit_buckets = {}

    unique_orbit_counts = torch.unique(
        num_orbits,
        sorted=True,
    )

    for orbit_count_tensor in unique_orbit_counts:
        orbit_count = int(orbit_count_tensor.item())

        indices = torch.nonzero(
            num_orbits == orbit_count,
            as_tuple=False,
        ).flatten()

        orbit_buckets[orbit_count] = indices

    total_bucketed = sum(
        indices.numel()
        for indices in orbit_buckets.values()
    )

    if total_bucketed != num_orbits.numel():
        raise RuntimeError(
            f"Expected {num_orbits.numel()} bucketed samples, "
            f"but found {total_bucketed}."
        )

    return orbit_buckets


def sample_num_orbits_given_sg(spacegroup: torch.Tensor, sg_to_orbit_dist: dict, device=None):
    """
    spacegroup: LongTensor [B] with values in 1..230 (or subset)
    returns: LongTensor [B] sampled num_orbits
    """
    if device is None:
        device = spacegroup.device

    B = spacegroup.numel()
    out = torch.empty(B, dtype=torch.long, device=device)

    # multinomial works on CPU/GPU; your dict tensors are probably CPU -> move per SG
    # Group by unique SG to avoid looping B times
    unique_sg = torch.unique(spacegroup).tolist()

    for sg in unique_sg:
        mask = (spacegroup == sg)
        n = int(mask.sum().item())
        if n == 0:
            continue

        dist = sg_to_orbit_dist.get(int(sg), None)
        if dist is None:
            raise KeyError(f"No conditional num_orbits distribution found for sg={sg}")

        values = dist["values"].to(device=device)
        probs  = dist["probs"].to(device=device)

        # sample n draws for this sg
        idx = torch.multinomial(probs, n, replacement=True)
        out[mask] = values[idx]

    return out

@torch.no_grad()
def sample_spacegroups_and_orbits(
    spacegroups_bincount,
    sg_num_orbits_conditionals,
    num_samples,
    device,
    allowed_spacegroups=None,
    forbidden_spacegroups=None,
):
    """
    Sample space groups and their corresponding number of orbits
    without running the ODE sampler or decoder.
    """

    sg_hist = spacegroups_bincount.to(
        device=device,
        dtype=torch.float32,
    )

    num_sg_entries = sg_hist.numel()

    mask = torch.ones_like(
        sg_hist,
        dtype=torch.bool,
    )

    # Index 0 is reserved for the null CFG condition.
    mask[0] = False

    if allowed_spacegroups is not None:
        allowed = torch.as_tensor(
            allowed_spacegroups,
            device=device,
            dtype=torch.long,
        )

        if torch.any(
            (allowed < 1)
            | (allowed >= num_sg_entries)
        ):
            raise ValueError(
                "Allowed space groups must be in "
                f"[1, {num_sg_entries - 1}]."
            )

        allowed = torch.unique(allowed)

        mask[:] = False
        mask[allowed] = True

    if forbidden_spacegroups is not None:
        forbidden = torch.as_tensor(
            forbidden_spacegroups,
            device=device,
            dtype=torch.long,
        )

        if torch.any(
            (forbidden < 1)
            | (forbidden >= num_sg_entries)
        ):
            raise ValueError(
                "Forbidden space groups must be in "
                f"[1, {num_sg_entries - 1}]."
            )

        forbidden = torch.unique(forbidden)
        mask[forbidden] = False

    filtered = sg_hist.clone()
    filtered[~mask] = 0.0

    total = filtered.sum()

    if total <= 0:
        raise ValueError(
            "No valid space groups remain after filtering."
        )

    probabilities = filtered / total

    spacegroup = torch.multinomial(
        probabilities,
        num_samples,
        replacement=True,
    )

    num_orbits = sample_num_orbits_given_sg(
        spacegroup=spacegroup,
        sg_to_orbit_dist=sg_num_orbits_conditionals,
        device=device,
    )

    return spacegroup, num_orbits

def sample_wyckoffs_and_atoms(
    wy_logits,
    atom_logits,
    spacegroup,
    INDEX_DICT,
    FLAT_PAIRS,
    WYCKOFF_POS_MASK,
    wyckoff_temperature,
    atom_temperature,
    device,
):
    """
    Sequentially sample Wyckoff positions and atom types for one crystal.

    Prevents repeated zero-DoF Wyckoff positions within the same generated crystal.
    """

    wy_start_idx, wy_end_idx = INDEX_DICT[spacegroup]

    used_zero_dof_wyckoffs = set()

    global_wp_idx_list = []
    wyckoff_types = []
    wyckoff_types_clean = []
    atom_type_idx_list = []

    # More efficient than repeated FLAT_PAIRS.index(...)
    pair_to_idx = {pair: i for i, pair in enumerate(FLAT_PAIRS)}

    num_orbits = wy_logits.shape[0]

    for orbit_i in range(num_orbits):
        # -------------------------
        # 1. Sample Wyckoff position
        # -------------------------
        logits_i = wy_logits[orbit_i].clone()

        valid_mask = torch.zeros_like(logits_i, dtype=torch.bool)
        valid_mask[wy_start_idx:wy_end_idx] = True

        # Remove already-used zero-DoF Wyckoff positions
        for used_w in used_zero_dof_wyckoffs:
            pair_key = f"{spacegroup}_{used_w}"
            if pair_key in pair_to_idx:
                valid_mask[pair_to_idx[pair_key]] = False

        logits_i = logits_i.masked_fill(~valid_mask, -1e9)

        probs_i = torch.softmax(logits_i / wyckoff_temperature, dim=-1)
        sampled_wp_idx = torch.multinomial(probs_i, 1).item()

        wyckoff_type = FLAT_PAIRS[sampled_wp_idx]          # e.g. "225_4a"
        wyckoff_clean = wyckoff_type.split("_", 1)[1]      # e.g. "4a"

        # WYCKOFF_POS_MASK may use either "225_4a" or "4a" depending on your dict
        if wyckoff_type in WYCKOFF_POS_MASK:
            num_dof, _ = WYCKOFF_POS_MASK[wyckoff_type]
        else:
            num_dof, _ = WYCKOFF_POS_MASK[wyckoff_clean]

        if num_dof == 0:
            used_zero_dof_wyckoffs.add(wyckoff_clean)

        global_wp_idx_list.append(sampled_wp_idx)
        wyckoff_types.append(wyckoff_type)
        wyckoff_types_clean.append(wyckoff_clean)

        # -------------------------
        # 2. Sample atom type
        # -------------------------
        atom_logits_i = atom_logits[orbit_i].clone()
        atom_logits_i = atom_logits_i / atom_temperature

        # forbid PAD/MASK token
        atom_logits_i[0] = -1e9

        atom_probs_i = torch.softmax(atom_logits_i, dim=-1)
        sampled_atom_idx = torch.multinomial(atom_probs_i, 1).item()

        if sampled_atom_idx == 0:
            sampled_atom_idx = 1

        atom_type_idx_list.append(sampled_atom_idx)

    global_wp_idx = torch.tensor(global_wp_idx_list, device=device)
    atom_type_idx = torch.tensor(atom_type_idx_list, device=device)
    atom_types_list = z_to_symbol(atom_type_idx)

    return {
        "global_wp_idx": global_wp_idx,
        "wyckoff_types": wyckoff_types,
        "wyckoff_types_clean": wyckoff_types_clean,
        "atom_type_idx": atom_type_idx,
        "atom_types_list": atom_types_list,
    }