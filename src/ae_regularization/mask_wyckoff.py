import torch
import torch.nn.functional as F


def mask_wyckoff_logits(logits_w, sg_per_atom, index_dict):
    # logits_w: [N_atoms, num_wp_tokens]
    mask = torch.zeros_like(logits_w, dtype=torch.bool)

    for sg in sg_per_atom.unique():
        sg_int = int(sg.item())
        atom_idx = (sg_per_atom == sg)
        wy_start, wy_end = index_dict[sg_int]
        mask[atom_idx, wy_start:wy_end] = True

    return logits_w.masked_fill(~mask, -1e9)


