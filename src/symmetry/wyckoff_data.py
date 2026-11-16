import torch
from pathlib import Path
from src.symmetry.wyckoff_vocab import build_wyckoff_dict, flatten_with_sg
from src.symmetry.wyckoff_masks import build_wyckoff_pos_mask_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WYCKOFF_DATA_PATH = PROJECT_ROOT / "data" / "symmetry" / "wyckoff_data.pt"


def load_or_create_wyckoff_data():
    if WYCKOFF_DATA_PATH.exists():
        return torch.load(WYCKOFF_DATA_PATH, weights_only=False)

    print(f"Creating Wyckoff data: {WYCKOFF_DATA_PATH}")

    wyckoff_dict, _ = build_wyckoff_dict(n_groups=230, reverse=True)

    flat_pairs, index_dict = flatten_with_sg(wyckoff_dict)

    wyckoff_pos_mask = build_wyckoff_pos_mask_dict()

    data = {
        "wyckoff_dict": wyckoff_dict,
        "flat_pairs": flat_pairs,
        "index_dict": index_dict,
        "wyckoff_pos_mask": wyckoff_pos_mask,
    }

    WYCKOFF_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, WYCKOFF_DATA_PATH)

    return data


