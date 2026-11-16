from pyxtal.symmetry import Group
from typing import Dict, List, Tuple

def build_wyckoff_dict(n_groups: int = 230, reverse: bool = True):
    index = 0
    """Return {space_group_number: [list of 'multiplicity+letter']}
    
    Format:
    {
    225: ["4a", "4b", "8c", "24e", ...],
    226: ["4a", "4b", ...],
    }
    
    """
    wyckoff_dict = {}
    for sg in range(1, n_groups + 1):
        try:
            g = Group(sg)
            labels = [wp.get_label() for wp in g]
            if reverse:
                labels.reverse()
            index += len(labels)
            wyckoff_dict[sg] = labels
        except Exception as e:
            print(f"Skipping SG {sg}: {e}")
    return wyckoff_dict, index


def flatten_with_sg(wyckoff_dict: Dict[int, List[str]], sep: str = "_") -> Tuple[List[str], Dict[int, Tuple[int, int]]]:
    flat_pairs: List[str] = []
    index_dict: Dict[int, Tuple[int, int]] = {}
    offset = 0
    for sg in sorted(wyckoff_dict):  # ensure 1..230 order
        labels = wyckoff_dict[sg]
        start = offset
        flat_pairs.extend([f"{sg}{sep}{lab}" for lab in labels])
        offset += len(labels)
        index_dict[sg] = (start, offset)  # half-open [start, end)
    return flat_pairs, index_dict





def build_wyckoff_dict_str_free(max_sg=230):
    '''
    {
    225: {
        "4a": "0, 0, 0",
        "24e": "x, 0, 0",
    }
}
    
    '''
    wyckoff_dict = {}

    for sg in range(1, max_sg + 1):
        g = Group(sg)
        sg_dict = {}

        for wp in g:
            label = f"{wp.multiplicity}{wp.letter}"
            rep_str = wp.ops[0].as_xyz_str()

            sg_dict[label] = rep_str

        wyckoff_dict[sg] = sg_dict

    return wyckoff_dict


def build_wyckoff_dict_from_data(flat_pairs, index_dict):
    wyckoff_dict = {}

    for sg, (start, end) in index_dict.items():
        wyckoff_dict[sg] = [
            token.split("_", 1)[1]
            for token in flat_pairs[start:end]
        ]

    return wyckoff_dict