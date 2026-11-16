"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import itertools
import warnings
from collections import Counter
import numpy as np
import smact
from smact.screening import pauling_test


warnings.filterwarnings("ignore", category=UserWarning) 


class Crystal:
    """Evaluation object for an already constructed crystal."""

    def __init__(self, crys_array_dict: dict):
        self.crys_array_dict = crys_array_dict
        self.dict = crys_array_dict

        self.sample_idx = crys_array_dict["sample_idx"]
        self.id = crys_array_dict.get("id", self.sample_idx)

        self.frac_coords = np.asarray(crys_array_dict["frac_coords"])
        self.atom_types = np.asarray(crys_array_dict["atom_types"])
        self.lengths = np.asarray(crys_array_dict["lengths"]).squeeze()
        self.angles = np.asarray(crys_array_dict["angles"]).squeeze()

        self.spacegroup = crys_array_dict["spacegroup"]
        
        self.pyxtal_crys = crys_array_dict.get("pyxtal_crys")
        self.structure = crys_array_dict["pymatgen_crys"]

        # Always initialize these
        self.comp_valid = False
        self.struct_valid = False
        self.valid = False
        self.min_dist = np.nan
        self.invalid_reason = crys_array_dict.get("reason", "unknown")

        # If we could not build pyxtal, we stop the function here
        if self.invalid_reason != "ok":
            return

        self.get_composition()
        self.get_validity()

    def get_composition(self) -> None:
        elem_counter = Counter(self.atom_types)

        composition = [(element, elem_counter[element]) for element in sorted(elem_counter)]

        elems, counts = zip(*composition)

        counts = np.asarray(counts, dtype=int)
        counts = counts // np.gcd.reduce(counts)

        self.elems = tuple(elems)
        self.comps = tuple(counts.tolist())

    def get_validity(self) -> None:
        self.comp_valid = smact_validity(self.elems, self.comps)

        self.struct_valid, self.min_dist = structure_validity(self.structure)

        self.valid = self.comp_valid and self.struct_valid

        # NOTE We ignore about comp_valid (see paper)
        if not self.struct_valid:
            self.invalid_reason = "invalid_structure"

        else:
            self.invalid_reason = "ok"

    def __repr__(self) -> str:
        return (
            f"Crystal("
            f"valid={self.valid}, "
            f"comp_valid={self.comp_valid}, "
            f"struct_valid={self.struct_valid}, "
            f"atoms={self.atom_types}, "
            f"lengths={self.lengths}, "
            f"angles={self.angles}, "
            f"idx={self.sample_idx}, "
            f"reason={self.invalid_reason}"
            f")"
        )


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    # if len(list(itertools.product(*ox_combos))) > 1e5:
    #     return False
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False



def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix

    # mask diagonal
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))

    min_dist = dist_mat.min()

    valid = (min_dist >= cutoff) and (crystal.volume >= 0.1)

    return valid, min_dist

chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

