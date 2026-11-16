import itertools
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
import smact
from smact.screening import pauling_test
from pymatgen.core.structure import Structure

'''
    Validty functions from crystal genreation, in order to check validity
'''

# --- same mapping you used ---
chemical_symbols = [
    "X","H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og",
]


def smact_validity(
    comp: Tuple[int, ...],
    count: Tuple[int, ...],
    use_pauling_test: bool = True,
    include_alloys: bool = True,
) -> bool:
    """
    Compositional validity using SMACT charge-balance + (optional) Pauling electronegativity test.

    Args:
        comp: tuple of atomic numbers (e.g., (8, 14) for O, Si)
        count: tuple of reduced integer stoichiometries aligned with comp (e.g., (2, 1))
    """
    elem_symbols = tuple([chemical_symbols[int(elem)] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]

    # unary crystals are always "valid" compositionally
    if len(set(elem_symbols)) == 1:
        return True

    # allow pure-metal alloys if requested
    if include_alloys:
        is_metal_list = [sym in smact.metals for sym in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = int(np.max(count))

    # guard against combinatorial explosion
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 1e7:
        return False

    for ox_states in itertools.product(*ox_combos):
        stoichs = [(int(c),) for c in count]
        cn_e, _ = smact.neutral_ratios(ox_states, stoichs=stoichs, threshold=threshold)

        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_ok = pauling_test(ox_states, electronegs)
                except TypeError:
                    # missing electronegativity data -> assume OK
                    electroneg_ok = True
            else:
                electroneg_ok = True

            if electroneg_ok:
                return True

    return False


def structure_validity(structure: Structure, cutoff: float = 0.5) -> bool:
    """
    Structural validity: rejects if atoms are closer than cutoff (Å) or volume is unrealistically small.

    Args:
        structure: pymatgen Structure
        cutoff: minimum allowed interatomic distance (Å)
    """
    if structure is None:
        return False

    if structure.volume < 0.1:
        return False

    dist_mat = structure.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(dist_mat.shape[0]) * (cutoff + 10.0))
    return bool(dist_mat.min() >= cutoff)





def crystal_validity(
    crys_array_dict: Dict[str, Any],
    *,
    cutoff: float = 0.5,
) -> Dict[str, bool]:
    structure = crys_array_dict["pymatgen_crys"]
    atom_types = np.asarray(
        crys_array_dict["atom_types"],
        dtype=int,
    )

    # Reduce composition to its simplest integer ratio.
    elem_counter = Counter(atom_types.tolist())
    elems = tuple(sorted(elem_counter))

    counts = np.asarray(
        [elem_counter[element] for element in elems],
        dtype=int,
    )
    counts = counts // np.gcd.reduce(counts)
    comps = tuple(counts.tolist())

    comp_valid = bool(
        smact_validity(
            elems,
            comps,
        )
    )

    # Your structure_validity function returns:
    # (valid_boolean, minimum_distance)
    struct_valid = structure_validity(structure, cutoff=cutoff)
    struct_valid = bool(struct_valid)

    valid = comp_valid and struct_valid

    return {
        "comp_valid": comp_valid,
        "struct_valid": struct_valid,
        "valid": valid,
    }
