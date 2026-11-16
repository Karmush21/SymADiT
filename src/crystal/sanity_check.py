import numpy as np
from pymatgen.core import Lattice, Structure
from pyxtal import pyxtal
from typing import Tuple
import networkx as nx
from pymatgen.core import Structure

def create_absurd_crystal(
    tol: float = 1e-4,
) -> Tuple[Structure, pyxtal]:
    """
    Create the same deliberately invalid crystal as both:

      1. a pymatgen Structure
      2. a PyXtal object

    The minimum interatomic distance is approximately 0.1 Å,
    so it should fail a structural-validity cutoff of 0.5 Å.
    """
    lattice = Lattice.cubic(10.0)

    species = ["Na", "Cl", "Cl"]

    frac_coords = [
        [0.00, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.02, 0.0, 0.0],
    ]

    pmg_structure = Structure(
        lattice=lattice,
        species=species,
        coords=frac_coords,
        coords_are_cartesian=False,
        validate_proximity=False,
    )

    pyxtal_structure = pyxtal()
    pyxtal_structure.from_seed(
        pmg_structure,
        tol=tol,
        backend="pymatgen",
        style="pyxtal",
    )

    return pmg_structure, pyxtal_structure


def pyxtal_to_pmg(xtal):
    return xtal.to_pymatgen()

def check_duplicate_pyxtal_reps(xtal, tol=1e-5):
    seen = set()
    for site in xtal.atom_sites:
        key = tuple(np.round(site.position / tol).astype(int))
        if key in seen:
            return False, "duplicate_pyxtal_reps", {"dup_key": key}
        seen.add(key)
    return True, "ok", {}


def check_min_dist_distance_matrix(pmg_struct, dmin_tol=0.5):
    D = np.array(pmg_struct.distance_matrix, dtype=float)
    np.fill_diagonal(D, np.inf)
    min_dist = float(np.min(D))
    i, j = np.unravel_index(np.argmin(D), D.shape)
    if min_dist < dmin_tol:
        return False, "atoms_too_close", {"min_dist": min_dist, "pair": (int(i), int(j))}
    return True, "ok", {"min_dist": min_dist}

def check_max_nearest_neighbor_distance(pmg_struct, dmax_nn_tol=6.0):
    D = np.array(pmg_struct.distance_matrix, dtype=float)
    np.fill_diagonal(D, np.inf)
    nn = np.min(D, axis=1)
    max_nn = float(np.max(nn))
    i = int(np.argmax(nn))
    j = int(np.argmin(D[i, :]))
    info = {
        "max_nearest_neighbor_dist": max_nn,
        "worst_atom": i,
        "nearest_neighbor_of_worst": j,
        "worst_atom_species": str(pmg_struct[i].specie),
        "nearest_neighbor_species": str(pmg_struct[j].specie),
        "worst_atom_frac": [float(x) for x in pmg_struct[i].frac_coords],
    }
    if max_nn > dmax_nn_tol:
        return False, "atom_too_isolated", info
    return True, "ok", info

def check_lattice_bounds(pmg_struct, a_min=1.0, a_max=80.0, vol_min=1.0, vol_max=1e6,
                         angle_min=10.0, angle_max=170.0):
    lat = pmg_struct.lattice
    a, b, c = lat.abc
    alpha, beta, gamma = lat.angles
    vol = float(lat.volume)
    info = {"a": float(a), "b": float(b), "c": float(c),
            "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            "volume": vol}
    if min(a, b, c) < a_min:
        return False, "lattice_too_small", info
    if max(a, b, c) > a_max:
        return False, "lattice_too_large", info
    if not (vol_min <= vol <= vol_max):
        return False, "volume_out_of_range", info
    if min(alpha, beta, gamma) < angle_min or max(alpha, beta, gamma) > angle_max:
        return False, "angles_out_of_range", info
    return True, "ok", info



RADII = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70,
    'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73,
    'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 2.75, 'Ca': 2.31, 'Sc': 2.11, 'Ti': 2.00, 'V': 2.00, 'Cr': 2.00,
    'Mn': 2.00, 'Fe': 2.00, 'Co': 2.00, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
    'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'Y': 2.00, 'Zr': 2.00, 'Nb': 2.00, 'Mo': 2.00,
    'Tc': 2.00, 'Ru': 2.00, 'Rh': 2.00, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
    'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    'Cs': 3.43, 'Ba': 2.68, 'La': 2.50, 'Ce': 2.48, 'Pr': 2.47, 'Nd': 2.45,
    'Sm': 2.42, 'Eu': 2.40, 'Gd': 2.38, 'Dy': 2.35, 'Er': 2.32, 'Lu': 2.27,
    'Hf': 2.16, 'Ta': 2.09, 'W': 2.02, 'Re': 1.97, 'Os': 2.00, 'Ir': 2.00,
    'Pt': 1.75, 'Au': 1.66, 'Hg': 1.55, 'Pb': 2.02, 'Bi': 2.07, 'U': 1.86
}


def minimum_image_distances(frac_coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    diff = frac_coords[:, None, :] - frac_coords[None, :, :]   # [N, N, 3]
    diff -= np.round(diff)                                     # minimum image in fractional space
    cart_diff = diff @ lattice                                 # [N, N, 3]
    return np.linalg.norm(cart_diff, axis=-1)                  # [N, N]


def build_cutoff_matrix(species: list[str], tol: float = 0.5) -> np.ndarray:
    radii = np.array([RADII.get(s, 1.8) for s in species], dtype=float)
    return radii[:, None] + radii[None, :] + tol


def classify_structure_obj(
    structure: Structure,
    supercell: tuple[int, int, int] = (2, 2, 2),
    short_distance_factor: float = 0.4,
    tol: float = 0.5,
) -> str:
    """
    Classify a pymatgen Structure directly as:
      - 'too_close'
      - 'too_far'
      - 'ok'
    """
    info = {}

    structure_sc = structure.copy()
    structure_sc.make_supercell(supercell)

    frac_coords = structure_sc.frac_coords
    lattice = structure_sc.lattice.matrix
    species = [site.specie.symbol for site in structure_sc]
    n_atoms = len(species)

    dist_matrix = minimum_image_distances(frac_coords, lattice)
    cutoff_matrix = build_cutoff_matrix(species, tol=tol)

    # too close check
    min_allowed = cutoff_matrix * short_distance_factor
    np.fill_diagonal(min_allowed, 0.0)

    too_short_mask = (dist_matrix < min_allowed) & (dist_matrix > 0.0)
    if np.any(too_short_mask):
        return False, "Soheil_too_close", info

    # connectivity graph
    adjacency_mask = (dist_matrix < cutoff_matrix) & (dist_matrix > 0.0)

    G = nx.Graph()
    G.add_nodes_from(range(n_atoms))

    i_idx, j_idx = np.where(np.triu(adjacency_mask, k=1))
    G.add_edges_from(zip(i_idx, j_idx))

    if G.number_of_edges() == 0:
        return False, "Soheil_too_far", info

    if not nx.is_connected(G):
        return False, "Soheil_too_far", info

    return True, "ok", info
#---------------------_Soheil check-----------------------------
def sanity_check_crystal(c_pred):
    
    if c_pred is None:
        pyg_pred, c_pred = create_absurd_crystal()
        
        return False, "could_not_build_pyxtal", pyg_pred, c_pred
    

    frac_tol_rep = 1e-5
    dmin_tol     = 0.5
    dmax_nn_tol  = 6.0


    pyg_pred = pyxtal_to_pmg(c_pred)

    #all_info = {}

    ok, reason, info = check_duplicate_pyxtal_reps(c_pred, tol=frac_tol_rep)
    if not ok:
        absurd_pyg_pred, absurd_c_pred = create_absurd_crystal()
        return False, reason, absurd_pyg_pred, absurd_c_pred
    #all_info["duplicate_reps"] = info


    ok, reason, info = check_max_nearest_neighbor_distance(pyg_pred, dmax_nn_tol=dmax_nn_tol)
    if not ok:
        absurd_pyg_pred, absurd_c_pred = create_absurd_crystal()
        return False, reason, absurd_pyg_pred, absurd_c_pred
    #all_info["max_nn"] = info

    ok, reason, info = check_lattice_bounds(pyg_pred)
    if not ok:
        absurd_pyg_pred, absurd_c_pred = create_absurd_crystal()
        return False, reason, absurd_pyg_pred, absurd_c_pred
    #all_info["lattice"] = pyg_pred

    
    return True, "ok", pyg_pred, c_pred 