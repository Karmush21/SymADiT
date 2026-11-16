import numpy as np
from tqdm import tqdm
from pyxtal import pyxtal
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from src.evaluation_utils.cif_utils import canonicalize_structure
from src.symmetry.wyckoff_data import build_wyckoff_dict, flatten_with_sg

WYCKOFF_DICT, _ = build_wyckoff_dict(n_groups=230, reverse=True)
FLAT_PAIRS, INDEX_DICT = flatten_with_sg(WYCKOFF_DICT)





def build_training_data_from_csv(dataset_cif_list, cell_representation = "conventional", symprec=1e-3, angle_tolerance=10.0):
    """Convert CIF strings to pymatgen conventional-cell Structure objects."""
    print(f'Using cell representation: {cell_representation}')
    out = []
    for i, cif_str in enumerate(
        tqdm(dataset_cif_list, total=len(dataset_cif_list), desc="Loading Training CIFs")
    ):
        try:
            s = Structure.from_str(cif_str, fmt="cif") 
            
            # NOTE Sanity check that mp20 csv file is primitive 
            #sga = SpacegroupAnalyzer(s, symprec=1e-2)
            #primitive = sga.get_primitive_standard_structure()
            #is_primitive = len(primitive) == len(s)
            #print(is_primitive)

            if cell_representation == "primitive":
                
                # NOTE FOr mp20, this is not needed, these are the asme
                #sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance)
                #primitive = sga.get_primitive_standard_structure()
                
                out.append(s)
            
            elif cell_representation == "conventional": # NOTE When using your method you want conventinal
                # NOTE Converting things to conventional because pyxtal conversion also does that
                # NOTE Make this as a flag in the function, which one to return
                sga = SpacegroupAnalyzer(
                    s,
                    symprec=symprec,
                    angle_tolerance=angle_tolerance,
                )

                conventional = sga.get_conventional_standard_structure()
                
                out.append(conventional)
            
            elif cell_representation == "canonicalize":
                out.append(canonicalize_structure(s))
            
            else:
                raise ValueError("Wrong Rep.")
                

        except Exception as e:
            raise RuntimeError(f"Failed parsing/converting CIF at index {i}") from e

    return out

def build_training_materials_from_dataloader(
    dataset,
    show_error_examples: int = 5,
):
    goods = []
    bads = []

    for data in tqdm(dataset, desc="Building training structures"):

        try:
            spacegroup = int(data.sg_number)

            wyckoff_indices = data.wyckoff_ids.detach().cpu().tolist()
            wyckoff_sites = [FLAT_PAIRS[int(i)] for i in wyckoff_indices]
            wyckoff_list = [s.split("_", 1)[1] for s in wyckoff_sites]

            x_cell_free_dof = np.asarray(data.x_cell_free_dof, dtype=float)
            x_pos_free_dof = np.asarray(data.x_pos_free_dof, dtype=float)

            x_pyxtal = np.concatenate((x_cell_free_dof, x_pos_free_dof), axis=0)

            atom_types = np.asarray(data.atom_types, dtype=np.int64)
            atom_list = z_to_symbol(atom_types)

            c_gt = pyxtal()
            c_gt.from_spg_wps_rep(spacegroup, wyckoff_list, x_pyxtal, atom_list)

            pmg_gt = c_gt.to_pymatgen()

            goods.append(pmg_gt)

        except Exception as e:
            bads.append(str(e))

    if bads:
        print(f"[build_training_materials] failures: {len(bads)} / {len(dataset)}")
        for ex in bads[:show_error_examples]:
            print(" example error:", ex)

    return goods

# Face-centered (F)
FACE_CENTERED_SG = {
    22, 42, 43, 69, 70,
    196, 202, 203, 209, 210,
    216, 219, 225, 226, 227, 228
}

# Body-centered (I)
BODY_CENTERED_SG = {
    23, 24, 44, 45, 46,
    71, 72, 73, 74,
    79, 80, 82, 87, 88,
    97, 98, 107, 108, 109, 110,
    119, 120, 121, 122,
    139, 140, 141, 142,
    197, 199, 204, 206,
    211, 214, 217, 220, 229, 230
}

# Base-centered (A, B, or C)
BASE_CENTERED_SG = {
    5, 8, 9, 12, 15,
    20, 21, 35, 36, 37,
    38, 39, 40, 41,
    63, 64, 65, 66, 67, 68
}

# Rhombohedral (R)
RHOMBOHEDRAL_SG = {
    146, 148, 155, 160, 161, 166, 167
}

def centering_from_sg(sg):
    # crude but works if you define the sets
    if sg in RHOMBOHEDRAL_SG:
        return 3
    elif sg in FACE_CENTERED_SG:
        return 4
    elif sg in BODY_CENTERED_SG:
        return 2
    elif sg in BASE_CENTERED_SG:
        return 2
    return 1

def build_train_stats(dataset):
    """
    Per-material stats (aligned lists).
    Each index i corresponds to one material.
    """
    train_stats = {
        "spacegroup": [],          # [N]
        "num_orbits": [],          # [N]
        "num_atoms": [],           # [N]
        "num_atoms_prim": [],

        # per-material lists
        "atom_types": [],          # [N] of List[int] (orbit-rep Zs, in your dataset representation)
        "atom_types_unique": [],   # [N] of List[int]
        "wyckoff_types": [],       # [N] of List[str] (e.g. ["4a","8c",...], one per orbit)
        "volume": [],
        #"wyckoff_letters": [],     # [N] of List[str] (e.g. ["a","c",...])
        #"wyckoff_mults": [],       # [N] of List[int] (e.g. [4,8,...])
        "full_cell_rad": []
    }

    for data in tqdm(dataset, desc="Collecting train stats"):
        sg = int(data.sg_number)
        na = int(data.num_atoms_conv)
        no = int(data.num_orbits)
        vol = int(data.volume)
        full_cell_rad = data.full_cell_rad.tolist()

        m = centering_from_sg(sg)
        num_atoms_primitive = na // m
        train_stats["num_atoms_prim"].append(num_atoms_primitive)

        train_stats["spacegroup"].append(sg)
        train_stats["num_atoms"].append(na) # NOTE This is number of atoms in convenintal unit cell
        train_stats["num_orbits"].append(no)
        train_stats["volume"].append(vol)

        # --- wyckoff per orbit (material-level) ---
        wyckoff_ids = data.wyckoff_ids.detach().cpu().tolist()
        wyckoff_types = [FLAT_PAIRS[int(idx)].split("_", 1)[1] for idx in wyckoff_ids]  # e.g. "4a"
        train_stats["wyckoff_types"].append(wyckoff_types)

        # derived convenience: letters + multiplicities
        #train_stats["wyckoff_letters"].append([w[-1] for w in wyckoff_types])
        #train_stats["wyckoff_mults"].append([int(w[:-1]) for w in wyckoff_types])

        # --- atom types per orbit (material-level) ---
        z = np.asarray(list(data.atom_types), dtype=np.int64).tolist()
        train_stats["atom_types"].append(z)
        train_stats["atom_types_unique"].append(sorted(set(z)))

        train_stats["full_cell_rad"].append(full_cell_rad)


    return train_stats


#-------------------------------Parallel------------------------------------------------
import multiprocessing as mp
# Global dataset handle for worker access
_G_DATASET = None


def _init_training_worker(dataset):
    global _G_DATASET
    _G_DATASET = dataset

def collect_training_payloads(dataset):
    payloads = []
    bads = []

    for idx, data in enumerate(tqdm(dataset, desc="Collecting train payloads")):
        try:
            payload = {
                "idx": idx,
                "spacegroup": int(data.sg_number),
                "wyckoff_indices": data.wyckoff_ids.detach().cpu().tolist(),
                "x_cell_free_dof": np.asarray(data.x_cell_free_dof, dtype=float),
                "x_pos_free_dof": np.asarray(data.x_pos_free_dof, dtype=float),
                "atom_types": np.asarray(data.atom_types, dtype=np.int64),
            }
            payloads.append(payload)
        except Exception as e:
            bads.append(f"idx={idx}: {e}")

    return payloads, bads

def _build_one_training_structure_from_payload(payload):
    try:
        spacegroup = payload["spacegroup"]

        wyckoff_indices = payload["wyckoff_indices"]
        wyckoff_sites = [FLAT_PAIRS[int(i)] for i in wyckoff_indices]
        wyckoff_list = [s.split("_", 1)[1] for s in wyckoff_sites]

        x_cell_free_dof = payload["x_cell_free_dof"]
        x_pos_free_dof = payload["x_pos_free_dof"]
        x_pyxtal = np.concatenate((x_cell_free_dof, x_pos_free_dof), axis=0)

        atom_types = payload["atom_types"]
        atom_list = z_to_symbol(atom_types)

        c_gt = pyxtal()
        c_gt.from_spg_wps_rep(spacegroup, wyckoff_list, x_pyxtal, atom_list)

        pmg_gt = c_gt.to_pymatgen()

        return True, (c_gt, pmg_gt) # NOTE Change this to return both structures

    except Exception as e:
        return False, f'idx={payload.get("idx", "unknown")}: {e}'




def build_training_materials_from_dataloader_parallel(
    dataset,
    num_workers=8,
    chunksize=10,
    use_spawn=True,
    show_error_examples=5,
):
    payloads, collect_bads = collect_training_payloads(dataset)

    goods = []
    bads = list(collect_bads)

    ctx = mp.get_context("spawn" if use_spawn else "fork")

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.imap(
            _build_one_training_structure_from_payload,
            payloads,
            chunksize=chunksize,
        )

        for ok, payload in tqdm(results, total=len(payloads), desc="Building training structures (Parallel)"):
            if ok:
                goods.append(payload)
            else:
                bads.append(payload)

    if bads:
        print(f"[build_training_materials] failures: {len(bads)} / {len(dataset)}")
        for ex in bads[:show_error_examples]:
            print(" example error:", ex)

    return goods