from pathlib import Path
from tqdm import tqdm
import numpy as np
from pymatgen.core import Structure
from src.evaluation_utils.validity import crystal_validity
from pyxtal import pyxtal
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from src.evaluation_utils.cif_utils import get_absurd_reason, canonicalize_structure  
from src.symmetry.cell import cell_mask_from_sg
from src.symmetry.wyckoff_masks import wyckoff_mask

@dataclass
class CrystalSymInfo:
    # Space group
    spacegroup: int

    # lattice (global)
    full_cell_rad: List[List]          # (6,) radians/whatever you currently store
    
    #cell_mask: np.ndarray          # mask array
    #dof_cell: int                  # scalar

    # per-site (local)
    #pos_full: np.ndarray           # (num_orbits, 3)
    #pos_mask: np.ndarray           # (num_orbits, 3)
    wyckoff_sites: List[str]       # len = num_orbits
    #dof_pos: np.ndarray            # (num_orbits,) each orbit dof count

    # counts
    num_orbits: int
    num_atoms: int
    num_atoms_prim: int

    # chemistry
    atom_types: List         # (num_orbits,) Z per orbit representative
    atom_types_unique: List

    # pyxtal reconstruction dofs
    #x_cell_free_dof: np.ndarray    # (cell_dof,) in degrees (as your code)
    #x_pos_free_dof: np.ndarray     # (sum(dof_pos),)

    # misc
    volume: float

    #full_cell_rad: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convenience for your old code paths."""
        d = asdict(self)
        # dataclasses leaves numpy arrays as-is; that's fine
        return d





# NOTE use the function that you have in preprocessing instead
def build_crystal_pyxtal(c, return_pym_full: bool = False):
    pmg_full = c.to_pymatgen()
    pmg_prim = pmg_full.get_primitive_structure() # NOTE This is always the conventional

    
    # -------- lattice (global) --------
    free_cell_deg = np.asarray(c.lattice.encode(), dtype=float)      # minimal dofs, degrees
    full_cell_rad = np.array(c.lattice.get_para(), dtype=float)      # full params (your training rep)
    

    cell_mask = cell_mask_from_sg(c.group.number)
    dof_cell = int(np.sum(cell_mask == 0))  # <- make it a Python int

    num_orbits = int(len(c.atom_sites))
    num_atoms = int(np.asarray(c.numIons.sum()).item())  # conventional cell

    num_atoms_prim = len(pmg_prim.atomic_numbers) # primitive

    # -------- per-site (local) --------
    wyckoffs_list: List[str] = []
    atom_types_list: List[int] = []
    full_pos_list: List[np.ndarray] = []
    pos_mask_list: List[np.ndarray] = []

    x_pyxtal_pos_dof_list: List[np.ndarray] = []
    dof_list: List[int] = []

    for site in c.atom_sites:
        wp = site.wp
        wp_rep_str = wp.ops[0].as_xyz_str()

        num_dof, wy_mask = wyckoff_mask(wp_rep_str)

        wyckoffs_list.append(f"{wp.multiplicity}{wp.letter}")
        dof_list.append(int(num_dof))

        el = str(site.specie)
        atom_types_list.append(int(Element(el).Z))

        x_pyxtal_free_pos = wp.get_free_xyzs(site.position)
        x_pyxtal_pos_dof_list.append(np.asarray(x_pyxtal_free_pos, dtype=float))

        full_pos_list.append(np.asarray(site.position, dtype=float))
        pos_mask_list.append(np.asarray(wy_mask, dtype=float))

    pos_full = np.asarray(full_pos_list, dtype=float)          # (num_orbits, 3)
    pos_mask = np.asarray(pos_mask_list, dtype=float)          # (num_orbits, 3)

    atom_types = np.asarray(atom_types_list, dtype=int)        # (num_orbits,)
    atom_types_unique = np.unique(atom_types)
    dof_pos = np.asarray(dof_list, dtype=int)                  # (num_orbits,)

    volume = float(c.lattice.volume)

    # Flatten pos dofs (pyxtal uses concatenated free xyzs)
    any_nonempty = any(arr.size > 0 for arr in x_pyxtal_pos_dof_list)
    if any_nonempty:
        x_pos_free_dof = np.concatenate([arr.ravel() for arr in x_pyxtal_pos_dof_list if arr.size > 0], axis=0)
    else:
        x_pos_free_dof = np.array([], dtype=float)

    # Sanity
    assert int(np.sum(pos_mask).item()) == int(len(x_pos_free_dof)), \
        f"pos_mask sum {np.sum(pos_mask)} != len(x_pos_free_dof) {len(x_pos_free_dof)}"
    assert int(np.sum(dof_pos)) == int(len(x_pos_free_dof)), \
        f"sum(dof_pos) {np.sum(dof_pos)} != len(x_pos_free_dof) {len(x_pos_free_dof)}"

    # --- You had system_id / bravais_id in your dict but not defined here ---
    # Keep your existing logic; I'm assuming you already compute these outside.
    # If they are globals, this will work as-is.
    sym = CrystalSymInfo(
        spacegroup=int(c.group.number),

        full_cell_rad=full_cell_rad.tolist(),
        
        #cell_mask=np.asarray(cell_mask, dtype=float),
        #dof_cell=dof_cell,

        #pos_full=pos_full,
        #pos_mask=pos_mask,
        wyckoff_sites=wyckoffs_list,
        #dof_pos=dof_pos,

        num_orbits=num_orbits,
        num_atoms=num_atoms, # Conventional 
        num_atoms_prim=num_atoms_prim,

        atom_types=atom_types.tolist(),
        atom_types_unique=atom_types_unique.tolist(),

        #x_cell_free_dof=free_cell_deg,
        #x_pos_free_dof=x_pos_free_dof,

        volume=volume,
    )

    if return_pym_full:
        return sym, pmg_full
    return sym


def build_new_stats(cif_source):
    if isinstance(cif_source, (str, Path)):
        cif_dir = Path(cif_source)
        all_cifs = sorted(cif_dir.glob("*.cif"))
    else:
        all_cifs = sorted(Path(p) for p in cif_source)

    new_stats = {
        "spacegroup": [],        # [N]
        "num_orbits": [],        # [N]
        "num_atoms": [],         # [N]
        "num_atoms_prim": [],         # [N]
        "atom_types": [],        # [N] of List[int]
        "atom_types_unique": [], # [N] of List[int]
        "wyckoff_types": [],     # [N] of List[str]
        "volume": [],            # [N]
        "full_cell_rad": []
    }

    failed = []

    for sample_idx, p in enumerate(tqdm(all_cifs, desc="Collect stats from pyxtal CIFs", total=len(all_cifs))):
        try:
            xtal = pyxtal()
            xtal.from_seed(str(p), tol=1e-2) # NOTE This will work for pymatgen cif files from e.g. ADiT? 

            sym = build_crystal_pyxtal(xtal, return_pym_full=False)

            new_stats["spacegroup"].append(int(sym.spacegroup))
            new_stats["num_orbits"].append(int(sym.num_orbits))
            new_stats["num_atoms"].append(int(sym.num_atoms))
            new_stats["num_atoms_prim"].append(int(sym.num_atoms_prim))
            new_stats["atom_types"].append(sym.atom_types)
            new_stats["atom_types_unique"].append(sym.atom_types_unique)
            new_stats["wyckoff_types"].append(list(sym.wyckoff_sites))
            new_stats["volume"].append(float(sym.volume))
            new_stats["full_cell_rad"].append(sym.full_cell_rad)

        except Exception as e:
            failed.append((p, e))

    return new_stats



def structure_to_dict(structure, sample_idx):
    return {
        "atom_types": np.array([site.specie.Z for site in structure]),
        "lengths": np.array(structure.lattice.abc),
        "angles": np.array(structure.lattice.angles),
        "frac_coords": structure.frac_coords,
        "pymatgen_crys": structure,
        "sample_idx": sample_idx,
    }


def load_structures_from_cifs(
    cif_source,
    cell_representation,
    strict=False,
):
    """
    Load and evaluate all CIFs, including files marked as absurd.

    Returns:
        structures:
            Successfully parsed and converted structures.

        validity_results:
            Validity result corresponding one-to-one with `structures`.

        absurd_paths:
            Paths whose filenames are marked as absurd.
            These paths are logged but are not skipped.

        failed_parse:
            Files that could not be parsed, standardized, or evaluated.
    """
    if isinstance(cif_source, (str, Path)):
        cif_dir = Path(cif_source)
        all_cifs = sorted(cif_dir.glob("*.cif"))
    else:
        all_cifs = sorted(Path(p) for p in cif_source)

    # Detect absurd files for logging only.
    # Do not remove them.
    absurd_paths = [
        p for p in all_cifs
        if get_absurd_reason(p) is not None
    ]

    structures = []
    validity_results = []
    failed_parse = []

    for source_idx, path in enumerate(
        tqdm(
            all_cifs,
            desc="Parsing CIFs",
            total=len(all_cifs),
        )
    ):
        try:
            structure = Structure.from_file(str(path))

            sga = SpacegroupAnalyzer(
                structure,
                symprec=1e-2,
                angle_tolerance=10,
            )

            if cell_representation == "primitive":
                structure = sga.get_primitive_standard_structure()

            elif cell_representation == "conventional":
                structure = sga.get_conventional_standard_structure()

            elif cell_representation == "canonicalize":
                structure = canonicalize_structure(structure)

            else:
                raise ValueError(
                    f"Unknown representation: {cell_representation}"
                )

            if structure is None:
                raise ValueError(
                    f"Cell conversion returned None for {path}"
                )

            crys_dict = structure_to_dict(
                structure,
                sample_idx=source_idx,
            )

            # Keep False because you want to evaluate the placeholder's
            # actual structural and compositional validity.
            result = crystal_validity(crys_dict)


        except Exception as exc:
            if strict:
                raise RuntimeError(
                    f"Failed to load or evaluate CIF: {path}"
                ) from exc

            failed_parse.append((path, exc))
            continue

        # Append these together only after every operation succeeds,
        # preserving one-to-one correspondence.
        structures.append(structure)

        validity_results.append(
            {
                "sample_idx": source_idx,
                "path": path,
                "filename_absurd": get_absurd_reason(path) is not None,
                "filename_absurd_reason": get_absurd_reason(path),
                **result,
            }
        )

    return (
        structures,
        validity_results,
        absurd_paths,
        failed_parse,
    )


#-------------------Parallel things------------------

import multiprocessing as mp

def _build_one_new_stat(p):
    """
    Worker for one CIF path.
    Returns (ok, payload)
    """
    try:
        p = Path(p)

        xtal = pyxtal()
        xtal.from_seed(str(p), tol=1e-2)

        sym = build_crystal_pyxtal(xtal, return_pym_full=False)

        row = {
            "spacegroup": sym.spacegroup,
            "num_orbits": sym.num_orbits,
            "num_atoms": sym.num_atoms,
            "num_atoms_prim": sym.num_atoms_prim,
            "atom_types": sym.atom_types,
            "atom_types_unique": sym.atom_types_unique,
            "wyckoff_types": sym.wyckoff_sites,
            "volume": sym.volume,
            "full_cell_rad": sym.full_cell_rad
        }
        return True, row

    except Exception as e:
        return False, (str(p), repr(e))


def build_new_stats_parallel(cif_source, num_workers=8, chunksize=10, use_spawn=True):
    if isinstance(cif_source, (str, Path)):
        cif_dir = Path(cif_source)
        all_cifs = sorted(cif_dir.glob("*.cif"))
    else:
        all_cifs = sorted(Path(p) for p in cif_source)

    new_stats = {
        "spacegroup": [],
        "num_orbits": [],
        "num_atoms": [],
        "num_atoms_prim": [],
        "atom_types": [],
        "atom_types_unique": [],
        "wyckoff_types": [],
        "volume": [],
        "full_cell_rad": []
    }

    failed = []

    ctx = mp.get_context("spawn" if use_spawn else "fork")

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.imap(_build_one_new_stat, all_cifs, chunksize=chunksize)

        for ok, payload in tqdm(
            results,
            desc="Collect stats from pyxtal CIFs (Parallel)",
            total=len(all_cifs),
        ):
            if ok:
                new_stats["spacegroup"].append(payload["spacegroup"])
                new_stats["num_orbits"].append(payload["num_orbits"])
                new_stats["num_atoms"].append(payload["num_atoms"])
                new_stats["num_atoms_prim"].append(payload["num_atoms_prim"])
                new_stats["atom_types"].append(payload["atom_types"])
                new_stats["atom_types_unique"].append(payload["atom_types_unique"])
                new_stats["wyckoff_types"].append(payload["wyckoff_types"])
                new_stats["volume"].append(payload["volume"])
                new_stats["full_cell_rad"].append(payload["full_cell_rad"])
            else:
                failed.append(payload)

    return new_stats, failed



def _load_one_cif(payload):
    """
    Process one CIF file.

    Returns:
        (True, (structure, validity_row))
        or
        (False, (error_type, path, error_message))
    """
    (
        source_idx,
        cell_representation,
        path,
        symprec,
        angle_tolerance,
    ) = payload

    path = Path(path)

    try:
        structure = Structure.from_file(str(path))

        sga = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        if cell_representation == "primitive":
            structure = sga.get_primitive_standard_structure()

        elif cell_representation == "conventional":
            structure = sga.get_conventional_standard_structure()

        elif cell_representation == "canonicalize":
            structure = canonicalize_structure(
                structure,
                symprec=symprec,
            )

        else:
            raise ValueError(
                f"Unknown representation: {cell_representation}"
            )

        if structure is None:
            raise ValueError(
                f"Cell conversion returned None for {path}"
            )

    except Exception as exc:
        return False, (
            "parse_or_conversion_error",
            path,
            repr(exc),
        )

    try:
        crys_dict = structure_to_dict(
            structure,
            sample_idx=source_idx,
        )

        # Evaluate the actual placeholder structure rather than
        # automatically declaring an absurd filename invalid.
        result = crystal_validity(crys_dict)


        absurd_reason = get_absurd_reason(path)

        validity_row = {
            "sample_idx": source_idx,
            "path": path,
            "filename_absurd": absurd_reason is not None,
            "filename_absurd_reason": absurd_reason,
            **result,
        }

        return True, (
            structure,
            validity_row,
        )

    except Exception as exc:
        return False, (
            "validity_error",
            path,
            repr(exc),
        )


def load_structures_from_cifs_parallel(
    cif_source,
    cell_representation,
    strict=False,
    num_workers=8,
    chunksize=10,
    use_spawn=True,
    symprec=1e-2,
    angle_tolerance=10.0,
):
    """
    Load and evaluate all CIFs in parallel, including files whose
    filenames are marked as absurd.

    Returns:
        structures:
            Successfully parsed, converted, and evaluated structures.

        validity_results:
            Validity rows corresponding one-to-one with `structures`.

        absurd_paths:
            Paths marked as absurd by filename. These are logged but
            are not skipped.

        failed_parse:
            Files that failed parsing, cell conversion, or validity
            evaluation.
    """
    if isinstance(cif_source, (str, Path)):
        cif_dir = Path(cif_source)
        all_cifs = sorted(cif_dir.glob("*.cif"))
    else:
        all_cifs = sorted(
            Path(path)
            for path in cif_source
        )

    # Detect only for logging. Do not remove these paths.
    absurd_paths = [
        path
        for path in all_cifs
        if get_absurd_reason(path) is not None
    ]

    payloads = [
        (
            source_idx,
            cell_representation,
            path,
            symprec,
            angle_tolerance,
        )
        for source_idx, path in enumerate(all_cifs)
    ]

    structures = []
    validity_results = []
    failed_parse = []

    ctx = mp.get_context(
        "spawn" if use_spawn else "fork"
    )

    with ctx.Pool(processes=num_workers) as pool:
        results = pool.imap(
            _load_one_cif,
            payloads,
            chunksize=chunksize,
        )

        for ok, payload in tqdm(
            results,
            desc="Parsing CIFs (Parallel)",
            total=len(all_cifs),
        ):
            if ok:
                structure, validity_row = payload

                # Append together so these lists remain aligned.
                structures.append(structure)
                validity_results.append(validity_row)

            else:
                error_type, path, error_message = payload

                if strict:
                    raise RuntimeError(
                        f"Failed processing CIF: {path}\n"
                        f"Error type: {error_type}\n"
                        f"Error: {error_message}"
                    )

                failed_parse.append(
                    {
                        "path": path,
                        "error_type": error_type,
                        "error": error_message,
                    }
                )

    return (
        structures,
        validity_results,
        absurd_paths,
        failed_parse,
    )