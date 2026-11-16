"""
Crystal preprocessing with optional multiprocessing.

Usage:
    debug=True
        -> sequential processing
        -> no multiprocessing
        -> exceptions propagate normally
        -> good for breakpoints/debugging

    debug=False
        -> multiprocessing
        -> errors are caught and written to bad_cifs.csv

You can optionally select:
    - particular material IDs
    - a starting index
    - a limited number of structures
"""

import faulthandler
import multiprocessing as mp
import os
import time
import traceback
import warnings
import numpy as np
import pandas as pd

from pymatgen.core import Element, Structure
from pyxtal import pyxtal
from tqdm import tqdm

from src.symmetry.cell import cell_mask_from_sg
from src.symmetry.wyckoff_masks import wyckoff_mask
from src.symmetry.wyckoff_data import load_or_create_wyckoff_data
from src.symmetry.wyckoff_affine import wp_representative_to_A_b



# -------------------------------------------------------------------------
# Warnings
# -------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\.io\.cif")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pymatgen\.core\.structure")
faulthandler.enable()



# =========================================================================
# Worker globals
# =========================================================================
#
# These are initialized once per multiprocessing worker.
#
# In debug mode, _init_worker() is simply called in the main process instead.
# Therefore the exact same _process_one_task() is used in both modes.
# =========================================================================

_G_TOKEN_TO_ID = None
_G_UNK_ID = None


def _init_worker(token_to_id, unk_id):
    """Initialize Wyckoff token information."""

    global _G_TOKEN_TO_ID, _G_UNK_ID

    _G_TOKEN_TO_ID = token_to_id
    _G_UNK_ID = unk_id


# =========================================================================
# Crystal construction
# =========================================================================

def build_crystal_pyxtal(cif_str: str, tol: float = 1e-2, return_pym_full: bool = False):

    # ---------------------------------------------------------------------
    # CIF -> PyXtal
    # ---------------------------------------------------------------------

    pmg_seed = Structure.from_str(cif_str, fmt="cif")

    c = pyxtal()
    c.from_seed(pmg_seed, tol=tol)

    pmg_full = c.to_pymatgen()


    # ---------------------------------------------------------------------
    # Lattice information
    # ---------------------------------------------------------------------

    # Minimal lattice DOFs.
    free_cell_deg = np.asarray(c.lattice.encode(), dtype=float)

    # Full cell representation that we train on.
    full_cell_rad = np.array(c.lattice.get_para(), dtype=float)
    full_cell_mask = cell_mask_from_sg(c.group.number)


    num_dof_cell = np.asarray(np.sum(full_cell_mask == 1))


    # ---------------------------------------------------------------------
    # Global information
    # ---------------------------------------------------------------------
    num_orbits = np.asarray(len(c.atom_sites), int)

    num_atoms_conv = np.asarray(c.numIons.sum(), int)


    # ---------------------------------------------------------------------
    # Per-orbit/site information
    # ---------------------------------------------------------------------

    wyckoffs_list = []
    atom_types_list = []

    full_rep_list = []
    full_rep_mask_list = []

    x_pyxtal_pos_dof_list = []
    num_dof_rep_list = []

    # All equivalent representatives for each orbit
    all_reps_list = []

    # Corresponding symmetry operations
    all_R_list = []
    all_t_list = []

    A_list = []
    b_list = []


    # ---------------------------------------------------------------------
    # Loop over symmetry orbits
    # ---------------------------------------------------------------------

    for site in c.atom_sites:

        wp = site.wp

        R_list = []
        t_list = []

        # -------------------------------------------------------------
        # Symmetry operations belonging to this Wyckoff position
        # -------------------------------------------------------------

        for op in wp.ops:

            affine = op.affine_matrix

            R = affine[:3, :3]
            t = affine[:3, 3]

            R_list.append(R)
            t_list.append(t)

        all_R_list.append(R_list)
        all_t_list.append(t_list)

        # -------------------------------------------------------------
        # Orbit representative
        # -------------------------------------------------------------

        full_rep = site.position

        all_reps_list.append([op.operate(full_rep) for op in wp.ops])

        # Pick first representative expression.
        wp_representative_str = wp.ops[0].as_xyz_str()

        # -------------------------------------------------------------
        # New positional representation
        # -------------------------------------------------------------

        A, b = wp_representative_to_A_b(wp_representative_str)

        A_list.append(A)
        b_list.append(b)


        # -------------------------------------------------------------
        # Positional DOF mask
        # -------------------------------------------------------------

        num_dof_rep, full_rep_mask = wyckoff_mask(wp_representative_str)

        wyckoffs_list.append(f"{c.group.number}_{wp.multiplicity}{wp.letter}")

        num_dof_rep_list.append(num_dof_rep)

        # -------------------------------------------------------------
        # Atom type
        # -------------------------------------------------------------

        element = str(site.specie)

        atom_types_list.append(int(Element(element).Z))

        # -------------------------------------------------------------
        # Free coordinates used by PyXtal
        # -------------------------------------------------------------

        x_pyxtal_free_pos = (wp.get_free_xyzs(full_rep))

        x_pyxtal_pos_dof_list.append(x_pyxtal_free_pos)

        full_rep_list.append(full_rep)

        full_rep_mask_list.append(full_rep_mask)

    # ---------------------------------------------------------------------
    # Pad orbit expansions
    # ---------------------------------------------------------------------

    # NOTE:
    # 48 for MP20.
    # If you later use a dataset requiring larger multiplicities,
    # increase this accordingly.
    MAX_REPS = 48

    num_orbits_local = len(all_reps_list)

    all_reps_array = np.zeros((num_orbits_local, MAX_REPS, 3), dtype=float)

    all_reps_mask = np.zeros((num_orbits_local, MAX_REPS), dtype=bool)

    all_R_array = np.zeros((num_orbits_local, MAX_REPS, 3, 3), dtype=float)

    all_t_array = np.zeros((num_orbits_local, MAX_REPS, 3), dtype=float)

    for i, (reps, R_ops, t_ops) in enumerate(zip(all_reps_list, all_R_list, all_t_list)):

        reps = np.asarray(reps, dtype=float)

        R_ops = np.asarray(R_ops, dtype=float)

        t_ops = np.asarray(t_ops, dtype=float)

        # Wrap fractional coordinates to [0, 1)
        reps = np.mod(reps, 1.0)

        t_ops = np.mod(t_ops, 1.0)

        k = reps.shape[0]

        if R_ops.shape[0] != k or t_ops.shape[0] != k:
            raise ValueError(
                f"Mismatch in orbit {i}: "
                f"len(reps)={k}, "
                f"len(R_ops)={R_ops.shape[0]}, "
                f"len(t_ops)={t_ops.shape[0]}"
            )

        if k > MAX_REPS:
            raise ValueError(
                f"Found {k} reps, "
                f"but MAX_REPS={MAX_REPS}"
            )

        all_reps_array[i, :k] = reps
        all_R_array[i, :k] = R_ops
        all_t_array[i, :k] = t_ops

        all_reps_mask[i, :k] = True

    # ---------------------------------------------------------------------
    # Convert accumulated lists to arrays
    # ---------------------------------------------------------------------

    full_rep_list = np.asarray(full_rep_list, dtype=float)

    full_rep_mask_list = np.asarray(full_rep_mask_list, dtype=float)

    atom_types_list = np.asarray(atom_types_list, dtype=int)

    num_dof_rep_list = np.asarray(num_dof_rep_list, dtype=int)

    volume = float(c.lattice.volume)

    # ---------------------------------------------------------------------
    # Flatten free positional DOFs
    # ---------------------------------------------------------------------

    if any(arr.size > 0 for arr in x_pyxtal_pos_dof_list):
        pos_dof_flat = np.concatenate([arr.ravel() for arr in x_pyxtal_pos_dof_list if arr.size > 0], axis=0)

    else:
        pos_dof_flat = np.array([], dtype=float)

    # ---------------------------------------------------------------------
    # Sanity checks
    # ---------------------------------------------------------------------

    assert (int(np.sum(full_rep_mask_list).item()) == len(pos_dof_flat))

    assert (sum(num_dof_rep_list) == len(pos_dof_flat))

    # ---------------------------------------------------------------------
    # New representation arrays
    # ---------------------------------------------------------------------

    A_array = np.asarray(A_list, dtype=float)

    b_array = np.asarray(b_list, dtype=float)


    # ---------------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------------

    sym_info = {

        # -------------------------------------------------------------
        # Space group
        # -------------------------------------------------------------

        "space_group": c.group.number,

        # -------------------------------------------------------------
        # Lattice
        # -------------------------------------------------------------

        "full_cell_rad": full_cell_rad,

        "full_cell_mask": full_cell_mask,

        "num_dof_cell": num_dof_cell,

        # -------------------------------------------------------------
        # Orbit/site information
        # -------------------------------------------------------------

        "full_rep": full_rep_list,

        "full_rep_mask": full_rep_mask_list,

        "wyckoff_sites": wyckoffs_list,

        "num_orbits": num_orbits,

        "atom_types": atom_types_list,

        # Number of atoms in entire conventional cell
        "num_atoms_conv": num_atoms_conv,

        "num_dof_rep": num_dof_rep_list,

        # -------------------------------------------------------------
        # Free DOFs
        # -------------------------------------------------------------

        "x_cell_free_dof": free_cell_deg,

        "x_pos_free_dof": pos_dof_flat,

        # -------------------------------------------------------------
        # Misc
        # -------------------------------------------------------------

        "volume": volume,
        

        # -------------------------------------------------------------
        # All equivalent orbit representatives
        # -------------------------------------------------------------

        "all_reps": all_reps_array,

        "all_reps_mask": all_reps_mask,

        # -------------------------------------------------------------
        # Symmetry operations used to expand orbit
        # -------------------------------------------------------------

        "R": all_R_array,

        "t": all_t_array,

        "A": A_array,
        
        "b": b_array,


    }

    if return_pym_full:
        return sym_info, pmg_full

    return sym_info


# =========================================================================
# Process ONE material
# =========================================================================

def _process_one_task(task, catch_errors: bool = True):
    """
    Process one material.

    Parameters
    ----------
    task
        Tuple:
            (material_id, cif_string)

    catch_errors
        True:
            Convert errors into dictionaries.
            Used for multiprocessing.

        False:
            Let exceptions propagate normally.
            Used for debugging.
    """

    material_id, cif_str = task

    def _process():

        sym_info = build_crystal_pyxtal(cif_str)

        # -------------------------------------------------------------
        # Convert Wyckoff strings -> token IDs
        # -------------------------------------------------------------

        wy_strs = list(map(str, sym_info["wyckoff_sites"]))

        wy_ids = [_G_TOKEN_TO_ID.get(wy, _G_UNK_ID) for wy in wy_strs]

        sym_info["wyckoff_sites"] = np.asarray(wy_ids, dtype=np.int64)

        return {
            "ok": True,
            "mp_id": material_id,
            "cif": cif_str,
            "graph_arrays": sym_info,
        }

    # ---------------------------------------------------------------------
    # Debug mode:
    # do NOT catch errors.
    #
    # You want pdb / VS Code to stop exactly where the error occurs.
    # ---------------------------------------------------------------------

    if not catch_errors:
        return _process()

    # ---------------------------------------------------------------------
    # Production mode:
    # catch errors so that one bad CIF does not kill preprocessing.
    # ---------------------------------------------------------------------

    try:
        return _process()

    except Exception as e:

        return {
            "ok": False,
            "mp_id": material_id,
            "err_type": type(e).__name__,
            "err_msg": str(e),
            "tb": traceback.format_exc(
                limit=10
            ),
            "cif_head": (
                cif_str[:400]
                if isinstance(cif_str, str)
                else None
            ),
        }


# =========================================================================
# Main preprocessing function
# =========================================================================

def sym_preprocess(
    input_file: str,
    debug: bool = False,
    material_ids=None,
    start_idx: int = 0,
    num_to_view: int | None = None,
    num_workers: int | None = None,
    chunksize: int = 8,
    use_spawn: bool = True,
):
    """
    Preprocess a crystal dataset.

    Parameters
    ----------
    input_file
        CSV file containing:
            material_id
            cif

    debug
        False:
            multiprocessing enabled.

        True:
            sequential execution,
            no multiprocessing,
            exceptions propagate normally.

    material_ids
        Optional collection of material IDs.

        Examples:
            ["mp-22862"]

            [
                "mp-22862",
                "mp-34284",
            ]

        None:
            process all materials.

    start_idx
        Start processing from this position AFTER filtering
        by material_ids.

    num_to_view
        Number of structures to process.

        None:
            process everything after start_idx.

    num_workers
        Number of multiprocessing workers.

        None:
            use os.cpu_count().

        Ignored in debug mode.

    chunksize
        Multiprocessing imap chunksize.

    use_spawn
        True:
            multiprocessing uses spawn.

        False:
            multiprocessing uses fork.
    """

    # =====================================================================
    # Wyckoff vocabulary
    # =====================================================================

    wyckoff_data = load_or_create_wyckoff_data()

    flat_pairs = wyckoff_data["flat_pairs"]

    token_to_id = {token: idx for idx, token in enumerate(flat_pairs)}

    unk_id = len(flat_pairs)

    # =====================================================================
    # Read dataset
    # =====================================================================

    df = pd.read_csv(input_file, usecols=["material_id", "cif"])

    print(
        f"Loaded {len(df)} structures "
        f"from {input_file}"
    )

    # =====================================================================
    # Filter by material ID
    # =====================================================================

    if material_ids is not None:

        # Allow user to supply either:
        #
        #   material_ids="mp-22862"
        #
        # or
        #
        #   material_ids=["mp-22862", "mp-34284"]
        #
        if isinstance(material_ids, str):
            material_ids = [material_ids]

        material_ids = list(material_ids)

        requested_ids = set(material_ids)

        # Preserve CSV ordering.
        df = df[df["material_id"].isin(requested_ids)]

        found_ids = set(df["material_id"])

        missing_ids = requested_ids - found_ids

        print(
            f"Selected {len(df)} structures "
            f"from {len(requested_ids)} requested IDs"
        )

        if missing_ids:

            print("WARNING: material IDs not found:")

            for mp_id in sorted(missing_ids):
                print(f"  {mp_id}")

    # =====================================================================
    # Slice dataset
    # =====================================================================

    if num_to_view is not None:

        end_idx = start_idx + num_to_view
        

        df = df.iloc[start_idx:end_idx]

    else:

        df = df.iloc[start_idx:]

    # =====================================================================
    # Convert dataframe -> lightweight worker tasks
    #
    # Each task:
    #
    #       (material_id, cif)
    # =====================================================================

    tasks = list(df[["material_id", "cif"]].itertuples(index=False, name=None))

    if len(tasks) == 0:

        print("No structures selected.")

        return []

    print(f"Processing {len(tasks)} structures")

    # =====================================================================
    # DEBUG MODE
    # =====================================================================
    #
    # Important:
    #
    # This does NOT use a pool with one worker.
    #
    # It is ordinary sequential Python execution:
    #
    #     material 1
    #        ↓
    #     material 2
    #        ↓
    #     material 3
    #
    # This is what you want for:
    #
    #     breakpoint()
    #     pdb
    #     VS Code debugger
    #     print debugging
    #
    # =====================================================================

    if debug:

        print()
        print(
            "=" * 70
        )

        print(
            "DEBUG MODE"
        )

        print(
            "Multiprocessing disabled."
        )

        print(
            "Exceptions will propagate normally."
        )

        print(
            "=" * 70
        )

        # Normally this happens inside each multiprocessing worker.
        #
        # Since debugging runs in the main process, initialize the
        # same globals here.
        _init_worker(token_to_id, unk_id)

        results = []

        t0 = time.perf_counter()

        for task in tqdm(tasks, desc="Preprocessing"):

            material_id, _ = task

            print()
            print(
                f"Working with material: {material_id}"
            )

            # catch_errors=False is intentional:
            #
            # if something fails we want the debugger to stop exactly
            # where the actual failure occurred.
            result = _process_one_task(task, catch_errors=False)

            results.append(result)

        dt = time.perf_counter() - t0

    # =====================================================================
    # MULTIPROCESSING MODE
    # =====================================================================

    else:

        if num_workers is None:

            num_workers = os.cpu_count() or 1

        # -----------------------------------------------------------------
        # Avoid each multiprocessing worker creating its own BLAS threads.
        # -----------------------------------------------------------------

        os.environ.setdefault("OMP_NUM_THREADS", "1")

        os.environ.setdefault("MKL_NUM_THREADS", "1")

        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        context = "spawn" if use_spawn else "fork"

        print()
        print(
            "=" * 70
        )

        print(
            "MULTIPROCESSING MODE"
        )

        print(
            f"Workers: {num_workers}"
        )

        print(
            f"Context: {context}"
        )

        print(
            f"Chunksize: {chunksize}"
        )

        print(
            "=" * 70
        )

        ctx = mp.get_context(
            context
        )

        t0 = time.perf_counter()

        # Need a single-argument callable for Pool.imap.
        #
        # _process_one_task's default:
        #
        #       catch_errors=True
        #
        # is exactly what we want here.
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(token_to_id, unk_id)) as pool:

            results = list(
                tqdm(
                    pool.imap(
                        _process_one_task,
                        tasks,
                        chunksize=chunksize,
                    ),
                    total=len(tasks),
                    desc="Preprocessing",
                )
            )

        dt = time.perf_counter() - t0

    # =====================================================================
    # Successes / failures
    # =====================================================================

    results_ok = [result for result in results if result.get("ok", False)]

    results_bad = [result for result in results if not result.get("ok", False)]

    # =====================================================================
    # Statistics
    # =====================================================================

    print()
    print(
        "=" * 70
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Processed: {len(results)}"
    )

    print(
        f"OK:        {len(results_ok)}"
    )

    print(
        f"BAD:       {len(results_bad)}"
    )

    print(
        f"Time:      {dt:.3f} s"
    )

    if len(results) > 0:

        print(
            f"Throughput: "
            f"{len(results) / dt:.2f} structures/s"
        )

        print(
            f"Time/structure: "
            f"{1e3 * dt / len(results):.2f} ms"
        )

    # =====================================================================
    # Save failures
    # =====================================================================

    if results_bad:

        bad_df = pd.DataFrame(results_bad)

        bad_path = os.path.join(os.path.dirname(input_file), "bad_cifs.csv")

        bad_df.to_csv(bad_path, index=False)

        print()
        print(
            f"Saved failures to:"
        )

        print(
            f"  {bad_path}"
        )

        print()
        print(
            "First failures:"
        )

        for result in results_bad[:5]:

            print()
            print(
                "--- FAIL ---"
            )

            print(
                "mp_id :",
                result["mp_id"],
            )

            print(
                "type  :",
                result["err_type"],
            )

            print(
                "msg   :",
                result["err_msg"],
            )

    # =====================================================================
    # Return successful structures only.
    #
    # imap preserves input order, so the structures remain in dataset order.
    # =====================================================================

    return results_ok