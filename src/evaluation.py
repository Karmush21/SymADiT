#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from pathlib import Path
import os
import time
import argparse
from collections import Counter
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from src.data.dataset import SymDataset
from torch.utils.data import Subset
import torch
from src.evaluation_utils.build_training_materials import build_train_stats, build_training_data_from_csv
from src.evaluation_utils.plots import (
    plot_wyckoff_counts_for_sg, 
    plot_num_atoms, 
    plot_volume, 
    plot_atom_type_counts, 
    plot_num_atoms_per_sg_boxplot, 
    plot_atom_type_unique_counts
)

from src.evaluation_utils.read_in_new_materials import (
    build_new_stats_parallel, 
    load_structures_from_cifs_parallel, 
    load_structures_from_cifs,
    build_new_stats
)

from src.evaluation_utils.stats import (
    js_global_divergence_discrete_feature,
    wyckoff_jsd_over_top_spacegroups,
    p1_rate

)

from src.evaluation_utils.cif_utils import list_cif_paths
from src.evaluation_utils.novel_and_unique import (
    evaluate_novelty_and_save_parallel, 
    compute_uniqueness, 
    evaluate_novelty_and_save
)
from src.evaluation_utils.stats_wasserstein import (
    wasserstein_atomic_density,
    wasserstein_num_unique_elements,
    wasserstein_num_atoms,
    wasserstein_num_orbits
)

from src.symmetry.wyckoff_data import load_or_create_wyckoff_data 




warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="spglib")
warnings.filterwarnings(
    "ignore",
    message=r"Weights only load failed\.",
    category=UserWarning,
    module=r"torch_geometric\.io\.fs",
)



# ===================== DEFAULTS (OVERRIDABLE VIA CLI) =====================
DEFAULT_GLOBAL_PATH = ""
DEFAULT_RUN_DIRS = ["cif_temp/ADiT_10_crystals"] # Replace with whatever 

DEFAULT_OUT_DIR = f"{DEFAULT_GLOBAL_PATH}/eval_offline_dir"
DEFAULT_FILTER_ABSURD = False

DEFAULT_STOL = 0.3 
DEFAULT_ANGLE_TOL = 10.0 
DEFAULT_LTOL = 0.2 

DEFAULT_PARALLEL = False
DEFAULT_NUM_WORKERS = 18
DEFAULT_CHUNKSIZE = 10

DEFAULT_USE_SPAWN = True
DEFAULT_NOVEL_SAMPLE_K = 1000
DEFAULT_SEED = 0

# ========================================================================


def parse_args():
    p = argparse.ArgumentParser("eval_offline (CPU)")

    # Paths / run selection
    p.add_argument("--global_path", type=str, default=DEFAULT_GLOBAL_PATH)
    p.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        default=DEFAULT_RUN_DIRS,
        help="One or more run dirs (relative to global_path).",
    )

    # Filtering
    p.add_argument("--filter_absurd", action="store_true", default=DEFAULT_FILTER_ABSURD)
    p.add_argument("--no_filter_absurd", action="store_true", default=False)


    # Dataset size controls)
    p.add_argument("--dataset", type=str, default="mp20")


    # Matching tolerances
    p.add_argument("--stol", type=float, default=DEFAULT_STOL)
    p.add_argument("--angle_tol", type=float, default=DEFAULT_ANGLE_TOL)
    p.add_argument("--ltol", type=float, default=DEFAULT_LTOL)
    
    # NOTE
    # Whether you want to do real novelty
    # So you add skip_structure_reduction to make it False in novelty code. (Follow pymatgen convention)
    p.add_argument("--skip_structure_reduction", action="store_false", default=True, help="Skip structure reduction step when calculating novelty")

    # Parallelism
    p.add_argument("--parallel", action="store_true", default=DEFAULT_PARALLEL,
                   help="Enable multiprocessing where supported.")
    p.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                   help="Number of worker processes for multiprocessing.")
    p.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE,
                   help="Chunksize passed to Pool.imap.")
    p.add_argument("--use_spawn", action="store_true", default=DEFAULT_USE_SPAWN,
                   help="Use multiprocessing spawn context.")
    p.add_argument("--no_use_spawn", action="store_true", default=False,
                   help="Use fork instead of spawn (Linux only, if safe).")

    # Novelty eval settings
    p.add_argument("--novel_sample_k", type=int, default=DEFAULT_NOVEL_SAMPLE_K,
                   help="If set, evaluate novelty on at most this many unique reps. Use 0 or negative for all.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed for novelty subsampling.")
    
    p.add_argument("--save_novelty", action="store_true", default=False)


    p.add_argument(
    "--cell_representation",
    type=str,
    default="conventional",
    choices=["primitive", "conventional", "canonicalize"],
    help=(
        "Type of cell to output after relaxation. "
        "'primitive' = smallest cell, "
        "'conventional' = symmetry-standard cell, "
        "'canonicalize' = Own function to try"
    ),
    )


    return p.parse_args()

def extract_run_name(run_dir: Path):
    if run_dir.name.lower() in {"cif", "cifs"}:
        return run_dir.parent.name
    return run_dir.name


# ------------------ Evaluate one run ------------------

def evaluate_one_run(run_dir: Path, training_materials, train_stats, wyckoff_dict, args):
    if not run_dir.exists():
        print(f"[WARN] Run dir does not exist, skipping: {run_dir}")
        return


    run_name = extract_run_name(run_dir)
    run_out_dir = Path(args.out_dir) / run_name

    run_out_dir.mkdir(parents=True, exist_ok=True)
    

    summary_lines = []
    def log(msg: str):
        print(msg)
        summary_lines.append(msg)

    log(f"\n=== RUN: {run_dir} ===")
    log(f"Output dir: {run_out_dir}")
    log(f"Number of training Materials: {len(training_materials)}\n")

    
    

    # all_cif_paths is for valdiity, non_absurd_cif_paths is for stats
    all_cif_paths, n_absurd, n_total, absurd_reason_counts, absurd_sg_counts = list_cif_paths(run_dir, filter_absurd=False)
    non_absurd_cif_paths, _, _, _, _ = (list_cif_paths(run_dir, filter_absurd=True))

    log(f"Found CIFs total: {n_total}")
    log(f"Absurd CIFs: {n_absurd}")
    log(f"Non-absurd CIFs: {n_total - n_absurd}")

    log("Absurd breakdown by reason:")
    for reason, count in absurd_reason_counts.most_common():
        log(f"  - {reason}: {count}")

    log("Absurd breakdown by spacegroup:")
    for sg, count in sorted(absurd_sg_counts.items(), key=lambda x: int(x[0][2:])):
        log(f"  - {sg}: {count}")



    if args.parallel:
        t0 = time.perf_counter()
        new_stats, failed = build_new_stats_parallel(
            non_absurd_cif_paths,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            use_spawn=args.use_spawn,
        )
        t1 = time.perf_counter()
    
    else:
        t0 = time.perf_counter()
        new_stats = build_new_stats(non_absurd_cif_paths)
        t1 = time.perf_counter()


    log(f"Build_new_stats time: {t1 - t0:.3f} seconds | Using Parallel: {args.parallel}")


    stats = {"train" : train_stats, "new": new_stats}

    df_spacegroups = pd.DataFrame({"material_id": range(len(new_stats["spacegroup"])), "spacegroup": new_stats["spacegroup"]})
    df_spacegroups.to_csv(f"{run_out_dir}/spacegroups.csv", index=False)



    #-------------------Checking Distributions------------------------------

    log("Computing stats...")

    # NOTE Spacegroup metrics don't apply for symmetry-aware methods
    # number of atoms don't apply for symmetry-aware methods and DiffCSP++
    p1 = p1_rate(stats)
    log(f"P1 rate: {p1:.6f}")
    res_sg = js_global_divergence_discrete_feature(stats, key="spacegroup")
    res_orbits = js_global_divergence_discrete_feature(stats, key="num_orbits")
    res_atoms = js_global_divergence_discrete_feature(stats, key="num_atoms")
    res_unique = js_global_divergence_discrete_feature(stats, key="atom_types_unique")

    
    log(f"JSD Distance | Space group: {res_sg['js_dist']:.6f}")
    log(f"JSD Distance | Num orbits: {res_orbits['js_dist']:.6f}")
    log(f"JSD Distance | Num atoms: {res_atoms['js_dist']:.6f}")
    log(f"JSD Distance | Unique atom types: {res_unique['js_dist']:.6f}")
    
    res_wy = wyckoff_jsd_over_top_spacegroups(stats, num_sg=10)
    log(f"Weighted Wyckoff JSD distance: {res_wy['weighted_js_dist']:.6f}")
    log(f"Simple mean Wyckoff JSD distance: {res_wy['simple_mean_js_dist']:.6f}")
    for sg in res_wy["top_spacegroups"]:
        r = res_wy["per_sg"][sg]
        log(
            f"SG {sg:3d} | "
            f"JSD distance: {r['js_dist']:.6f} | "
            f"train structs: {r['n_train_structures']} | "
            f"new structs: {r['n_new_structures']}"
        )

    res_density = wasserstein_atomic_density(stats)
    res_unique_el = wasserstein_num_unique_elements(stats, key="atom_types_unique")
    res_num_atoms = wasserstein_num_atoms(stats)
    res_num_orbits = wasserstein_num_orbits(stats, key="num_orbits")

    log(f"Wasserstein Distance | Atomic density: {res_density['wasserstein']:.6f}")
    log(f"Wasserstein Distance | Num unique elements: {res_unique_el['wasserstein']:.6f}")
    log(f"Wasserstein Distance | Num atoms (conv): {res_num_atoms['conv']['wasserstein']:.6f}")
    log(f"Wasserstein Distance | Num atoms (prim): {res_num_atoms['prim']['wasserstein']:.6f}")
    log(f"Wasserstein Distance | Num orbits: {res_num_orbits['wasserstein']:.6f}")
    #-------------------Checking Distributions------------------------------

    #-------------------Plotting------------------------------
    print("Plotting...")
    # Plotting functions per spacegroup
    plot_wyckoff_counts_for_sg(stats, 225, wyckoff_dict, # Most popular (6686) # Cubic system
                                save_dir=str(run_out_dir))
    
    plot_wyckoff_counts_for_sg(stats, 164, wyckoff_dict, # Has 866 # Trigonal system
                            save_dir=str(run_out_dir))
    

    plot_wyckoff_counts_for_sg(stats, 36, wyckoff_dict, # Has 224 in total # Orthorhombic system
                            save_dir=str(run_out_dir))
    
    # Plotting functions per GLOBAL info
    plot_num_atoms(stats, save_dir=str(run_out_dir))
    

    plot_volume(stats, save_dir=str(run_out_dir))                                         

    plot_atom_type_counts(stats, save_dir=str(run_out_dir))
    plot_atom_type_unique_counts(stats, save_dir=str(run_out_dir))

    plot_num_atoms_per_sg_boxplot(stats, save_dir=str(run_out_dir))
    #-------------------Plotting------------------------------

    
    #-------------------Load New Structures------------------------------
    if args.parallel:
        t0 = time.perf_counter()
        new_structures, validity_results, absurd_paths, failed_parse = load_structures_from_cifs_parallel(
            all_cif_paths,
            args.cell_representation,
            strict=False,
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            use_spawn=args.use_spawn,
        )
        t1 = time.perf_counter()

    else:
        t0 = time.perf_counter()
        new_structures, validity_results, absurd_paths, failed_parse = load_structures_from_cifs(all_cif_paths, args.cell_representation)
        t1 = time.perf_counter()

    #-------------------Load New Structures------------------------------
    
    log(f"Load new structures time: {t1 - t0:.3f} seconds | Using Parallel: {args.parallel}")

    assert len(new_structures) == len(validity_results), (
    f"Mismatch: {len(new_structures)} structures versus "
    f"{len(validity_results)} validity results"
    )

    n_total = len(all_cif_paths)
    n_evaluated = len(validity_results)
    n_failed = len(failed_parse)

    log(f"Total generated files: {n_total}")
    log(f"Successfully evaluated: {n_evaluated}")
    log(f"Absurd filenames evaluated: {len(absurd_paths)}")
    log(f"Failed processing: {n_failed}")


    val_stats = Counter()

    for result in validity_results:
        val_stats["valid"] += int(result["valid"])
        val_stats["comp_valid"] += int(result["comp_valid"])
        val_stats["struct_valid"] += int(result["struct_valid"])

    if n_total > 0:
        log(
            f"Valid:        {val_stats['valid']}/{n_total} "
            f"({val_stats['valid'] / n_total:.4f})"
        )
        log(
            f"Comp valid:   {val_stats['comp_valid']}/{n_total} "
            f"({val_stats['comp_valid'] / n_total:.4f})"
        )
        log(
            f"Struct valid: {val_stats['struct_valid']}/{n_total} "
            f"({val_stats['struct_valid'] / n_total:.4f})"
        )
    else:
        log("No generated files found.")

    

    # -------- Filter to only structurally valid structures --------
    struct_valid_structures = [
        s for s, r in zip(new_structures, validity_results) if r["struct_valid"] # NOTE We don't care about comp valid
    ]

    log(f"Keeping {len(struct_valid_structures)} structurally valid structures for uniqueness")

    
    
    matcher = StructureMatcher(stol=args.stol, angle_tol=args.angle_tol, ltol=args.ltol)

    unique_groups, unique_reps, unique_rate = compute_uniqueness(struct_valid_structures, matcher)
    log(f"Unique rate: {unique_rate:.6f} ({len(unique_groups)} unique / {len(struct_valid_structures)} total)")


    novel_save_dir = run_out_dir / "novel_unique_cifs"
    save_dir = novel_save_dir if args.save_novelty else None

    if args.parallel:
        t0 = time.perf_counter()
        is_novel_flags, novel_rate, novel_structures, evaluated_reps = evaluate_novelty_and_save_parallel(
            unique_reps=unique_reps,
            training_structs=training_materials,
            matcher=matcher,
            save_dir=save_dir,
            sample_k=args.novel_sample_k,
            seed=args.seed,
            skip_structure_reduction = args.skip_structure_reduction,
            desc="Novelty (Parallel)",
            num_workers=args.num_workers,
            chunksize=args.chunksize,
            use_spawn=args.use_spawn,
        )
        t1 = time.perf_counter()

    else:
        t0 = time.perf_counter()
        is_novel_flags, novel_rate, novel_structures, evaluated_reps = evaluate_novelty_and_save(
            unique_reps=unique_reps,
            training_structs=training_materials,
            matcher=matcher,
            save_dir=save_dir,
            sample_k=args.novel_sample_k,   # or None to evaluate all unique reps
            seed=args.seed,
            skip_structure_reduction = args.skip_structure_reduction,
            desc="Novelty",
        )
        t1 = time.perf_counter()

    log(f"Novel time: {t1 - t0:.3f} seconds | Using Parallel: {args.parallel} | Skip Structure Reduction: {args.skip_structure_reduction}")

    log(f"Novel rate: {novel_rate:.4f} ({sum(is_novel_flags)}/{len(evaluated_reps)})")
    if save_dir is not None:
        log(f"Saved {len(novel_structures)} novel unique CIFs to: {save_dir}")


    # NOTE here you could do new things with the materials that are unique AND NOVEL
    # i.e. novel_structures. 


    summary_path = run_out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Saved summary to: {summary_path}")


# ------------------ Main ------------------


def main():
    args = parse_args()

    wyckoff_data = load_or_create_wyckoff_data()


    wyckoff_dict = wyckoff_data["wyckoff_dict"]


    # Normalize path-ish inputs
    args.global_path = str(Path(args.global_path).resolve())
    args.out_dir = str(Path(args.out_dir).resolve())

    # Print config once
    print("=== eval_offline config ===")
    print(f"global_path      = {args.global_path}")
    print(f"out_dir          = {args.out_dir}")
    print(f"save_novelty     = {args.save_novelty}")
    print(f"run_dirs         = {args.run_dirs}")
    print(f"dataset          = {args.dataset}")
    print(f"filter_absurd    = {args.filter_absurd}")
    print(f"stol/angle/ltol  = {args.stol}/{args.angle_tol}/{args.ltol}")
    print(f"Parallel         = {args.parallel}")
    print(f"Skip Structure reduction = {args.skip_structure_reduction}")
    print(f"num_workers      = {args.num_workers}")
    print(f"chunksize        = {args.chunksize}")
    print(f"use_spawn        = {args.use_spawn}")
    print(f"novel_sample_k   = {args.novel_sample_k}")
    print(f"seed             = {args.seed}")
    print(f"cell representation             = {args.cell_representation}")
    print("===========================")
    
    dataset_root = os.path.join(args.global_path, "data", args.dataset)
    dataset = SymDataset(root=dataset_root, debug=False)

    dataset_df = pd.read_csv(os.path.join(dataset_root, "raw", "raw.csv"))
    
    
    if args.dataset == "mp20":
        train_idx = torch.arange(0, 27138)

    elif args.dataset == "mpts52":
        train_idx = torch.arange(0, 27380)

    elif args.dataset == "mp20_mpts52_merged":
        split_path = os.path.join(dataset_root, "train_val_split.pt")
        split_dict = torch.load(split_path, weights_only=False)
        train_idx = split_dict["train_idx"]

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    training_data_cif = dataset_df.iloc[train_idx.tolist()]["cif"].tolist()

    # (optional) if you also need dataset subset
    training_data = Subset(dataset, train_idx.tolist())

    
    train_stats = build_train_stats(training_data)  
    
    
    training_structures = build_training_data_from_csv(training_data_cif, args.cell_representation)
    
    
    # Evaluate each run dir
    for rel in args.run_dirs:
        run_dir = Path(args.global_path) / rel
        evaluate_one_run(run_dir, training_structures, train_stats, wyckoff_dict, args)


if __name__ == "__main__":
    main()
