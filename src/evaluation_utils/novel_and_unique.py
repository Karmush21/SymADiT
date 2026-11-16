from pathlib import Path
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.structure_matcher import StructureMatcher
import random

def compute_uniqueness(structs, matcher):

    print(f"Computing uniqueness for {len(structs)} structures...")

    groups = matcher.group_structures(structs)

    reps = [g[0] for g in groups]
    rate = len(groups) / len(structs)

    return groups, reps, rate




def is_novel(struct: Structure, training_structs: list[Structure], matcher: StructureMatcher, skip_structure_reduction = True) -> bool:
    for train_struct in training_structs:
        # fit() returns True if they match
        if matcher.fit(struct, train_struct, skip_structure_reduction=skip_structure_reduction): 
                                                                            # NOTE skip_structure_reduction Should really be set to False for novelty, not true
                                                                             # ADiT had set it to True. 
            return False
    return True


def evaluate_novelty_and_save(
    unique_reps: list[Structure],
    training_structs: list[Structure],
    matcher: StructureMatcher,
    save_dir: str | Path | None = None,
    sample_k: int | None = None,
    seed: int = 0,
    skip_structure_reduction: bool = True,
    desc: str = "Novelty",
):
    """
    Evaluate novelty on unique representatives and save the novel ones as CIFs.

    Args:
        unique_reps: unique generated structures (one rep per uniqueness cluster)
        training_structs: training structures to compare against
        matcher: pymatgen StructureMatcher
        save_dir: directory where novel CIFs will be written
        sample_k: if not None, evaluate only a random sample of this many unique reps
        seed: RNG seed for sampling
        desc: tqdm description

    Returns:
        is_novel_flags: list[bool]
        novel_rate: float
        novel_structures: list[Structure]
        evaluated_reps: list[Structure]   # the actual list that was evaluated
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Optional subsampling
    if sample_k is not None:
        k = min(sample_k, len(unique_reps))
        random.seed(seed)
        evaluated_reps = random.sample(unique_reps, k)
    else:
        evaluated_reps = unique_reps

    is_novel_flags = []
    novel_structures = []

    for i, s in enumerate(tqdm(evaluated_reps, desc=desc, total=len(evaluated_reps))):
        novel = is_novel(s, training_structs, matcher, skip_structure_reduction)
        is_novel_flags.append(novel)

        if novel:
            novel_structures.append(s)
            if save_dir is not None:
                formula = s.composition.reduced_formula.replace("/", "_")
                out_path = save_dir / f"novel_unique_{i:05d}_{formula}.cif"
                CifWriter(s).write_file(str(out_path))

    novel_rate = (sum(is_novel_flags) / len(is_novel_flags)) if is_novel_flags else 0.0
    return is_novel_flags, novel_rate, novel_structures, evaluated_reps


#-----------------------------Parallel----------------------------------
import multiprocessing as mp
from functools import partial




def _novelty_worker(struct, training_structs, matcher_kwargs, skip_structure_reduction):
    matcher = StructureMatcher(**matcher_kwargs)
    novel = is_novel(struct, training_structs, matcher, skip_structure_reduction)
    return novel, struct


def evaluate_novelty_and_save_parallel(
    unique_reps: list[Structure],
    training_structs: list[Structure],
    matcher: StructureMatcher,
    save_dir: str | Path,
    sample_k: int | None = None,
    seed: int = 0,
    skip_structure_reduction: bool = True,
    desc: str = "Novelty (Parallel)",
    num_workers: int = 16,
    chunksize: int = 64,
    use_spawn: bool = True,
):
    """
    Parallel novelty evaluation over unique representatives.
    CIF writing is done sequentially in the parent process.
    """

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Optional subsampling
    if sample_k is not None:
        k = min(sample_k, len(unique_reps))
        random.seed(seed)
        evaluated_reps = random.sample(unique_reps, k)
    else:
        evaluated_reps = unique_reps

    matcher_kwargs = dict(
        ltol=matcher.ltol,
        stol=matcher.stol,
        angle_tol=matcher.angle_tol,
        primitive_cell=matcher._primitive_cell,
        scale=matcher._scale,
        attempt_supercell=matcher._supercell,
        allow_subset=matcher._subset,
    )

    ctx = mp.get_context("spawn" if use_spawn else "fork")

    with ctx.Pool(processes=num_workers) as pool:
        worker = partial(
            _novelty_worker,
            training_structs=training_structs,
            matcher_kwargs=matcher_kwargs,
            skip_structure_reduction=skip_structure_reduction,
        )

        results = list(
            tqdm(
                pool.imap(worker, evaluated_reps, chunksize=chunksize),
                desc=desc,
                total=len(evaluated_reps),
            )
        )

    is_novel_flags = []
    novel_structures = []

    # Save in parent process for stable naming/order
    for i, (novel, s) in enumerate(results):
        is_novel_flags.append(novel)

        if novel:
            novel_structures.append(s)
            if save_dir is not None:
                formula = s.composition.reduced_formula.replace("/", "_")
                out_path = save_dir / f"novel_unique_{i:05d}_{formula}.cif"
                CifWriter(s).write_file(str(out_path))

    novel_rate = (sum(is_novel_flags) / len(is_novel_flags)) if is_novel_flags else 0.0
    return is_novel_flags, novel_rate, novel_structures, evaluated_reps