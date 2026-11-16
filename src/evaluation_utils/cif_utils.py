from pathlib import Path
from typing import Optional
from collections import Counter
import re
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core import Structure

def filter_cif_paths_by_sg(cif_paths, new_stats, target_sg=None, exclude_sg=None):
    """
    Filter cif_paths based on spacegroup from new_stats.

    target_sg: keep only this SG
    exclude_sg: remove this SG (e.g. 1 for non-P1)
    """
    sg_list = new_stats["spacegroup"]

    filtered_paths = []
    indices = []

    for i, (p, sg) in enumerate(zip(cif_paths, sg_list)):
        if target_sg is not None and sg == target_sg:
            filtered_paths.append(p)
            indices.append(i)
        elif exclude_sg is not None and sg != exclude_sg:
            filtered_paths.append(p)
            indices.append(i)

    return filtered_paths, indices

def subset_stats(stats_dict, indices):
    return {
        k: [v[i] for i in indices]
        for k, v in stats_dict.items()
    }


def canonicalize_structure(structure: Structure, symprec: float = 1e-2) -> Structure:
    """
    Convert structure to a canonical primitive representation.
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)

    # 1. primitive standard
    structure = sga.get_primitive_standard_structure()

    # NOTE Is this needed? YES IT IS
    # 2. Niggli reduction (canonical lattice)
    structure = structure.get_reduced_structure()

    # 3. deterministic ordering
    structure = structure.get_sorted_structure()

    return structure


def resolve_cif_dir(cif_dir: Path) -> Path:
    """
    Accept either:
      - a directory containing .cif files
      - a run root that contains a 'cif/' subdir
    """
    cif_dir = cif_dir.resolve()
    if (cif_dir / "cif").is_dir():
        return cif_dir / "cif"
    return cif_dir


def get_spacegroup_from_name(path: Path) -> Optional[str]:
    """
    Extract spacegroup token from filename, e.g.:
      crystal_2_sg97.cif -> sg97
      crystal_4_absurd_duplicate_pyxtal_reps_sg14.cif -> sg14
      crystal_1_absurd_could_not_build_pyxtal_sg0.txt -> sg0
    """
    stem = path.stem
    m = re.search(r"(?:^|_)sg(\d+)(?:_|$)", stem)
    if m:
        return f"sg{m.group(1)}"
    return None


def get_absurd_reason(path: Path) -> Optional[str]:
    """
    Extract absurd reason from filenames like:
      crystal_1_absurd_could_not_build_pyxtal.txt
      crystal_1_absurd_could_not_build_pyxtal_sg0.txt
      crystal_2_absurd_atoms_too_close.cif
      crystal_4_absurd_duplicate_pyxtal_reps_sg14.cif

    Returns:
      reason string if absurd, else None

    Important:
      strips trailing _sg<number> from the absurd reason
    """
    stem = path.stem

    if "_absurd_" in stem:
        reason = stem.split("_absurd_", 1)[1]
        reason = re.sub(r"_sg\d+$", "", reason)
        return reason

    if stem.endswith("_absurd"):
        return "absurd"

    return None

def list_cif_paths(
    cif_dir: Path,
    filter_absurd: bool = True
) -> tuple[list[Path], int, int, Counter, Counter]:
    """
    Returns:
      kept_paths, n_absurd, n_total, absurd_reason_counts, absurd_sg_counts

    Absurd heuristic:
      any filename whose stem contains '_absurd'
    """
    cif_dir = resolve_cif_dir(cif_dir)
    if not cif_dir.exists():
        raise FileNotFoundError(f"cif_dir does not exist: {cif_dir}")

    all_paths = sorted(list(cif_dir.glob("*.cif")) + list(cif_dir.glob("*.txt")))
    n_total = len(all_paths)

    absurd_reasons = [get_absurd_reason(p) for p in all_paths]
    absurd_mask = [reason is not None for reason in absurd_reasons]
    n_absurd = int(sum(absurd_mask))

    absurd_reason_counts = Counter(
        reason for reason in absurd_reasons if reason is not None
    )

    absurd_sg_counts = Counter(
        get_spacegroup_from_name(p)
        for p, is_absurd in zip(all_paths, absurd_mask)
        if is_absurd and get_spacegroup_from_name(p) is not None
    )

    if filter_absurd:
        kept = [p for p, is_absurd in zip(all_paths, absurd_mask) if not is_absurd]
    else:
        kept = all_paths

    return kept, n_absurd, n_total, absurd_reason_counts, absurd_sg_counts