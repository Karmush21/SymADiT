from __future__ import annotations

from typing import Any, Callable
import math

import numpy as np
from scipy.stats import wasserstein_distance


def _to_list(x: Any) -> list:
    """Convert supported containers to a Python list."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    # torch support without importing torch explicitly
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "tolist"):
        return x.detach().cpu().tolist()
    if hasattr(x, "tolist"):
        return x.tolist()
    return [x]


def _extract_unique_atom_count(entry: Any) -> int:
    """
    Return the number of unique atom types for one crystal entry.

    Supported cases:
    - entry is already an int, e.g. 3
    - entry is a list/array/tensor of atomic numbers, e.g. [14, 14, 8, 8]
    - entry is a list of already-unique atomic numbers, e.g. [14, 8]
    """
    if isinstance(entry, (int, np.integer)):
        return int(entry)

    vals = _to_list(entry)
    if len(vals) == 0:
        return 0

    # Remove padding-like zeros or None if needed.
    # Uncomment the next line if 0 is padding in your representation.
    # vals = [v for v in vals if v not in (0, None)]

    return len(set(vals))


def _safe_density(num_atoms: Any, volume: Any, eps: float = 1e-12) -> float:
    """Compute atomic density = num_atoms / volume with basic safety."""
    n = float(num_atoms)
    v = float(volume)

    if not math.isfinite(n) or not math.isfinite(v):
        raise ValueError(f"Non-finite num_atoms={num_atoms}, volume={volume}")
    if v <= eps:
        raise ValueError(f"Volume must be > 0, got {volume}")

    return n / v


def wasserstein_feature(
    stats: dict,
    feature_name: str,
    extract_fn: Callable[[dict], list[float]],
) -> dict:
    """
    Generic 1D Wasserstein distance between train and new feature distributions.

    Parameters
    ----------
    stats : dict
        Expected format:
        {
            "train": train_stats,
            "new": new_stats
        }
    feature_name : str
        Human-readable feature name for reporting.
    extract_fn : callable
        Function taking train_stats or new_stats and returning a list of numbers.

    Returns
    -------
    dict
        {
            "feature": feature_name,
            "train_values": [...],
            "new_values": [...],
            "wasserstein": float,
            "n_train": int,
            "n_new": int,
        }
    """
    train_values = extract_fn(stats["train"])
    new_values = extract_fn(stats["new"])

    if len(train_values) == 0:
        raise ValueError(f"No train values extracted for feature '{feature_name}'")
    if len(new_values) == 0:
        raise ValueError(f"No new values extracted for feature '{feature_name}'")

    wdist = wasserstein_distance(train_values, new_values)

    return {
        "feature": feature_name,
        "train_values": train_values,
        "new_values": new_values,
        "wasserstein": float(wdist),
        "n_train": len(train_values),
        "n_new": len(new_values),
    }


def wasserstein_atomic_density(stats: dict) -> dict:
    """
    Compute Wasserstein distance for atomic density rho = num_atoms / volume.
    """
    def extract_fn(split_stats: dict) -> list[float]:
        num_atoms = _to_list(split_stats["num_atoms"])
        volumes = _to_list(split_stats["volume"])

        if len(num_atoms) != len(volumes):
            raise ValueError(
                f"Length mismatch: num_atoms={len(num_atoms)} vs volume={len(volumes)}"
            )

        densities = []
        for n, v in zip(num_atoms, volumes):
            try:
                densities.append(_safe_density(n, v))
            except ValueError:
                # Skip invalid entries if needed
                continue

        return densities

    return wasserstein_feature(
        stats=stats,
        feature_name="atomic_density",
        extract_fn=extract_fn,
    )


def wasserstein_num_unique_elements(stats: dict, key: str = "atom_types_unique") -> dict:
    """
    Compute Wasserstein distance for number of unique elements per crystal.

    Parameters
    ----------
    stats : dict
        {
            "train": train_stats,
            "new": new_stats
        }
    key : str
        Usually "atom_types_unique". Can also be "atom_types" if needed.
    """
    def extract_fn(split_stats: dict) -> list[float]:
        entries = _to_list(split_stats[key])
        return [float(_extract_unique_atom_count(entry)) for entry in entries]

    return wasserstein_feature(
        stats=stats,
        feature_name="num_unique_elements",
        extract_fn=extract_fn,
    )


def wasserstein_num_atoms(stats: dict) -> dict:
    """
    Compute Wasserstein distances for:
      - conventional atoms
      - primitive atoms
    """

    def extract_fn(key):
        def _fn(split_stats: dict) -> list[float]:
            values = _to_list(split_stats.get(key, []))
            return [float(v) for v in values]
        return _fn

    res_conv = wasserstein_feature(
        stats=stats,
        feature_name="num_atoms (conventional)",
        extract_fn=extract_fn("num_atoms"),
    )

    res_prim = wasserstein_feature(
        stats=stats,
        feature_name="num_atoms_prim (primitive)",
        extract_fn=extract_fn("num_atoms_prim"),
    )

    return {
        "conv": res_conv,
        "prim": res_prim,
    }

def wasserstein_num_orbits(stats: dict, key: str = "num_orbits") -> dict:
    """
    Compute Wasserstein distance for number of symmetry orbits per crystal.

    Parameters
    ----------
    stats : dict
        {
            "train": train_stats,
            "new": new_stats
        }
    key : str
        Usually "num_orbits".
    """
    def extract_fn(split_stats: dict) -> list[float]:
        values = _to_list(split_stats[key])
        return [float(v) for v in values]

    return wasserstein_feature(
        stats=stats,
        feature_name="num_orbits",
        extract_fn=extract_fn,
    )
