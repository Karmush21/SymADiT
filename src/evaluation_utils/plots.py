from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from collections import defaultdict, Counter



def plot_atom_type_counts(
    stats,
    save_dir: str | None = None,
    sg: int | None = None,          # optional: filter by spacegroup
    dpi: int = 150,
):
    """
    Plot atom-type (Z) counts.
    - stats[split]["atom_types"] is expected to be: List[List[int]] (per material, per orbit).
    - If sg is provided: only materials with that spacegroup contribute.
    - Train (blue) on top, New (orange) below.
    - Auto-saves into save_dir if provided.
    """

    def collect_atom_z(split: str):
        sgs = stats[split]["spacegroup"]        # [N]
        per_mat_z = stats[split]["atom_types"]  # [N] of List[int]
        out = []
        for sgi, zi in zip(sgs, per_mat_z):
            if (sg is None) or (int(sgi) == int(sg)):
                out.extend([int(z) for z in zi])
        return out

    train_z = collect_atom_z("train")
    new_z   = collect_atom_z("new")

    c_train = Counter(train_z)
    c_new   = Counter(new_z)

    # X-axis support = union of observed Z across train+new
    x_vals = sorted(set(c_train) | set(c_new))
    if not x_vals:
        print(f"[WARN] No atom types found (sg={sg}). Nothing to plot.")
        return None

    y_train = [c_train.get(z, 0) for z in x_vals]
    y_new   = [c_new.get(z, 0) for z in x_vals]

    title_suffix = f" (SG {sg})" if sg is not None else ""

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6), constrained_layout=True)

    axes[0].bar(x_vals, y_train, color="tab:blue")
    axes[0].set_title(f"train — atom_types{title_suffix}")
    axes[0].set_ylabel("count")

    axes[1].bar(x_vals, y_new, color="tab:orange")
    axes[1].set_title(f"new — atom_types{title_suffix}")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("atomic number (Z)")

    # Treat Z as numeric axis
    xmin, xmax = min(x_vals), max(x_vals)
    span = xmax - xmin

    # choose reasonable tick spacing
    if span > 80:
        step = 10
    elif span > 40:
        step = 5
    else:
        step = 2

    xticks = list(range(xmin, xmax + 1, step))
    axes[1].set_xticks(xticks)
    axes[1].set_xlim(xmin - 1, xmax + 1)
    axes[1].tick_params(axis="x", rotation=0)

    fig.suptitle(f"atom_types{title_suffix}", y=1.02)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"atom_types_train_vs_new.png" if sg is None else f"sg{sg}_atom_types_train_vs_new.png"
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig


def plot_atom_type_unique_counts(
    stats,
    save_dir: str | None = None,
    sg: int | None = None,          # optional SG filter
    dpi: int = 150,
):
    """
    Plot unique atom-type (Z) counts per structure.

    - stats[split]["atom_types_unique"] must be List[List[int]]
    - Each material contributes at most 1 count per element.
    - If sg is provided: only materials with that spacegroup contribute.
    - Train (blue) on top, New (orange) below.
    """



    def collect_unique_z(split: str):
        sgs = stats[split]["spacegroup"]
        per_mat_unique = stats[split]["atom_types_unique"]
        out = []
        for sgi, zi in zip(sgs, per_mat_unique):
            if (sg is None) or (int(sgi) == int(sg)):
                out.extend([int(z) for z in zi])
        return out

    train_z = collect_unique_z("train")
    new_z   = collect_unique_z("new")

    c_train = Counter(train_z)
    c_new   = Counter(new_z)

    x_vals = sorted(set(c_train) | set(c_new))
    if not x_vals:
        print(f"[WARN] No unique atom types found (sg={sg}).")
        return None

    y_train = [c_train.get(z, 0) for z in x_vals]
    y_new   = [c_new.get(z, 0) for z in x_vals]

    title_suffix = f" (SG {sg})" if sg is not None else ""

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6), constrained_layout=True)

    axes[0].bar(x_vals, y_train, color="tab:blue")
    axes[0].set_title(f"train — atom_types_unique{title_suffix}")
    axes[0].set_ylabel("count (structures)")

    axes[1].bar(x_vals, y_new, color="tab:orange")
    axes[1].set_title(f"new — atom_types_unique{title_suffix}")
    axes[1].set_ylabel("count (structures)")
    axes[1].set_xlabel("atomic number (Z)")

    xmin, xmax = min(x_vals), max(x_vals)
    span = xmax - xmin

    if span > 80:
        step = 10
    elif span > 40:
        step = 5
    else:
        step = 2

    xticks = list(range(xmin, xmax + 1, step))
    axes[1].set_xticks(xticks)
    axes[1].set_xlim(xmin - 1, xmax + 1)

    fig.suptitle(f"atom_types_unique{title_suffix}", y=1.02)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = (
            "atom_types_unique_train_vs_new.png"
            if sg is None
            else f"sg{sg}_atom_types_unique_train_vs_new.png"
        )
        save_path = os.path.join(save_dir, fname)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    return fig


def _plot_atom_count_key(
    stats: dict,
    key: str,
    save_dir: str | None = None,
    filename: str = "num_atoms.png",
    dpi: int = 150,
    max_ticks: int = 15,
    rotate_xticks: int = 45,
):
    train_vals = [int(x) for x in stats.get("train", {}).get(key, [])]
    new_vals   = [int(x) for x in stats.get("new", {}).get(key, [])]

    c_train = Counter(train_vals)
    c_new   = Counter(new_vals)

    x_vals = sorted(set(c_train) | set(c_new))
    if not x_vals:
        print(f"[WARN] No {key} values found in stats.")
        return

    y_train = [c_train.get(x, 0) for x in x_vals]
    y_new   = [c_new.get(x, 0) for x in x_vals]

    xmin, xmax = x_vals[0], x_vals[-1]
    span = xmax - xmin
    step = max(1, int(np.ceil(span / max_ticks))) if span > 0 else 1
    xticks = list(range(xmin, xmax + 1, step))

    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(10, 6), constrained_layout=True
    )

    axes[0].bar(x_vals, y_train, color="tab:blue")
    axes[0].set_title(f"train — {key}")
    axes[0].set_ylabel("count")

    axes[1].bar(x_vals, y_new, color="tab:orange")
    axes[1].set_title(f"new — {key}")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel(key)

    axes[1].set_xticks(xticks)
    axes[1].tick_params(axis="x", rotation=rotate_xticks)
    for lab in axes[1].get_xticklabels():
        lab.set_ha("right")

    fig.suptitle(f"{key} exact-value frequencies", y=1.02)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    plt.close(fig)


def plot_num_atoms(
    stats: dict,
    save_dir: str | None = None,
    dpi: int = 150,
    max_ticks: int = 15,
    rotate_xticks: int = 45,
):
    """
    stats format:
      stats["train"]["num_atoms"]      = list[int]  # conventional
      stats["new"]["num_atoms"]        = list[int]
      stats["train"]["num_atoms_prim"] = list[int]  # primitive
      stats["new"]["num_atoms_prim"]   = list[int]
    """

    _plot_atom_count_key(
        stats=stats,
        key="num_atoms",
        save_dir=save_dir,
        filename="num_atoms.png",
        dpi=dpi,
        max_ticks=max_ticks,
        rotate_xticks=rotate_xticks,
    )

    _plot_atom_count_key(
        stats=stats,
        key="num_atoms_prim",
        save_dir=save_dir,
        filename="num_atoms_prim.png",
        dpi=dpi,
        max_ticks=max_ticks,
        rotate_xticks=rotate_xticks,
    )

def plot_wyckoff_counts_for_sg(
    stats,
    sg: int,
    wyckoff_dict,
    save_dir: str | None = None,
    title_prefix: str = "",
    dpi: int = 150,
):
    """
    Stacked plot like your screenshot:
      - train (top)
      - new   (bottom)

    X-axis is constructed from wyckoff_dict[sg] (canonical support).
    Counts are how many times each Wyckoff type (e.g. "4a") occurs across *materials of that SG*.
    Assumes:
      stats["train"]["spacegroup"] is [N]
      stats["train"]["wyckoff_types"] is [N] of List[str]
      same for stats["new"]
    """
    # -------- build canonical x-axis from WYCKOFF_DICT --------
    entries = wyckoff_dict[int(sg)]
    if isinstance(entries, dict):
        x_vals = list(entries.keys())
    else:
        x_vals = list(entries)
    x_vals = [str(x) for x in x_vals]  # ensure comparable strings

    # -------- collect observed wyckoffs for a given split & SG --------
    def collect(split: str) -> list[str]:
        sgs = stats[split]["spacegroup"]
        wys = stats[split]["wyckoff_types"]
        out = []
        for sgi, wyi in zip(sgs, wys):
            if int(sgi) == int(sg):
                out.extend([str(w) for w in wyi])
        return out

    train_obs = collect("train")
    new_obs   = collect("new")

    c_train = Counter(train_obs)
    c_new   = Counter(new_obs)

    y_train = np.array([c_train.get(w, 0) for w in x_vals], dtype=float)
    y_new   = np.array([c_new.get(w, 0) for w in x_vals], dtype=float)

    # -------- plot: train on top, new below (same as your example) --------
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6), constrained_layout=True)

    axes[0].bar(x_vals, y_train, color="tab:blue")
    axes[0].set_title(f"train — wyckoff_types (SG {sg})")
    axes[0].set_ylabel("count")

    axes[1].bar(x_vals, y_new, color="tab:orange")
    axes[1].set_title(f"new — wyckoff_types (SG {sg})")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("wyckoff_types")

    # rotate x labels like typical categorical plots
    axes[1].tick_params(axis="x", rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")

    fig.suptitle(f"{title_prefix}wyckoff_types (SG {sg})".strip(), y=1.02)

    if save_dir is not None:
        out_path = os.path.join(save_dir, f'sg_{sg}_wyckoffs.png')
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_volume(
    stats: dict,
    save_dir: str | None = None,
    filename: str = "volume.png",
    sg: int | None = None,              # optional: only volumes for this SG
    bins: int = 50,
    dpi: int = 150,
    log_x: bool = False,                # optional: log-scale x-axis
):
    """
    Plot volume distributions (continuous) for train vs new.

    Expected:
      stats["train"]["volume"]     = list[float|int]
      stats["train"]["spacegroup"] = list[int]
      stats["new"]["volume"]       = list[float|int]
      stats["new"]["spacegroup"]   = list[int]

    If sg is provided, filter to materials with that spacegroup.
    Produces a 2-row histogram: train (top), new (bottom).
    """

    def collect(split: str) -> list[float]:
        vols = stats.get(split, {}).get("volume", [])
        if sg is None:
            return [float(v) for v in vols]
        sgs = stats.get(split, {}).get("spacegroup", [])
        out = []
        for sgi, vi in zip(sgs, vols):
            if int(sgi) == int(sg):
                out.append(float(vi))
        return out

    train_vol = collect("train")
    new_vol   = collect("new")

    if len(train_vol) == 0 and len(new_vol) == 0:
        print(f"[WARN] No volume values found (sg={sg}). Nothing to plot.")
        return None

    # Use a shared binning range so train/new are directly comparable
    all_vol = []
    if train_vol: all_vol.extend(train_vol)
    if new_vol:   all_vol.extend(new_vol)

    vmin, vmax = min(all_vol), max(all_vol)
    if vmin == vmax:
        vmax = vmin + 1e-6  # avoid zero-range hist

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), constrained_layout=True)

    # train
    axes[0].hist(train_vol, bins=bins, range=(vmin, vmax))
    title_suffix = f" (SG {sg})" if sg is not None else ""
    axes[0].set_title(f"train — volume{title_suffix}")
    axes[0].set_ylabel("count")

    # new
    axes[1].hist(new_vol, bins=bins, range=(vmin, vmax))
    axes[1].set_title(f"new — volume{title_suffix}")
    axes[1].set_ylabel("count")
    axes[1].set_xlabel("volume")

    if log_x:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")

    fig.suptitle(f"volume distribution{title_suffix}", y=1.02)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir,
            filename if sg is None else f"sg{sg}_" + filename
        )
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    return fig


def plot_num_atoms_per_sg_boxplot(
    stats: dict,
    save_dir: str | None = None,
    filename: str = "num_atoms_per_sg_boxplot.png",
    top_k: int = 20,
    dpi: int = 150,
    showfliers: bool = False,
    rotate_xticks: int = 90,
):
    """
    Boxplot of num_atoms per spacegroup for BOTH splits:
      - train (top)
      - new   (bottom)

    Uses the same SG x-axis for both (chosen from union, ranked by total count).

    Expected:
      stats["train"]["spacegroup"] = list[int]
      stats["train"]["num_atoms"]  = list[int]
      stats["new"]["spacegroup"]   = list[int]
      stats["new"]["num_atoms"]    = list[int]
    """

    def build_sg_to_atoms(split: str) -> dict[int, list[int]]:
        sgs = stats.get(split, {}).get("spacegroup", [])
        nums = stats.get(split, {}).get("num_atoms", [])
        sg_atoms = defaultdict(list)
        for sg, n in zip(sgs, nums):
            sg_atoms[int(sg)].append(int(n))
        return sg_atoms

    sg_atoms_train = build_sg_to_atoms("train")
    sg_atoms_new   = build_sg_to_atoms("new")

    # choose SGs to plot: top_k by total count across both splits
    all_sgs = sorted(set(sg_atoms_train) | set(sg_atoms_new))
    if not all_sgs:
        print("[WARN] No SGs found in stats.")
        return None

    sg_total_counts = {sg: len(sg_atoms_train.get(sg, [])) + len(sg_atoms_new.get(sg, [])) for sg in all_sgs}
    sgs_sorted = sorted(all_sgs, key=lambda sg: sg_total_counts[sg], reverse=True)[:top_k]
    sgs_sorted = sorted(sgs_sorted)  # nicer increasing SG labels like your screenshot

    # data arrays aligned to sgs_sorted (skip empty groups to avoid empty box warnings)
    def aligned_data(sg_atoms: dict[int, list[int]]) -> tuple[list[list[int]], list[int]]:
        data, labels = [], []
        for sg in sgs_sorted:
            vals = sg_atoms.get(sg, [])
            if len(vals) > 0:
                data.append(vals)
                labels.append(sg)
        return data, labels

    data_train, labels_train = aligned_data(sg_atoms_train)
    data_new,   labels_new   = aligned_data(sg_atoms_new)

    if len(data_train) == 0 and len(data_new) == 0:
        print("[WARN] No num_atoms per SG data found (train/new).")
        return None

    # share y-range for easy comparison
    all_vals = []
    for lst in data_train: all_vals.extend(lst)
    for lst in data_new:   all_vals.extend(lst)
    y_min, y_max = min(all_vals), max(all_vals)

    fig, axes = plt.subplots(2, 1, sharex=False, sharey=True, figsize=(12, 8), constrained_layout=True)

    # --- train plot ---
    if len(data_train) > 0:
        axes[0].boxplot(
            data_train,
            tick_labels=[str(sg) for sg in labels_train],  # Matplotlib>=3.9
            showfliers=showfliers,
        )
    axes[0].set_title(f"train — num_atoms per SG (top {top_k} by total count)")
    axes[0].set_ylabel("num_atoms")
    axes[0].set_ylim(y_min - 1, y_max + 1)
    axes[0].tick_params(axis="x", rotation=rotate_xticks)

    # --- new plot ---
    if len(data_new) > 0:
        axes[1].boxplot(
            data_new,
            tick_labels=[str(sg) for sg in labels_new],
            showfliers=showfliers,
        )
    axes[1].set_title(f"new — num_atoms per SG (top {top_k} by total count)")
    axes[1].set_ylabel("num_atoms")
    axes[1].set_xlabel("space group")
    axes[1].set_ylim(y_min - 1, y_max + 1)
    axes[1].tick_params(axis="x", rotation=rotate_xticks)

    fig.suptitle("num_atoms per space group (boxplots)", y=1.02)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out_path

    return fig