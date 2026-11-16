from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon



def p1_rate(stats):
    sg_new = stats["new"].get("spacegroup", [])
    if len(sg_new) == 0:
        return 0.0
    return sum(sg == 1 for sg in sg_new) / len(sg_new)



def _counter_to_prob_arrays(counter_a, counter_b, support=None):
    """
    Convert two Counters into aligned probability arrays.
    """
    if support is None:
        support = sorted(set(counter_a.keys()) | set(counter_b.keys()))

    p = np.array([counter_a.get(x, 0) for x in support], dtype=float)
    q = np.array([counter_b.get(x, 0) for x in support], dtype=float)

    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()

    return support, p, q



def js_global_divergence_discrete_feature(
    stats,
    key,
    base=2,
    preprocess_fn=None,
    support=None,
):
    """
    Compute JSD between train and new distributions for a discrete feature.

    Special handling:
    - if key == "atom_types_unique", each structure is assumed to contain
      a list of unique atom types, and the function automatically converts
      that to "number of unique atom types per structure".

    Parameters
    ----------
    stats : dict
        Dictionary like:
        {
            "train": train_stats,
            "new": new_stats
        }

    key : str
        Feature key, e.g.:
        'spacegroup', 'num_orbits', 'num_atoms', 'atom_types_unique'

    base : float
        Log base for JSD. With base=2, values are in [0, 1].

    preprocess_fn : callable or None
        Optional function applied to stats['train'][key] and stats['new'][key]
        before counting. If provided, this overrides the built-in handling.

    support : list or None
        Optional fixed support. If None, uses union of observed values.

    Returns
    -------
    result : dict
        {
            "feature": key,
            "js_div": ...,
            "js_dist": ...,
            "support": [...],
            "p_train": np.ndarray,
            "p_new": np.ndarray,
            "train_counts": Counter,
            "new_counts": Counter,
        }
    """
    train_vals = stats["train"][key]
    new_vals = stats["new"][key]

    # User-supplied preprocessing takes priority
    if preprocess_fn is not None:
        train_vals = preprocess_fn(train_vals)
        new_vals = preprocess_fn(new_vals)

    # Built-in special handling for atom_types_unique:
    # convert per-structure unique-type lists into counts
    elif key == "atom_types_unique":
        train_vals = [len(x) for x in train_vals]
        new_vals = [len(x) for x in new_vals]

    train_vals = [int(x) for x in train_vals]
    new_vals = [int(x) for x in new_vals]

    train_counts = Counter(train_vals)
    new_counts = Counter(new_vals)

    support, p_train, p_new = _counter_to_prob_arrays(
        train_counts, new_counts, support=support
    )

    if p_train.sum() == 0 or p_new.sum() == 0:
        return {
            "feature": key,
            "js_div": float("nan"),
            "js_dist": float("nan"),
            "support": support,
            "p_train": p_train,
            "p_new": p_new,
            "train_counts": train_counts,
            "new_counts": new_counts,
        }

    js_dist = float(jensenshannon(p_train, p_new, base=base))
    js_div = js_dist ** 2

    return {
        "feature": key,
        "js_div": js_div,
        "js_dist": js_dist,
        "support": support,
        "p_train": p_train,
        "p_new": p_new,
        "train_counts": train_counts,
        "new_counts": new_counts,
    }


from collections import Counter
import numpy as np
from scipy.spatial.distance import jensenshannon


def wyckoff_jsd_over_top_spacegroups(
    stats,
    num_sg=10,
    base=2,
    weighted_average=True,
    flatten=True,
):
    """
    Compute JSD of Wyckoff distributions per space group over the top `num_sg`
    most frequent space groups in the training data.

    Parameters
    ----------
    stats : dict
        Dictionary like:
        {
            "train": train_stats,
            "new": new_stats
        }

        Required keys:
        - stats["train"]["spacegroup"]
        - stats["new"]["spacegroup"]
        - stats["train"]["wyckoff_types"]
        - stats["new"]["wyckoff_types"]

        Assumes wyckoff_types is stored per structure, e.g.
        [
            ["4a", "4b"],
            ["8c"],
            ...
        ]
        or
        [
            [0, 1, 1],
            [2, 3],
            ...
        ]

    num_sg : int
        Number of top space groups to evaluate, chosen from training data.

    base : float
        Log base for JSD. With base=2, values are in [0, 1].

    weighted_average : bool
        If True, also compute weighted average over SGs using the number of
        training structures in each SG as weights.

    flatten : bool
        If True, flatten Wyckoff labels across all structures within a given SG.
        This is usually what you want.

    Returns
    -------
    results : dict
        {
            "top_spacegroups": [...],
            "per_sg": {
                sg: {
                    "js_div": ...,
                    "js_dist": ...,
                    "support": [...],
                    "p_train": np.ndarray,
                    "p_new": np.ndarray,
                    "train_counts": Counter,
                    "new_counts": Counter,
                    "n_train_structures": int,
                    "n_new_structures": int,
                    "n_train_wyckoffs": int,
                    "n_new_wyckoffs": int,
                },
                ...
            },
            "weighted_js_dist": ...,
            "weighted_js_div": ...,
            "simple_mean_js_dist": ...,
            "simple_mean_js_div": ...,
        }
    """
    train_sg = [int(x) for x in stats["train"]["spacegroup"]]
    new_sg = [int(x) for x in stats["new"]["spacegroup"]]

    train_wy = stats["train"]["wyckoff_types"]
    new_wy = stats["new"]["wyckoff_types"]

    # Top SGs from training data
    sg_counts_train = Counter(train_sg)
    top_spacegroups = [sg for sg, _ in sg_counts_train.most_common(num_sg)]

    per_sg = {}

    js_dist_list = []
    js_div_list = []
    weight_list = []

    for sg in top_spacegroups:
        train_idx = [i for i, x in enumerate(train_sg) if x == sg]
        new_idx = [i for i, x in enumerate(new_sg) if x == sg]

        train_wy_sg = [train_wy[i] for i in train_idx]
        new_wy_sg = [new_wy[i] for i in new_idx]

        if flatten:
            train_vals = [w for struct in train_wy_sg for w in struct]
            new_vals = [w for struct in new_wy_sg for w in struct]
        else:
            train_vals = train_wy_sg
            new_vals = new_wy_sg

        train_counts = Counter(train_vals)
        new_counts = Counter(new_vals)

        support = sorted(set(train_counts.keys()) | set(new_counts.keys()))

        p_train = np.array([train_counts.get(x, 0) for x in support], dtype=float)
        p_new = np.array([new_counts.get(x, 0) for x in support], dtype=float)

        if p_train.sum() > 0:
            p_train = p_train / p_train.sum()
        if p_new.sum() > 0:
            p_new = p_new / p_new.sum()

        if p_train.sum() == 0 or p_new.sum() == 0:
            js_dist = float("nan")
            js_div = float("nan")
        else:
            js_dist = float(jensenshannon(p_train, p_new, base=base))
            js_div = js_dist ** 2

            js_dist_list.append(js_dist)
            js_div_list.append(js_div)
            weight_list.append(len(train_idx))

        per_sg[sg] = {
            "js_div": js_div,
            "js_dist": js_dist,
            "support": support,
            "p_train": p_train,
            "p_new": p_new,
            "train_counts": train_counts,
            "new_counts": new_counts,
            "n_train_structures": len(train_idx),
            "n_new_structures": len(new_idx),
            "n_train_wyckoffs": len(train_vals),
            "n_new_wyckoffs": len(new_vals),
        }

    # Aggregates
    if len(js_dist_list) == 0:
        weighted_js_dist = float("nan")
        weighted_js_div = float("nan")
        simple_mean_js_dist = float("nan")
        simple_mean_js_div = float("nan")
    else:
        simple_mean_js_dist = float(np.mean(js_dist_list))
        simple_mean_js_div = float(np.mean(js_div_list))

        if weighted_average and np.sum(weight_list) > 0:
            weights = np.array(weight_list, dtype=float)
            weights = weights / weights.sum()
            weighted_js_dist = float(np.sum(weights * np.array(js_dist_list)))
            weighted_js_div = float(np.sum(weights * np.array(js_div_list)))
        else:
            weighted_js_dist = float("nan")
            weighted_js_div = float("nan")

    return {
        "top_spacegroups": top_spacegroups,
        "per_sg": per_sg,
        "weighted_js_dist": weighted_js_dist,
        "weighted_js_div": weighted_js_div,
        "simple_mean_js_dist": simple_mean_js_dist,
        "simple_mean_js_div": simple_mean_js_div,
    }