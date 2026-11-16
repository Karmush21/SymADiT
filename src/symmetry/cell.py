import numpy as np

def cell_mask_from_sg(sg: int) -> np.ndarray:
    """
    Returns a 6-vector mask over [a,b,c,alpha,beta,gamma] with 1=free, 0=fixed.
    Assumes conventional settings used by your dataset (PyXtal-style),
    and forces trigonal to hex axes (a=b, alpha=beta=90, gamma=120).
    """
    sg = int(sg)
    mask = np.zeros(6, dtype=float)

    if 195 <= sg <= 230:          # cubic
        mask[[0]] = 1.0           # a
    elif 168 <= sg <= 194:        # hexagonal
        mask[[0, 2]] = 1.0        # a, c
    elif 143 <= sg <= 167:        # trigonal (FORCE HEX SETTING)
        mask[[0, 2]] = 1.0        # a, c
    elif 75 <= sg <= 142:         # tetragonal
        mask[[0, 2]] = 1.0        # a, c
    elif 16 <= sg <= 74:          # orthorhombic
        mask[[0, 1, 2]] = 1.0     # a, b, c
    elif 3 <= sg <= 15:           # monoclinic (unique axis b conventional)
        mask[[0, 1, 2, 4]] = 1.0  # a, b, c, beta
    elif 1 <= sg <= 2:            # triclinic
        mask[[0, 1, 2, 3, 4, 5]] = 1.0  # a,b,c,alpha,beta,gamma
    else:
        raise ValueError(f"Invalid space group: {sg}")

    return mask