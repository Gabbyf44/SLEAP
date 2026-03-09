import numpy as np

from preprocessing import fill_missing, signed_angle

def compute_individual_kinematics(tracks, features=None, ctr_ind=1, fwd_ind=0, delt=1):
    """
    Compute per-individual kinematic features for all flies, based on the same math as compute_features,
    excluding pairwise and relative terms.

    Args:
        tracks: (time, nodes, 2, fly)
        ctr_ind: thorax index (centroid)
        fwd_ind: head index
        delt: frame offset for rotational speed calculation (as in legacy)

    Returns:
        dict of arrays, each shaped (time, fly):
            FV, FA, LV, LA, LS, RS
    """
    n_frames, _, _, n_flies = tracks.shape

    FV = np.full((n_frames, n_flies), np.nan)
    FA = np.full((n_frames, n_flies), np.nan)
    LV = np.full((n_frames, n_flies), np.nan)
    LA = np.full((n_frames, n_flies), np.nan)
    LS = np.full((n_frames, n_flies), np.nan)
    RS = np.full((n_frames, n_flies), np.nan)

    for fly in range(n_flies):
        thx = fill_missing(tracks[:, ctr_ind, :, fly], kind="nearest")
        hd  = fill_missing(tracks[:, fwd_ind, :, fly], kind="nearest")

        # Velocity vectors (same idea as aV_vec/bV_vec)
        V_vec = np.diff(thx, axis=0)
        V_vec = np.pad(V_vec, ((0, 1), (0, 0)), mode="edge")

        # Body direction (same as aDir_unit/bDir_unit)
        Dir = hd - thx
        Dir_unit = Dir / np.linalg.norm(Dir, axis=1, keepdims=True)

        # Forward velocity and acceleration (same as aFV/bFV and aFA/bFA)
        fv = np.sum(V_vec * Dir_unit, axis=1)
        fa = np.diff(fv, axis=0)
        fa = np.pad(fa, (0, 1), mode="edge")

        # Lateral velocity and acceleration (same as aLV/bLV and aLA/bLA)
        perp = np.stack([-Dir_unit[:, 1], Dir_unit[:, 0]], axis=1)
        lv = np.sum(V_vec * perp, axis=1)
        la = np.diff(lv)
        la = np.pad(la, (0, 1), mode="edge")

        # Lateral speed (same as abs(aLV), abs(bLV))
        ls = np.abs(lv)

        # Rotational speed (same signed_angle approach in compute_features)
        if n_frames >= (delt + 2):
            rs = signed_angle(Dir[0:(-1 - delt), :], Dir[delt:-1, :])
            rs = np.pad(rs, (1, 1), mode="edge")
        else:
            rs = np.full((n_frames,), np.nan)

        FV[:, fly] = fv
        FA[:, fly] = fa
        LV[:, fly] = lv
        LA[:, fly] = la
        LS[:, fly] = ls
        RS[:, fly] = rs

    return {
        "FV": FV,
        "FA": FA,
        "LV": LV,
        "LA": LA,
        "LS": LS,
        "RS": RS,
    }