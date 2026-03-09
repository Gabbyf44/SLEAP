import numpy as np

from preprocessing import fill_missing, signed_angle

def feature_nearest_neighbor(tracks, features=None, ctr_ind=1, **kwargs):
    """
    Compute per-frame nearest-neighbor distance and identity for each fly.

    Args:
        tracks: (time, nodes, 2, fly)
        features:
        ctr_ind: index of centroid node (thorax)

    Returns:
        dict with:
            "minDist": (time, fly) float, minimum distance to any other fly
            "nearestFlyIdx": (time, fly) int, index of nearest fly (0..n_flies-1), -1 if none
    """
    # positions: (time, fly, 2)
    pos = tracks[:, ctr_ind, :, :].transpose(0, 2, 1)

    n_frames, n_flies, _ = pos.shape

    # if only one fly exists, no neighbors
    if n_flies <= 1:
        minDist = np.full((n_frames, n_flies), np.nan, dtype=np.float32)
        nearest = np.full((n_frames, n_flies), -1, dtype=np.int32)
        return {"minDist": minDist, "nearestFlyIdx": nearest}

    # fill missing per fly to reduce NaN explosions
    # uses your existing fill_missing (expects (time, 2))
    for j in range(pos.shape[1]):
        pos[:, j, :] = fill_missing(pos[:, j, :], kind="nearest")

    # pairwise diffs: (time, fly, fly, 2)
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (time, fly, fly)

    # ignore self distance
    diag_idx = np.arange(n_flies)
    dist[:, diag_idx, diag_idx] = np.inf

    # handle any remaining NaNs as inf so they won't be selected as minimum
    dist = np.where(np.isfinite(dist), dist, np.inf)

    minDist = np.min(dist, axis=2)            # (time, fly)
    nearest = np.argmin(dist, axis=2).astype(np.int32)  # (time, fly)

    # if there was no valid other fly (all inf), mark as -1 and minDist as NaN
    no_neighbor = ~np.isfinite(minDist)
    nearest[no_neighbor] = -1
    minDist[no_neighbor] = np.nan

    return {"minDist": minDist, "nearestFlyIdx": nearest}


def feature_nearest_geom(tracks, features, ctr_ind=1, fwd_ind=0, **kwargs):
    if "nearestFlyIdx" not in features:
        raise KeyError("feature_nearest_geom requires 'nearestFlyIdx' in features")

    nearest = features["nearestFlyIdx"]  # (time, fly)
    n_frames, _, _, n_flies = tracks.shape

    # fill required keypoints
    thx_xy = np.zeros((n_frames, 2, n_flies))
    hd_xy = np.zeros((n_frames, 2, n_flies))

    for fly in range(n_flies):
        thx_xy[:, :, fly] = fill_missing(tracks[:, ctr_ind, :, fly], kind="nearest")
        hd_xy[:, :, fly] = fill_missing(tracks[:, fwd_ind, :, fly], kind="nearest")

    # velocity from thorax
    thx_v = np.diff(thx_xy, axis=0)
    thx_v = np.pad(thx_v, ((0, 1), (0, 0), (0, 0)), mode="edge")

    nearest_valid = (nearest >= 0)

    nearest_dir_unit = np.full((n_frames, 2, n_flies), np.nan)
    nearest_dist = np.full((n_frames, n_flies), np.nan)

    # compute direction per fly, vectorized over frames where valid
    for i in range(n_flies):
        ok = nearest_valid[:, i]
        if not np.any(ok):
            continue

        t = np.where(ok)[0]
        j = nearest[ok, i].astype(int)

        # direction: head_i -> thorax_j
        d = thx_xy[t, :, j] - hd_xy[t, :, i]  # (n_valid, 2)

        n = np.linalg.norm(d, axis=1, keepdims=True)
        n[n == 0] = np.nan

        nearest_dir_unit[t, :, i] = d / n
        nearest_dist[t, i] = n.squeeze()

    # perpendicular direction (left)
    nearest_dir_perp = np.stack([-nearest_dir_unit[:, 1, :], nearest_dir_unit[:, 0, :]], axis=1)

    return {
        "nearest_valid": nearest_valid,
        "nearest_dir_unit": nearest_dir_unit,
        "nearest_dir_perp": nearest_dir_perp,
        "thx_xy": thx_xy,
        "hd_xy": hd_xy,
        "thx_v": thx_v,
        "nearest_dist": nearest_dist,
    }

def feature_FV_to_nearest(tracks, features, ctr_ind=1, fwd_ind=0, **kwargs):
    """
    Forward velocity of each fly towards its nearest neighbor.

    This matches the original implementation:
    dot(v_i, unit(head_i -> thorax_nearest))

    Requires (from nearest_geom):
        features["thx_v"]              : (time, 2, fly)
        features["nearest_dir_unit"]   : (time, 2, fly)

    Returns:
        FV_to_nearest : (time, fly)
    """
    V = features["thx_v"]                 # thorax velocity of each fly
    u = features["nearest_dir_unit"]      # unit direction to nearest neighbor

    # projection of velocity onto direction to nearest
    FV = np.sum(V * u, axis=1)            # (time, fly)

    return {"FV_to_nearest": FV.astype(np.float32)}

def feature_relFV_to_nearest(tracks, features, ctr_ind=1, fwd_ind=0, **kwargs):
    """
    Relative forward velocity between a fly and its nearest neighbor.

    Computes:
        dot(v_i - v_j, unit(head_i -> thorax_j))

    Requires (from nearest_geom):
        features["thx_v"]            : (time, 2, fly)
        features["nearestFlyIdx"]    : (time, fly)
        features["nearest_valid"]    : (time, fly)
        features["nearest_dir_unit"] : (time, 2, fly)

    Returns:
        relFV_to_nearest : (time, fly)
    """
    V = features["thx_v"]
    nearest = features["nearestFlyIdx"]
    valid = features["nearest_valid"]
    u = features["nearest_dir_unit"]

    n_frames, _, n_flies = V.shape
    out = np.full((n_frames, n_flies), np.nan, dtype=np.float32)

    for i in range(n_flies):
        ok = valid[:, i]
        if not np.any(ok):
            continue

        t = np.where(ok)[0]
        j = nearest[ok, i].astype(int)

        # relative velocity
        v_rel = V[t, :, i] - V[t, :, j]

        out[t, i] = np.sum(v_rel * u[t, :, i], axis=1)

    return {"relFV_to_nearest": out}

def feature_LS_to_nearest(tracks, features, ctr_ind=1, fwd_ind=0, **kwargs):
    """
    Lateral speed of each fly relative to its nearest neighbor.

    Mirrors the original abLS/baLS logic:
        LS = | dot(v_i, perp(unit(head_i -> thorax_nearest))) |

    Requires (from nearest_geom):
        features["thx_v"]             : (time, 2, fly)
        features["nearest_dir_perp"]  : (time, 2, fly)

    Returns:
        LS_to_nearest : (time, fly)
    """
    V = features["thx_v"]                 # thorax velocity of each fly
    perp = features["nearest_dir_perp"]

    LS = np.abs(np.sum(V * perp, axis=1))  # (time, fly)

    return {"LS_to_nearest": LS.astype(np.float32)}


def feature_relLS_to_nearest(tracks, features, **kwargs):
    """
    Relative lateral speed between a fly and its nearest neighbor.

    Computes:
        | dot(v_i - v_j, perp(unit(head_i -> thorax_j))) |

    Requires (from nearest_geom):
        features["thx_v"]              : (time, 2, fly)
        features["nearestFlyIdx"]      : (time, fly)
        features["nearest_valid"]      : (time, fly)
        features["nearest_dir_perp"]   : (time, 2, fly)

    Returns:
        relLS_to_nearest : (time, fly)
    """
    V = features["thx_v"]
    nearest = features["nearestFlyIdx"]
    valid = features["nearest_valid"]
    perp = features["nearest_dir_perp"]

    n_frames, _, n_flies = V.shape
    out = np.full((n_frames, n_flies), np.nan, dtype=np.float32)

    for i in range(n_flies):
        ok = valid[:, i]
        if not np.any(ok):
            continue

        t = np.where(ok)[0]
        j = nearest[ok, i].astype(int)

        # relative velocity
        v_rel = V[t, :, i] - V[t, :, j]

        # lateral projection
        out[t, i] = np.abs(np.sum(v_rel * perp[t, :, i], axis=1))

    return {"relLS_to_nearest": out}


def feature_ang_to_nearest(tracks, features, ctr_ind=1, fwd_ind=0, **kwargs):
    """
    Signed angle between each fly's body axis and the vector pointing to its nearest neighbor.

    Mirrors original:
        abAng = signed_angle(aDir, abDir)
    but with nearest neighbor per-frame.

    Requires (from nearest_geom):
        features["body_dir_unit"]      : (time, 2, fly)
        features["nearest_dir_unit"]   : (time, 2, fly)
        features["nearest_valid"]      : (time, fly)

    Returns:
        ang_to_nearest : (time, fly)
    """
    nearest = features["nearestFlyIdx"]
    valid = features.get("nearest_valid", nearest >= 0)

    n_frames, _, _, n_flies = tracks.shape
    out = np.full((n_frames, n_flies), np.nan, dtype=np.float64)

    thx = np.zeros((n_frames, 2, n_flies), dtype=np.float64)
    hd  = np.zeros((n_frames, 2, n_flies), dtype=np.float64)
    for i in range(n_flies):
        thx[:, :, i] = fill_missing(tracks[:, ctr_ind, :, i], kind="nearest")
        hd[:, :, i]  = fill_missing(tracks[:, fwd_ind, :, i], kind="nearest")

    for i in range(n_flies):
        ok = valid[:, i]
        if not np.any(ok):
            continue
        t = np.where(ok)[0]
        j = nearest[ok, i].astype(int)

        body = hd[t, :, i] - thx[t, :, i]
        toN  = thx[t, :, j] - hd[t, :, i]

        out[t, i] = signed_angle(body, toN)

    return {"ang_to_nearest": out}