import numpy as np

from preprocessing import fill_missing

def _compute_wing_angles_from_ego(ego_xy, left_ind=3, right_ind=4):
    """Returns the wing angles in degrees from normalized pose.

    Args:
        ego_xy: array (time, nodes, 2), egocentric, where forward axis is +X.
            raw pose coordinates before passing to this function.
        left_ind: Index of the left wing. Defaults to 3.
        right_ind: Index of the right wing. Defaults to 4.

    Returns:
        Arrays (time,), in degrees in range [-180, 180]

        Tuple of (thetaL, thetaR) containing the left and right wing angles.

        Both are in the range [-180, 180], where 0 is when the wings are exactly aligned
        to the midline (thorax to head axis).

        Positive angles denote extension away from the midline in the direction of the
        wing. For example, a right wing extension may have thetaR > 0.
    """
    xL, yL = ego_xy[:, left_ind, 0], ego_xy[:, left_ind, 1]
    xR, yR = ego_xy[:, right_ind, 0], ego_xy[:, right_ind, 1]

    thetaL = np.rad2deg(np.arctan2(yL, xL)) + 180
    thetaL[np.greater(thetaL, 180, where=np.isfinite(thetaL))] -= 360

    thetaR = np.rad2deg(np.arctan2(yR, xR)) + 180
    thetaR[np.greater(thetaR, 180, where=np.isfinite(thetaR))] -= 360
    thetaR = -thetaR

    return thetaL, thetaR

def feature_wingLR(tracks, features, left_ind=3, right_ind=4, **kwargs):
    """
    Compute left and right wing angles for all individuals.

    Requires:
        features["ego_tracks"] of shape (time, nodes, 2, fly)

    Returns:
        wingL, wingR: both (time, fly)
    """
    ego = features["ego_tracks"]
    n_frames, _, _, n_flies = ego.shape

    wingL = np.full((n_frames, n_flies), np.nan)
    wingR = np.full((n_frames, n_flies), np.nan)

    for fly in range(n_flies):
        thetaL, thetaR = _compute_wing_angles_from_ego(ego[..., fly], left_ind=left_ind, right_ind=right_ind)
        wingL[:, fly] = thetaL
        wingR[:, fly] = thetaR

    return {"wingL": wingL, "wingR": wingR}


def feature_minmax_wing_angle(tracks, features, **kwargs):
    """
    Per-frame min and max wing angle across left and right wing, for each individual.

    Requires:
        features["wingL"], features["wingR"] of shape (time, fly)

    Returns:
        dict with:
            "minWingAng": (time, fly)
            "maxWingAng": (time, fly)
    """
    if "wingL" not in features or "wingR" not in features:
        raise KeyError(
            "feature_minmax_wing_angle requires 'wingL' and 'wingR' in features. "
            f"Available keys: {sorted(features.keys())}"
        )

    wingL = features["wingL"]
    wingR = features["wingR"]

    # Stack to shape (time, fly, 2) then reduce over last axis
    W = np.stack([wingL, wingR], axis=-1)

    minWingAng = np.nanmin(W, axis=-1)
    maxWingAng = np.nanmax(W, axis=-1)

    wingAmp = maxWingAng - minWingAng

    return {"minWingAng": minWingAng, "maxWingAng": maxWingAng, "wingAmp": wingAmp}


def _compute_wing_arc_angles(XB, XA, ctr_ind=1, fwd_ind=0, left_ind=3, right_ind=4,):
    """
    Same math as original compute_wing_arc_angles, but XA is already aligned per time.
    XB, XA: (time, nodes, 2)
    Returns: arcThetaL, arcThetaR of shape (time,)
    """
    # Fill missing values
    XB_Th = fill_missing(XB[:, ctr_ind], kind="nearest")    # thorax
    XB_WL = fill_missing(XB[:, left_ind], kind="nearest")   # left wing
    XB_WR = fill_missing(XB[:, right_ind], kind="nearest")  # right wing

    XA_H = fill_missing(XA[:, fwd_ind], kind="nearest")     # head

    # Compute wing midpoints
    XB_WRm = (XB_Th + XB_WR) / 2
    XB_WLm = (XB_Th + XB_WL) / 2

    # Compute offset of wing midpoint to tip
    XB_WRm_to_WR = XB_WR - XB_WRm
    XB_WLm_to_WL = XB_WL - XB_WLm

    # Compute angle of relative midpoint offset
    angWR = np.rad2deg(np.arctan2(XB_WRm_to_WR[:, 1], XB_WRm_to_WR[:, 0])) % 360
    angWL = np.rad2deg(np.arctan2(XB_WLm_to_WL[:, 1], XB_WLm_to_WL[:, 0])) % 360

    # Right wing arc
    A = XA_H - XB_WRm
    B = np.stack([np.cos(np.deg2rad(angWR - 90)), np.sin(np.deg2rad(angWR - 90))], axis=-1)
    denom = (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))
    C = np.sum(A * B, axis=-1) / denom
    arcThetaR = np.rad2deg(np.arccos(np.clip(C, -1, 1)))

    # Left wing arc
    A = XA_H - XB_WLm
    B = np.stack([np.cos(np.deg2rad(angWL + 90)), np.sin(np.deg2rad(angWL + 90))], axis=-1)
    denom = (np.linalg.norm(A, axis=-1) * np.linalg.norm(B, axis=-1))
    C = np.sum(A * B, axis=-1) / denom
    arcThetaL = np.rad2deg(np.arccos(np.clip(C, -1, 1)))

    return arcThetaL.astype(np.float32), arcThetaR.astype(np.float32)


def feature_wing_arc_to_nearest(tracks, features, **kwargs):
    """
    Reproduce compute_wing_arc_angles, but for each fly relative to its nearest neighbor
    at each frame.

    Returns:
        arcThetaL_to_nearest: (time, fly)
        arcThetaR_to_nearest: (time, fly)
    """
    nearest = features["nearestFlyIdx"]   # (time, fly)
    valid = features.get("nearest_valid", nearest >= 0)

    n_frames, n_nodes, _, n_flies = tracks.shape
    outL = np.full((n_frames, n_flies), np.nan, dtype=np.float32)
    outR = np.full((n_frames, n_flies), np.nan, dtype=np.float32)

    for i in range(n_flies):
        ok = valid[:, i]
        if not np.any(ok):
            continue

        t = np.where(ok)[0]
        j = nearest[ok, i].astype(int)

        XB = tracks[t, :, :, i]     # (n_valid, nodes, 2)
        XA = tracks[t, :, :, j]     # (n_valid, nodes, 2), time-aligned by indexing with j

        arcL, arcR = _compute_wing_arc_angles(XB, XA)
        outL[t, i] = arcL
        outR[t, i] = arcR

    return {
        "arcThetaL_to_nearest": outL,
        "arcThetaR_to_nearest": outR,
    }