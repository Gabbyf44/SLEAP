import numpy as np
from preprocessing import normalize_to_egocentric


def compute_ego_tracks(tracks, features=None, ctr_ind=1, fwd_ind=0, **kwargs):
    n_frames, n_nodes, _, n_flies = tracks.shape
    ego_tracks = np.zeros_like(tracks)
    for fly in range(n_flies):
        ego_tracks[..., fly] = normalize_to_egocentric(
            tracks[..., fly], ctr_ind=ctr_ind, fwd_ind=fwd_ind
        )
    return ego_tracks

def feature_ego_rel_nearest(tracks, features=None, ctr_ind=1, fwd_ind=0, **kwargs):
    """
    Egocentric pose of each fly relative to its nearest neighbor, per frame.
    Uses normalize_to_egocentric by constructing a time-varying rel_to.
    """
    if "nearestFlyIdx" not in features:
        raise KeyError("feature_ego_rel_nearest requires 'nearestFlyIdx' in features")

    nearest = features["nearestFlyIdx"]  # (time, fly)
    n_frames, n_nodes, _, n_flies = tracks.shape

    ego_rel_nearest = np.full((n_frames, n_nodes, 2, n_flies), np.nan, dtype=np.float32)

    for i in range(n_flies):
        x = tracks[..., i]  # (time, nodes, 2)

        rel_to = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float32)
        j_idx = nearest[:, i]

        valid = j_idx >= 0
        if not np.any(valid):
            continue

        t_idx = np.where(valid)[0]
        j = j_idx[valid].astype(int)

        # build per-frame rel_to from the nearest fly
        rel_to[t_idx] = tracks[t_idx, :, :, j]

        ego_i = normalize_to_egocentric(x, rel_to=rel_to, ctr_ind=ctr_ind, fwd_ind=fwd_ind)
        ego_rel_nearest[..., i] = ego_i

    return ego_rel_nearest