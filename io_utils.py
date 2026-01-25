import numpy as np
import h5py


def load_tracks(track_file):
    """Load proofread and exported pose tracks.
    Args:
        track_file: Path to a SLEAP '*.analysis.h5' file containing tracked poses.
    Returns:
        tracks: NumPy array of shape (time, joints, 2, n_flies) with pose
            coordinates for all tracked individuals.
        node_names contains a list of string names for the joints.
        track_names: Array of track identifiers defining the order of
            individuals along the fly axis.
    """
    with h5py.File(track_file, "r") as f:
        tracks = np.transpose(f["tracks"][:])  # (frame, joint, xy, fly)
        node_names = f["node_names"][:]
        node_names = [x.decode() for x in node_names]
        track_names = f["track_names"][:]

    # Crop to valid range.
    last_fidx = np.argwhere(np.isfinite(tracks.reshape(len(tracks), -1)).any(axis=-1)).squeeze()[-1]
    tracks = tracks[:last_fidx]

    return tracks, node_names, track_names


def encode_hdf5_strings(S):
    """Encodes a list of strings for writing to a HDF5 file.

    Args:
        S: List of strings.

    Returns:
        List of numpy arrays that can be written to HDF5.
    """
    return [np.bytes_(x) for x in S]


def units_for_key(key: str, registry) -> dict | None:
    for spec in registry.values():
        if key in spec.units:
            return spec.units[key]
    return None