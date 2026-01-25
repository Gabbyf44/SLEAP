import os
import h5py
import numpy as np

from io_utils import load_tracks, encode_hdf5_strings, units_for_key
from features.registry import compute_item, REGISTRY
from params import FPS, PXPERMM

def make_expt_dataset(expt_folder, h5_file, output_path=None, overwrite=False, ctr_ind=1, fwd_ind=0):
    """Gather experiment data into a single file.

    Args:
        expt_folder: Full absolute path to the experiment folder.
        h5_file: Path to a SLEAP HDF5 analysis file ('*.analysis.h5').
        output_path: Path to save the resulting dataset to. Can be specified as a folder
            or full path ending with ".h5". Defaults to saving to current folder. If a
            folder is specified, the dataset filename will be the experiment folder
            name with ".h5".
        overwrite: If True, overwrite even if the output path already exists. Defaults
            to False.
        ctr_ind: Index of centroid joint. Defaults to 1.
        fwd_ind: Index of "forward" joint (e.g., head). Defaults to 0.

    Returns:
        Path to output dataset.
    """

    # Resolve output path
    if output_path is None:
        if h5_file.endswith(".analysis.h5"):
            output_path = h5_file.replace(".analysis.h5", ".features.h5")
        else:
            output_path = h5_file + ".features.h5"

    # Overwrite policy
    if os.path.exists(output_path) and not overwrite:
        print(f"output path already exists and overwrite is set to False")
        return output_path

    # Load tracking
    tracks, node_names, track_names = load_tracks(h5_file)
    n_frames, n_nodes, _, n_flies = tracks.shape

    # Basic validation
    if tracks.ndim != 4 or tracks.shape[2] != 2:
        raise ValueError(
            f"Expected tracks with shape (time, joints, 2, fly). Got: {tracks.shape}"
        )

    if n_flies < 1:
        raise ValueError("No tracked individuals found (n_flies < 1).")

    # Ensure node_names length matches nodes axis
    if len(node_names) != n_nodes:
        raise ValueError(
            f"node_names length ({len(node_names)}) does not match nodes axis ({n_nodes})."
        )

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    expt_name = os.path.basename(expt_folder.rstrip("/\\"))
    print(f"\tCreating dataset for: {expt_name}")
    print(f"\tTracks: frames={n_frames}, nodes={n_nodes}, flies={n_flies}")

    computed = {}
    targets = [name for name, spec in REGISTRY.items() if spec.enabled]

    for name in targets:
        compute_item(
            name,
            tracks,
            REGISTRY,
            computed,
            ctr_ind=ctr_ind,
            fwd_ind=fwd_ind,
        )

    # Write output HDF5
    print(f"\tSaving to: {output_path}")

    with h5py.File(output_path, "w") as f:

        # Metadata
        meta = f.create_group("meta")

        meta.create_dataset("expt_name", data=np.bytes_(expt_name))
        meta.create_dataset("expt_folder", data=np.bytes_(expt_folder))
        meta.create_dataset("source_analysis_file", data=np.bytes_(h5_file))
        meta.create_dataset("node_names", data=encode_hdf5_strings(node_names))
        meta.create_dataset("track_names", data=track_names)
        meta.create_dataset("fps", data=FPS)
        meta.create_dataset("pxpermm", data=PXPERMM)

        meta.create_dataset("ctr_ind", data=ctr_ind)
        meta.create_dataset("fwd_ind", data=fwd_ind)

        # Pose
        pose = f.create_group("pose")

        d = pose.create_group("tracks")
        for fly in range(n_flies):
            name = f"fly_{fly:03d}"
            dd = d.create_dataset(name, data=tracks[..., fly], compression=1)
            dd.attrs["axes"] = ["time", "node", "xy"]
            dd.attrs["fly_index"] = fly


        p = pose.create_group("ego_tracks")
        for fly in range(n_flies):
            name = f"fly_{fly:03d}"
            pp = p.create_dataset(name, data=computed["ego_tracks"][..., fly], compression=1)
            pp.attrs["axes"] = ["time", "node", "xy"]
            pp.attrs["fly_index"] = fly

        k = pose.create_group("ego_rel_nearest")
        for fly in range(n_flies):
            name = f"fly_{fly:03d}"
            kk = k.create_dataset(name, data=computed["ego_rel_nearest"][..., fly], compression=1)
            kk.attrs["axes"] = ["time", "node", "xy"]
            kk.attrs["fly_index"] = fly

        save_keys = []
        for t in targets:
            if hasattr(REGISTRY[t], "outputs"):
                save_keys.extend(REGISTRY[t].outputs)

        save_keys = list(dict.fromkeys(save_keys))

        for k in save_keys:
            ds = f.create_dataset(k, data=computed[k], compression=1)

            u = units_for_key(k, REGISTRY)
            if u is not None:
                ds.attrs["quantity"] = u.get("quantity", "")
                ds.attrs["unit_raw"] = u.get("unit_raw", "")
                ds.attrs["unit_si"] = u.get("unit_si", "")
                ds.attrs["scale_expr"] = u.get("scale_expr", "")

    return output_path