import numpy as np
import h5py
from perframe.ds_utils import iter_datasets, safe_savemat

def export_perframe(features_h5, perframe_dir, overwrite) -> None:
    if not features_h5.is_file():
        raise FileNotFoundError(features_h5)

    # overwrite
    if perframe_dir.exists():
        if overwrite:
            for p in perframe_dir.rglob("*.mat"):
                p.unlink(missing_ok=True)
        else:
            raise FileExistsError(f"{perframe_dir} exists")
    perframe_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(features_h5, "r") as f:
        # collect candidate datasets
        all_ds = list(iter_datasets(f))

        # infer (frames, flies) from first 2D dataset anywhere
        n_frames = None
        n_flies = None
        for path, ds in all_ds:
            if ds.ndim == 2:
                n_frames, n_flies = ds.shape
                break

        if n_frames is None:
            raise RuntimeError("No 2D datasets found in the H5 file")

        # export only datasets exactly (frames, flies)
        for path, ds in all_ds:
            if ds.ndim != 2:
                continue
            if ds.shape != (n_frames, n_flies):
                continue

            values = ds[:]  # (frames, flies)

            data_cell = np.empty((1, n_flies), dtype=object)
            for fly in range(n_flies):
                v = values[:, fly]
                if np.issubdtype(v.dtype, np.floating):
                    v_out = v.astype(np.float64, copy=False)
                else:
                    v_out = v.astype(np.int32, copy=False)
                data_cell[0, fly] = v_out.reshape(1, -1)

            ds_name = path.split("/")[-1]
            out_path = perframe_dir / f"{ds_name}.mat"

            safe_savemat(out_path, {"data": data_cell})