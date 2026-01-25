from pathlib import Path
import h5py
import scipy.io

def iter_datasets(h5obj, prefix=""):
    """
    Yield (full_path, dataset) for all datasets under h5 obj (recursive).
    full_path uses POSIX separators: group/subgroup/name
    """
    for key in h5obj.keys():
        obj = h5obj[key]
        full = f"{prefix}{key}" if prefix == "" else f"{prefix}/{key}"
        if isinstance(obj, h5py.Dataset):
            yield full, obj
        elif isinstance(obj, h5py.Group):
            yield from iter_datasets(obj, prefix=full)


def safe_savemat(dest: Path, data_dict: dict) -> None:
    """safely saves a python dict into a mat file

    Args:
        dest (Path): mat file to save to
        data_dict (dict): dictionary to save
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        scipy.io.savemat(tmp, data_dict)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    else:
        dest.unlink(missing_ok=True)
        tmp.rename(dest)