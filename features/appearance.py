import numpy as np

def compute_xy(tracks, features=None, ctr_ind=1, fwd_ind=0, pxpermm=10.5, **kwargs):
    x = tracks[:, ctr_ind, 0, :] / float(pxpermm)
    y = tracks[:, ctr_ind, 1, :] / float(pxpermm)

    return {
        "x_mm": x.astype(np.float64),
        "y_mm": y.astype(np.float64),
    }

# a: Quarter major axis length, b: Quarter minor axis length
def compute_ab(tracks, features=None, ctr_ind=1, fwd_ind=0, abdomen_idx = 2, leftW_idx=3, rightW_idx=4, pxpermm=10.5, **kwargs):
    a = (np.sqrt(np.sum((tracks[:,fwd_ind,:,:] - tracks[:,abdomen_idx,:,:])**2, axis=1)) / 4).astype(np.float64)
    b = (np.sqrt(np.sum((tracks[:,leftW_idx,:,:] - tracks[:,rightW_idx,:,:])**2, axis=1)) / 4).astype(np.float64)

    a_mm = a / float(pxpermm)
    b_mm = b / float(pxpermm)

    return {
        "a_mm": a_mm,
        "b_mm": b_mm,
    }

# da:  Change in quarter major axis length from frame t to t+1,
# db:  Change in quarter minor axis length from frame t to t+1
def compute_dab(tracks, features=None, ctr_ind=1, fwd_ind=0, fps=30, **kwargs):
    a_mm = features["a_mm"]
    b_mm = features["b_mm"]

    da = np.diff(a_mm, axis=0, prepend=np.nan) * fps
    db = np.diff(b_mm, axis=0, prepend=np.nan) * fps

    return {
        "da": da.astype(np.float64),
        "db": db.astype(np.float64),
    }

# Area of the ellipse
def compute_area(tracks, features=None, ctr_ind=1, fwd_ind=0, **kwargs):
    a_mm = features["a_mm"]
    b_mm = features["b_mm"]

    area = np.pi * (2 * a_mm) * (2 * b_mm)
    return {
        "area": area.astype(np.float64),
    }

# Change in area from frame t to t+1
def compute_darea(tracks, features=None, ctr_ind=1, fwd_ind=0, fps=30, **kwargs):
    area = features["area"]
    darea = np.diff(area, axis=0, prepend=np.nan) * fps
    return {
        "darea": darea.astype(np.float64),
    }

#  Eccentricity of the ellipse
def compute_ecc(tracks, features=None, ctr_ind=1, fwd_ind=0, **kwargs):
    a_mm = features["a_mm"]
    b_mm = features["b_mm"]

    ecc = b_mm / np.maximum(a_mm, 1e-6)
    # ecc = np.sqrt(1 - (b_mm / a_mm) ** 2)
    return {
        "ecc": ecc.astype(np.float64),
    }

# Change in the eccentricity of the ellipse from frame t to t+1
def compute_decc(tracks, features=None, ctr_ind=1, fwd_ind=0, fps=30, **kwargs):
    ecc = features["ecc"]
    decc = np.diff(ecc, axis=0, prepend=np.nan) * fps
    return {
        "decc": decc.astype(np.float64),
    }

def compute_nose_tail(tracks, features=None, ctr_ind=1, pxpermm=10.5, **kwargs):
    """
    World-space nose position in mm.
    Nose = centroid + 2*a along forward heading (theta).
    Tail = centroid + 2*a along backward heading (-theta).
    """
    x_mm  = features["x_mm"]
    y_mm  = features["y_mm"]
    a_mm  = features["a_mm"]
    theta = features["theta"]

    return {
        "nose_x_mm": (x_mm + 2 * a_mm * np.cos(theta)).astype(np.float64),
        "nose_y_mm": (y_mm + 2 * a_mm * np.sin(theta)).astype(np.float64),
        "tail_x_mm": (x_mm - 2 * a_mm * np.cos(theta)).astype(np.float64),
        "tail_y_mm": (y_mm - 2 * a_mm * np.sin(theta)).astype(np.float64),
    }