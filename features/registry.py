from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Literal
import numpy as np

from features.ego import compute_ego_tracks, feature_ego_rel_nearest
from features.wings import feature_wingLR, feature_minmax_wing_angle, feature_wing_arc_to_nearest
from features.kinematics import compute_individual_kinematics
from features.pairwise import (
    feature_nearest_neighbor,
    feature_nearest_geom,
    feature_FV_to_nearest,
    feature_relFV_to_nearest,
    feature_LS_to_nearest,
    feature_relLS_to_nearest,
    feature_ang_to_nearest
)

@dataclass
class FeatureSpec:
    func: Callable
    requires: List[str]
    outputs: List[str]  # keys to save
    units: Dict[str, Dict[str, str]] = field(default_factory=dict)  # per-output metadata
    enabled: bool = True
    save_mode: Literal["scalar", "pose_per_fly", "none"] = "scalar"
    # "scalar"       -> shape (n_frames, n_flies), written as a flat dataset
    #                   via the existing save_keys loop in dataset.py
    # "pose_per_fly" -> shape (n_frames, nodes, 2, n_flies), written as
    #                   per-fly sub-datasets under pose/<registry_key>/fly_NNN
    # "none"         -> intermediate only, not written to HDF5
    intermediates: List[str] = field(default_factory=list)
    # Keys returned by the function that are available in `computed` for
    # downstream features but are NOT written to HDF5.

def validate_registry(registry: Dict[str, FeatureSpec]) -> None:
    errors = []
    seen_outputs: Dict[str, str] = {}
    seen_intermediates: Dict[str, str] = {}

    for name, spec in registry.items():

        # Check 1: valid requires keys (unchanged)
        for dep in spec.requires:
            if dep not in registry:
                errors.append(
                    f"  '{name}' requires '{dep}', which is not a registered key.\n"
                    f"    Available keys: {sorted(registry.keys())}"
                )

        # Check 2: no duplicate output keys
        for out_key in spec.outputs:
            if out_key in seen_outputs:
                errors.append(
                    f"  '{name}' declares output '{out_key}', already declared "
                    f"by '{seen_outputs[out_key]}'."
                )
            else:
                seen_outputs[out_key] = name

        # Check 2b: no duplicate intermediate keys
        for int_key in spec.intermediates:
            if int_key in seen_intermediates:
                errors.append(
                    f"  '{name}' declares intermediate '{int_key}', already declared "
                    f"by '{seen_intermediates[int_key]}'."
                )
            else:
                seen_intermediates[int_key] = name

        # Check 2c: intermediates must not overlap with outputs
        for int_key in spec.intermediates:
            if int_key in seen_outputs:
                errors.append(
                    f"  '{name}' declares '{int_key}' as an intermediate, but it is "
                    f"already declared as an output by '{seen_outputs[int_key]}'."
                )

        # Check 3: save_mode consistency
        if spec.save_mode == "pose_per_fly" and len(spec.outputs) == 0:
            errors.append(
                f"  '{name}' has save_mode='pose_per_fly' but outputs=[].\n"
                f"    Declare the output key(s) this feature writes."
            )
        if spec.save_mode == "scalar" and len(spec.outputs) == 0 and spec.enabled:
            errors.append(
                f"  '{name}' has save_mode='scalar' (default) but outputs=[].\n"
                f"    Set save_mode='none' explicitly if this entry is intentionally intermediate-only."
            )
        # Check 4: no self-dependency
        if name in spec.requires:
            errors.append(
                f"  '{name}' lists itself in requires. Self-dependencies cause "
                f"infinite recursion in compute_item()."
            )

    # Check 5: registry key names do not collide with output key names (unchanged)
    for name in registry.keys():
        if name in seen_outputs and seen_outputs[name] != name:
            errors.append(
                f"  Registry key '{name}' is also declared as an output key by "
                f"'{seen_outputs[name]}'. This causes a collision in the computed dict."
            )

    if errors:
        raise ValueError(
            f"Feature registry validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors)
        )


REGISTRY = {
    "ego_tracks": FeatureSpec(
        func=compute_ego_tracks,
        requires=[],
        outputs=["ego_tracks"],
        units={},
        enabled=True,
        save_mode="pose_per_fly",
    ),
    "kinematics": FeatureSpec(
        func=compute_individual_kinematics,
        requires=[],
        outputs=["FV","FA","LV","LA","LS","RS"],
        units={
            "FV": {
                "quantity": "velocity",
                "unit_raw": "px/frame",
                "unit_si": "mm/sec",
                "scale_expr": "fps/pxpermm",
            },
            "FA": {
                "quantity": "acceleration",
                "unit_raw": "px/frame^2",
                "unit_si": "mm/sec^2",
                "scale_expr": "fps^2/pxpermm",
            },
            "LV": {
                "quantity": "velocity",
                "unit_raw": "px/frame",
                "unit_si": "mm/sec",
                "scale_expr": "fps/pxpermm",
            },
            "LA": {
                "quantity": "acceleration",
                "unit_raw": "px/frame^2",
                "unit_si": "mm/sec^2",
                "scale_expr": "fps^2/pxpermm",
            },
            "LS": {
                "quantity": "velocity",
                "unit_raw": "px/frame",
                "unit_si": "mm/sec",
                "scale_expr": "fps/pxpermm",
            },
            "RS": {
                "quantity": "rot_speed",
                "unit_raw": "deg/frame",
                "unit_si": "deg/sec",
                "scale_expr": "fps",
            },
        },
        enabled=True,
        save_mode="scalar"
    ),
    "wing": FeatureSpec(
        func=feature_wingLR,
        requires=["ego_tracks"],
        outputs=["wingL","wingR"],
        units={
            "wingL": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
            "wingR": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "wing_minmax": FeatureSpec(
        func=feature_minmax_wing_angle,
        requires=["wing"],
        outputs=["minWingAng","maxWingAng","wingAmp"],
        units={
            "minWingAng": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
            "maxWingAng": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
            "wingAmp": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "nearest_neighbor": FeatureSpec(
        func=feature_nearest_neighbor,
        requires=[],
        outputs=["minDist"],
        intermediates=["nearestFlyIdx"],
        units={
            "minDist": {"quantity": "distance", "unit_raw": "px", "unit_si": "mm", "scale_expr": "1/pxpermm"},
            "nearestFlyIdx": {"quantity": "index", "unit_raw": "unitless", "unit_si": "unitless", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "nearest_geom": FeatureSpec(
        func=feature_nearest_geom,
        requires=["nearest_neighbor"],
        outputs=[],
        intermediates=[
            "thx_xy", "hd_xy", "thx_v",
            "nearest_dir_unit", "nearest_dir_perp",
            "nearest_dist", "nearest_valid",
        ],
        units={},
        enabled=True,
        save_mode="none",
    ),
    "ego_rel_nearest": FeatureSpec(
        func=feature_ego_rel_nearest,
        requires=["nearest_neighbor"],
        outputs=["ego_rel_nearest"],
        units={},
        enabled=True,
        save_mode="pose_per_fly",
    ),
    "fv_to_nearest": FeatureSpec(
        func=feature_FV_to_nearest,  # uses thx_v + nearest_dir_unit
        requires=["nearest_geom"],
        outputs=["FV_to_nearest"],
        units={
            "FV_to_nearest": {"quantity": "velocity", "unit_raw": "px/frame", "unit_si": "mm/sec",
                              "scale_expr": "fps/pxpermm"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "relfv_to_nearest": FeatureSpec(
        func=feature_relFV_to_nearest,  # uses thx_v + nearestFlyIdx + nearest_valid
        requires=["nearest_geom"],
        outputs=["relFV_to_nearest"],
        units={
            "relFV_to_nearest": {"quantity": "velocity", "unit_raw": "px/frame", "unit_si": "mm/sec",
                                 "scale_expr": "fps/pxpermm"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "ls_to_nearest": FeatureSpec(
        func=feature_LS_to_nearest,
        requires=["nearest_geom"],
        outputs=["LS_to_nearest"],
        units={
            "LS_to_nearest": {"quantity": "velocity", "unit_raw": "px/frame", "unit_si": "mm/sec",
                              "scale_expr": "fps/pxpermm"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "relLS_to_nearest": FeatureSpec(
        func=feature_relLS_to_nearest,
        requires=["nearest_geom"],
        outputs=["relLS_to_nearest"],
        units={
            "relLS_to_nearest": {"quantity": "velocity", "unit_raw": "px/frame", "unit_si": "mm/sec",
                                 "scale_expr": "fps/pxpermm"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "wing_arc_to_nearest": FeatureSpec(
        func=feature_wing_arc_to_nearest,
        requires=["nearest_neighbor"],
        outputs=["arcThetaL_to_nearest", "arcThetaR_to_nearest"],
        units={
            "arcThetaL_to_nearest": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
            "arcThetaR_to_nearest": {"quantity": "angle", "unit_raw": "deg", "unit_si": "deg", "scale_expr": "1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
    "ang_to_nearest": FeatureSpec(
        func=feature_ang_to_nearest,
        requires=["nearest_geom"],
        outputs=["ang_to_nearest"],
        units={
            "ang_to_nearest": {"quantity":"angle","unit_raw":"deg","unit_si":"deg","scale_expr":"1"},
        },
        enabled=True,
        save_mode="scalar"
    ),
}

# Validate registry consistency at import time.
# This raises ValueError immediately if any dependency is missing or
# any output key is declared more than once.
validate_registry(REGISTRY)

def get_units_for_key(key: str, registry: dict[str, FeatureSpec]) -> dict[str, str] | None:
    """Return units metadata dict for a saved output key, or None if not found."""
    for spec in registry.values():
        if key in spec.units:
            return spec.units[key]
    return None


def compute_item(
    name: str,
    tracks: np.ndarray,
    registry: dict[str, FeatureSpec],
    computed: dict[str, object],
    **kwargs: object,
) -> None:
    _COMPUTED_SENTINEL = f"__computed_{name}__"
    if _COMPUTED_SENTINEL in computed:
        return

    spec = registry[name]

    if not spec.enabled:
        raise RuntimeError(
            f"Feature '{name}' is disabled (enabled=False) but was requested "
            f"either directly or as a dependency. "
            f"To resolve this, either re-enable '{name}' or also disable the "
            f"features that depend on it."
        )

    for dep in spec.requires:
        compute_item(dep, tracks, registry, computed, **kwargs)

    value = spec.func(tracks, features=computed, **kwargs)

    if spec.outputs:
        if not isinstance(value, dict):
            raise TypeError(
                f"Feature '{name}' declares outputs {spec.outputs} but its function "
                f"returned {type(value).__name__} instead of a dict. "
                f"Functions with non-empty outputs must return a dict."
            )
        missing = [k for k in spec.outputs if k not in value]
        if missing:
            raise KeyError(
                f"Feature '{name}' declared output key(s) {missing} but its function "
                f"did not return them. "
                f"Returned keys: {sorted(value.keys())}"
            )

    # Store only the individual output keys
    if isinstance(value, dict):
        for k, v in value.items():
            computed[k] = v
    else:
        # Bare-array return: only valid when outputs=[]
        computed[name] = value

    # Mark this item as computed so repeated calls are no-ops.
    computed[_COMPUTED_SENTINEL] = True