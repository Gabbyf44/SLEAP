from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional

from features.ego import compute_ego_tracks, feature_ego_rel_nearest
from features.wings import feature_wingLR, feature_minmax_wing_angle, feature_wing_arc_to_nearest
from features.kinematics import compute_individual_kinematics
from features.pairwise import *

@dataclass
class FeatureSpec:
    func: Callable
    requires: List[str]
    outputs: List[str]  # keys to save
    units: Dict[str, Dict[str, str]] = field(default_factory=dict)  # per-output metadata
    enabled: bool = True


REGISTRY = {
    "ego_tracks": FeatureSpec(
        func=compute_ego_tracks,
        requires=[],
        outputs=[],
        units={},
        enabled=True,
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
    ),
    "nearest_neighbor": FeatureSpec(
        func=feature_nearest_neighbor,
        requires=[],
        outputs=["minDist","nearestFlyIdx"],
        units={
            "minDist": {"quantity": "distance", "unit_raw": "px", "unit_si": "mm", "scale_expr": "1/pxpermm"},
            "nearestFlyIdx": {"quantity": "index", "unit_raw": "unitless", "unit_si": "unitless", "scale_expr": "1"},
        },
        enabled=True,
    ),
    "nearest_geom": FeatureSpec(
        func=feature_nearest_geom,
        requires=["nearest_neighbor"],
        outputs=[],
        units={},
        enabled=True,
    ),
    "ego_rel_nearest": FeatureSpec(
        func=feature_ego_rel_nearest,
        requires=["nearest_neighbor"],
        outputs=[],
        units={},
        enabled=True,
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
    ),
    "ang_to_nearest": FeatureSpec(
        func=feature_ang_to_nearest,
        requires=["nearest_geom"],
        outputs=["ang_to_nearest"],
        units={
            "ang_to_nearest": {"quantity":"angle","unit_raw":"deg","unit_si":"deg","scale_expr":"1"},
        },
        enabled=True,
    ),
}


def get_units_for_key(key: str, registry: Dict[str, FeatureSpec]) -> Optional[Dict[str, str]]:
    """Find units metadata for a saved dataset name."""
    for spec in registry.values():
        if hasattr(spec, "units") and key in spec.units:
            return spec.units[key]
    return None


def compute_item(name, tracks, registry, computed, **kwargs):
    if name in computed:
        return computed[name]

    spec = registry[name]
    for dep in spec.requires:
        compute_item(dep, tracks, registry, computed, **kwargs)

    value = spec.func(tracks, features=computed, **kwargs)

    if isinstance(value, dict):
        computed[name] = value
        for k, v in value.items():
            computed[k] = v
    else:
        computed[name] = value

    return computed[name]