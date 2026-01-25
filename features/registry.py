from dataclasses import dataclass
from typing import Callable, List, Dict, Any

from features.ego import compute_ego_tracks, feature_ego_rel_nearest
from features.wings import feature_wingLR, feature_minmax_wing_angle, feature_wing_arc_to_nearest
from features.kinematics import compute_individual_kinematics
from features.pairwise import *

@dataclass
class FeatureSpec:
    func: Callable
    requires: List[str]
    outputs: List[str]  # keys to save
    enabled: bool = True


REGISTRY = {
    "ego_tracks": FeatureSpec(
        func=compute_ego_tracks,
        requires=[],
        outputs=[],
        enabled=True,
    ),
    "kinematics": FeatureSpec(
        func=compute_individual_kinematics,
        requires=[],
        outputs=["FV","FA","LV","LA","LS","RS"],
        enabled=True,
    ),
    "wing": FeatureSpec(
        func=feature_wingLR,
        requires=["ego_tracks"],
        outputs=["wingL","wingR"],
        enabled=True,
    ),
    "wing_minmax": FeatureSpec(
        func=feature_minmax_wing_angle,
        requires=["wing"],
        outputs=["minWingAng","maxWingAng","wingAmp"],
        enabled=True,
    ),
    "nearest_neighbor": FeatureSpec(
        func=feature_nearest_neighbor,
        requires=[],
        outputs=["minDist","nearestFlyIdx"],
        enabled=True,
    ),
    "nearest_geom": FeatureSpec(
        func=feature_nearest_geom,
        requires=["nearest_neighbor"],
        outputs=[],
        enabled=True,
    ),
    "ego_rel_nearest": FeatureSpec(
        func=feature_ego_rel_nearest,
        requires=["nearest_neighbor"],
        outputs=[],
        enabled=True,
    ),
    "fv_to_nearest": FeatureSpec(
        func=feature_FV_to_nearest,  # uses thx_v + nearest_dir_unit
        requires=["nearest_geom"],
        outputs=["FV_to_nearest"],
        enabled=True,
    ),
    "relfv_to_nearest": FeatureSpec(
        func=feature_relFV_to_nearest,  # uses thx_v + nearestFlyIdx + nearest_valid
        requires=["nearest_geom"],
        outputs=["relFV_to_nearest"],
        enabled=True,
    ),
    "ls_to_nearest": FeatureSpec(
        func=feature_LS_to_nearest,
        requires=["nearest_geom"],
        outputs=["LS_to_nearest"],
        enabled=True,
    ),
    "relLS_to_nearest": FeatureSpec(
        func=feature_relLS_to_nearest,
        requires=["nearest_geom"],
        outputs=["relLS_to_nearest"],
        enabled=True,
    ),
    "wing_arc_to_nearest": FeatureSpec(
        func=feature_wing_arc_to_nearest,
        requires=["nearest_neighbor"],
        outputs=["arcThetaL_to_nearest", "arcThetaR_to_nearest"],
        enabled=True,
    ),
    "ang_to_nearest": FeatureSpec(
        func=feature_ang_to_nearest,
        requires=["nearest_geom"],
        outputs=["ang_to_nearest"],
        enabled=True,
    ),
}

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