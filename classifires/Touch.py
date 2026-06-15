import numpy as np
from scipy.ndimage import binary_closing, binary_opening

def detect_touching_from_features(features, touch_dist_threshold=15.0, max_speed_threshold=5.0, min_touch_frames=4, max_jitter_frames=2):
    """
    Detects intentional touching/interaction between flies.
    
    Parameters:
    - features: Dictionary containing "minDist", "FV", and "LV".
    - touch_dist_threshold: Maximum distance (in pixels) to be considered physically touching.
    - max_speed_threshold: Maximum speed (px/frame) allowed to exclude high-speed collisions.
    - min_touch_frames: Minimum consecutive frames of contact to count as an intentional touch.
    - max_jitter_frames: Gap bridging for tracking noise during a touch.
    """
    minDist = features["minDist"]
    FV = features["FV"]
    LV = features["LV"]
    
    # Calculate the moving speed of the fly
    total_speeds = np.hypot(FV, LV)
    
    # Condition 1: Flies are physically close to each other
    is_close = minDist < touch_dist_threshold
    
    # Condition 2: The fly is not moving at breakneck speed (filtering out crashes)
    is_controlled = total_speeds < max_speed_threshold
    
    # Raw touch classification
    raw_touch = is_close & is_controlled
    
    # Morphological temporal filtering (Smoothing)
    n_frames, n_flies = raw_touch.shape
    smoothed_touch = np.zeros_like(raw_touch, dtype=bool)
    
    jitter_structure = np.ones(max_jitter_frames, dtype=bool)
    duration_structure = np.ones(min_touch_frames, dtype=bool)
    
    for fly in range(n_flies):
        fly_data = raw_touch[:, fly]
        
        # Bridge tiny gaps where SLEAP might have briefly lost the node
        fly_data = binary_closing(fly_data, structure=jitter_structure)
        
        # Erase super short bumps/collisions
        fly_data = binary_opening(fly_data, structure=duration_structure)
        
        smoothed_touch[:, fly] = fly_data
        
    return smoothed_touch