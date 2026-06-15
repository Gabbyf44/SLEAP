import numpy as np
from scipy.ndimage import binary_closing, binary_opening

def detect_stopping_from_features(features, max_speed_threshold=0.5, min_stop_frames=10, max_jitter_frames=5):
    """
    Detects true resting/stopping bouts for all flies.
    (Spinning in place is considered stopping, as long as translation is near zero).
    
    Parameters:
    - features: Dictionary containing "FV" and "LV" (Forward and Lateral Velocity).
    - max_speed_threshold: Maximum absolute speed (px/frame) to still be considered stopped.
    - min_stop_frames: Minimum consecutive frames the fly must be stationary to count as a 'Stop'.
    - max_jitter_frames: Small tracking jitters (movements) to ignore if surrounded by a long stop.
    """
    FV = features["FV"]
    LV = features["LV"]
    
    # 1. Calculate total translation speed
    total_speeds = np.hypot(FV, LV)
    
    # 2. Relaxed Stop Condition: Only translation speed matters
    raw_stop = total_speeds < max_speed_threshold
    
    # 3. Morphological temporal filtering (Smoothing)
    n_frames, n_flies = raw_stop.shape
    smoothed_stop = np.zeros_like(raw_stop, dtype=bool)
    
    # Structures for filtering
    jitter_structure = np.ones(max_jitter_frames, dtype=bool)
    duration_structure = np.ones(min_stop_frames, dtype=bool)
    
    for fly in range(n_flies):
        fly_data = raw_stop[:, fly]
        
        # Step A: Bridge tiny false movements (camera jitter) during an ongoing stop
        fly_data = binary_closing(fly_data, structure=jitter_structure)
        
        # Step B: Erase short pauses (must be stopped for AT LEAST min_stop_frames)
        fly_data = binary_opening(fly_data, structure=duration_structure)
        
        smoothed_stop[:, fly] = fly_data
        
    return smoothed_stop