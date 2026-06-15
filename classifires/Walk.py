import numpy as np
from scipy.ndimage import binary_closing

def detect_walking_from_features(features, speed_threshold=1.4, min_gap_frames=10):
    """
    Detects walking bouts for all flies using pre-computed kinematic features.
    
    Parameters:
    - features: Dictionary containing "FV" (Forward Velocity) and "LV" (Lateral Velocity).
                Both expected to be of shape (n_frames, n_flies).
    - speed_threshold: Minimum absolute speed (px/frame) to be considered walking.
    - min_gap_frames: Maximum number of stationary frames to bridge/ignore during a walking bout.
    
    Returns:
    - smoothed_walking: Boolean array of shape (n_frames, n_flies), where True indicates walking.
    """
    # Extract velocities from the pre-computed features dictionary
    FV = features["FV"]
    LV = features["LV"]
    
    # Calculate total absolute speed using the Pythagorean theorem: sqrt(FV^2 + LV^2)
    # np.hypot is highly optimized and computes this directly for the entire matrix
    total_speeds = np.hypot(FV, LV)
    
    # Basic classification: Is the total speed strictly greater than the threshold?
    # This creates a boolean matrix (True/False) of shape (n_frames, n_flies)
    raw_walking = total_speeds > speed_threshold
    
    # Gap bridging (smoothing) - Applied separately for each individual fly
    n_frames, n_flies = raw_walking.shape
    smoothed_walking = np.zeros_like(raw_walking, dtype=bool)
    
    # Create the sliding "window" structure to close small gaps (False surrounded by True)
    structure = np.ones(min_gap_frames, dtype=bool)
    
    for fly in range(n_flies):
        # Apply morphological closing on the specific fly's column
        smoothed_walking[:, fly] = binary_closing(raw_walking[:, fly], structure=structure)
        
    return smoothed_walking