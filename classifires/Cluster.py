import numpy as np
from scipy.ndimage import binary_closing, binary_opening

def detect_clustering_from_features(features, cluster_dist_threshold=40.0, min_cluster_frames=60, max_jitter_frames=10):
    """
    Detects Social Clustering (passive proximity / hanging out together).
    
    Parameters:
    - features: Dictionary containing "minDist" (or any other distance feature).
    - cluster_dist_threshold: Maximum distance (in pixels) to be considered in the same "cluster" (e.g., 2 body lengths).
    - min_cluster_frames: Minimum consecutive frames they must stay near each other (e.g., 60 frames = 2 seconds).
    - max_jitter_frames: Gap bridging if they briefly step away and come right back.
    """
    # Using the minimum distance between the flies
    minDist = features["minDist"]
    
    # Condition 1: Flies are within the "social space" radius
    raw_cluster = minDist < cluster_dist_threshold
    
    # Morphological temporal filtering (Smoothing)
    n_frames, n_flies = raw_cluster.shape
    smoothed_cluster = np.zeros_like(raw_cluster, dtype=bool)
    
    jitter_structure = np.ones(max_jitter_frames, dtype=bool)
    duration_structure = np.ones(min_cluster_frames, dtype=bool)
    
    for fly in range(n_flies):
        fly_data = raw_cluster[:, fly]
        
        # Step A: Bridge small gaps (e.g., one fly takes a quick step out of the radius and returns)
        fly_data = binary_closing(fly_data, structure=jitter_structure)
        
        # Step B: Erase "pass-bys" (If a fly just walks past another fly quickly, it's not clustering)
        # They MUST stay together for at least 'min_cluster_frames' to count.
        fly_data = binary_opening(fly_data, structure=duration_structure)
        
        smoothed_cluster[:, fly] = fly_data
        
    return smoothed_cluster