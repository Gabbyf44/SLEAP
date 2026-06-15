import os
import glob
import h5py
import numpy as np
# import scipy.io

# Import your walking detection function
from Walk import detect_walking_from_features
from Stop import detect_stopping_from_features
from Touch import detect_touching_from_features
from Cluster import detect_clustering_from_features

# Path to the main data directory containing all experiment folders
data_dir = r"C:\Users\sycoh\OneDrive\Documents\Bioinformatics\final project\project\SLEAP\data"

# Search specifically for the pre-computed features files across all subdirectories
feature_files = glob.glob(os.path.join(data_dir, "*", "*.features.h5"))

for file_path in feature_files:
    # Extract the specific folder path to save results in the correct location
    folder_path = os.path.dirname(file_path)
    
    # Open the features file in read mode
    with h5py.File(file_path, 'r') as f:
        # Extract the matrices directly from the file based on the keys
        # The [:] syntax converts the HDF5 dataset into a usable NumPy array
        FV_matrix = f['FV'][:]
        LV_matrix = f['LV'][:]
        minDist_matrix = f['minDist'][:]

    # Build the dictionary structure expected by the Walk function
    features_dict = {
        "FV": FV_matrix,
        "LV": LV_matrix,
        "minDist": minDist_matrix
    }
    
    # Run the walking classification using the optimal threshold
    walking_matrix = detect_walking_from_features(features_dict, speed_threshold=1.4)
    stopping_matrix = detect_stopping_from_features(features_dict, max_speed_threshold=0.5)
    touching_matrix = detect_touching_from_features(features_dict, touch_dist_threshold=15.0) # may change it to lower depends on the experiments
    clustering_matrix = detect_clustering_from_features(features_dict, cluster_dist_threshold=80.0, min_cluster_frames=60) # may check munDist later
    

    # --- CHANGED: Save as H5 file instead of MATLAB ---
    # We define the path for the new behaviors.h5 file
    behaviors_path = os.path.join(folder_path, "behaviors.h5")

    # Open the behaviors file in 'w' mode (write mode) to create it
    with h5py.File(behaviors_path, 'w') as f:
        f.create_dataset('walking', data=walking_matrix)
        f.create_dataset('stopping', data=stopping_matrix)
        f.create_dataset('touching', data=touching_matrix)
        f.create_dataset('clustering', data=clustering_matrix)

    # Save the resulting boolean matrix as a MATLAB file in the same experiment folder
    # save_path = os.path.join(folder_path, "hard_coded_walking.mat")
    # scipy.io.savemat(save_path, {"walking_labels": walking_matrix})
    
    print(f"Processed features and saved behaviors.h5 for: {os.path.basename(folder_path)}")

print("Pipeline completed for all experiments!")