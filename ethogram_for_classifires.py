import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load the classification results we just saved
behaviors_path = r"D:\DuoTrax\Rotem_IsolatedGrouped_2025-07-08_10-47-52\behaviors.h5"

with h5py.File(behaviors_path, 'r') as f:
    walking = f['walking'][:]
    stopping = f['stopping'][:]

# Display settings: focus on fly 0, and the first 1000 frames (to avoid overcrowding the plot)
fly_id = 0
start_frame = 0
end_frame = 1000

# Slice the required data based on our settings
walk_data = walking[start_frame:end_frame, fly_id]
stop_data = stopping[start_frame:end_frame, fly_id]
frames = np.arange(start_frame, end_frame)

# --- Plotting the Ethogram ---
plt.figure(figsize=(15, 4))

# Draw blue blocks where the fly is walking
plt.fill_between(frames, 0, 1, where=walk_data, color='blue', label='Walking', alpha=0.7)

# Draw red blocks where the fly is stopping 
# (Plotted slightly higher on the Y-axis so behaviors don't overlap visually)
plt.fill_between(frames, 1.2, 2.2, where=stop_data, color='red', label='Stopping', alpha=0.7)

# Format the graph appearance
plt.yticks([0.5, 1.7], ['Walking', 'Stopping'])
plt.xlabel('Frame Number')
plt.title(f'Ethogram - Fly {fly_id} (Frames {start_frame} to {end_frame})')
plt.legend(loc='upper right')
plt.tight_layout()

# Display the final graph on the screen
plt.show()