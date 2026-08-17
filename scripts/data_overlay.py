import zarr
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the dataset
zarr_path = 'data/opticalfiber/ryhand_opticalfiber_dataset.zarr'

# Open the Zarr root
root = zarr.open(zarr_path, mode='r')


# Load episodes
episode_ends = root['meta/episode_ends'][:120]

# The first frame of each episode is at index 0 and at (last_end + 1)
first_indices = [0] + [int(i) + 1 for i in episode_ends[:-1]]

def get_overlay(camera_data):
    frames = []
    num_frames = len(camera_data)
    for idx in first_indices:
        if 0 <= idx < num_frames:
            frame = camera_data[idx]
            frames.append(frame)
        else:
            print(f'Frame {idx} not found in camera_data (out of bounds)')

    if not frames:
        return None

    # Normalize and convert to float for blending
    frames = [f.astype(np.float32) / 255.0 if f.dtype == np.uint8 else f.astype(np.float32) for f in frames]

    # Onion skin overlay: equally distribute opacity among all frames
    n = len(frames)
    alpha = 1.0 / n
    overlay = np.zeros_like(frames[0])
    for frame in frames:
        overlay += frame * alpha

    # Clip and convert to uint8 for display
    overlay_img = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return overlay_img

camera_env = root['data/camera_env']
camera_wrist = root['data/camera_wrist']

overlay_env = get_overlay(camera_env)
overlay_wrist = get_overlay(camera_wrist)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

if overlay_env is not None:
    axes[0].imshow(overlay_env)
axes[0].axis('off')
axes[0].set_title('Onion Skin Overlay - Env Camera')

if overlay_wrist is not None:
    axes[1].imshow(overlay_wrist)
axes[1].axis('off')
axes[1].set_title('Onion Skin Overlay - Wrist Camera')

plt.tight_layout()
plt.show()
