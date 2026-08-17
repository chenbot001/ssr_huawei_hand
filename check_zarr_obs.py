import zarr
import numpy as np

zarr_path = 'data/pickplace_earbud/pick/ryhand_pick_earbud.zarr'
root = zarr.open(zarr_path, mode='r')

hand_joints = root['data/hand_joint_angles'][:]
print("Shape of hand_joints:", hand_joints.shape)

disabled_indices = list(range(6, 15))

disabled_obs = hand_joints[:, disabled_indices]

max_vals = np.max(np.abs(disabled_obs), axis=0)
mean_vals = np.mean(np.abs(disabled_obs), axis=0)

print(f"Max absolute values for middle, ring, pinky joint observations:")
print(max_vals)
print(f"Mean absolute values:")
print(mean_vals)

