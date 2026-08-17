import zarr
import numpy as np

zarr_path = 'data/pickplace_earbud/pick/ryhand_pick_earbud.zarr'
root = zarr.open(zarr_path, mode='r')

actions = root['data/action'][:]
print("Shape of actions:", actions.shape)

# Hand joints start at index 6.
# Middle: indices 6 + 2*3 = 12, 13, 14
# Ring: indices 6 + 3*3 = 15, 16, 17
# Pinky: indices 6 + 4*3 = 18, 19, 20
disabled_indices = list(range(12, 21))

disabled_actions = actions[:, disabled_indices]

max_vals = np.max(np.abs(disabled_actions), axis=0)
mean_vals = np.mean(np.abs(disabled_actions), axis=0)

print(f"Max absolute values for middle, ring, pinky joints (indices 12-20):")
print(max_vals)
print(f"Mean absolute values:")
print(mean_vals)

