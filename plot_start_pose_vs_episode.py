#!/usr/bin/env python3
"""Plot each arm EEF coordinate against episode index."""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

import zarr


def collect_start_poses(zarr_path: str) -> np.ndarray:
    """Return (E, 6) array of arm EEF poses from the first frame of every episode."""
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        print(f"Error: cannot open '{zarr_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if "meta/episode_ends" not in root:
        print("Error: dataset missing 'meta/episode_ends'", file=sys.stderr)
        sys.exit(1)
    if "data/arm_eef_pose" not in root:
        print("Error: dataset missing 'data/arm_eef_pose'", file=sys.stderr)
        sys.exit(1)

    ends = root["meta/episode_ends"][:]
    start_indices = [0] + [int(i) for i in ends[:-1]]
    arm_data = root["data/arm_eef_pose"]
    return np.array([arm_data[i] for i in start_indices], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot arm EEF coordinates against episode index."
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default="data/opticalfiber/ryhand_opticalfiber_dataset.zarr",
        help="Path to the zarr dataset",
    )
    args = parser.parse_args()

    poses = collect_start_poses(args.zarr_path)
    n_episodes = poses.shape[0]
    episodes = np.arange(n_episodes)
    print(f"Loaded {n_episodes} start poses from '{args.zarr_path}'")

    labels = ["x", "y", "z", "rx", "ry", "rz"]
    units = ["m", "m", "m", "rad", "rad", "rad"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Arm EEF Coordinate vs Episode  ({n_episodes} episodes)",
        fontsize=14, fontweight="bold",
    )

    for i, (ax, label, unit) in enumerate(zip(axes.flat, labels, units)):
        col = poses[:, i]
        ax.scatter(episodes, col, s=12, c="steelblue", alpha=0.7, edgecolors="none")
        ax.axhline(col.mean(), color="crimson", ls="--", lw=1.5,
                   label=f"mean={col.mean():+.4f}")
        ax.axhline(np.median(col), color="darkgreen", ls=":", lw=1.5,
                   label=f"median={np.median(col):+.4f}")
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(f"{label}  [{unit}]", fontsize=11)
        ax.set_title(f"${label}$", fontsize=11)
        ax.legend(fontsize=7)
        ax.set_xlim(-0.5, n_episodes - 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
