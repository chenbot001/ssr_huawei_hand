#!/usr/bin/env python3
"""Plot the distribution of initial arm EEF poses across all episodes in a zarr dataset."""

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


def auto_bins(data: np.ndarray) -> int:
    """Choose a reasonable number of bins using Freedman-Diaconis rule."""
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    if iqr == 0:
        return max(10, int(np.sqrt(len(data))))
    width = 2.0 * iqr / (len(data) ** (1.0 / 3.0))
    n = int((data.max() - data.min()) / width)
    return max(5, min(n, 80))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot histogram of initial arm EEF pose coordinates across episodes."
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default="data/pickplace_fruit/ryhand_dp_dataset.zarr",
        help="Path to the zarr dataset",
    )
    args = parser.parse_args()

    poses = collect_start_poses(args.zarr_path)
    n_episodes = poses.shape[0]
    print(f"Loaded {n_episodes} start poses from '{args.zarr_path}'")

    labels = ["x", "y", "z", "rx", "ry", "rz"]
    units = ["m", "m", "m", "rad", "rad", "rad"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"Initial Arm EEF Pose Distribution  ({n_episodes} episodes)",
        fontsize=14, fontweight="bold",
    )

    for i, (ax, label, unit) in enumerate(zip(axes.flat, labels, units)):
        col = poses[:, i]
        bins = auto_bins(col)
        ax.hist(col, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel(f"{label}  [{unit}]", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"$\\mathbf{{{label}}}$  —  mean={col.mean():.4f}, std={col.std():.4f}",
                     fontsize=10)
        ax.axvline(col.mean(), color="crimson", ls="--", lw=1.5, label=f"mean={col.mean():.4f}")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
