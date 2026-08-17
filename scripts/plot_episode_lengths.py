#!/usr/bin/env python3
"""Plot episode lengths from a zarr dataset."""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

import zarr


def compute_episode_lengths(zarr_path: str) -> np.ndarray:
    """Read meta/episode_ends from a zarr file and return per-episode lengths."""
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        print(f"Error: Cannot open zarr dataset at '{zarr_path}': {e}", file=sys.stderr)
        sys.exit(1)

    if "meta/episode_ends" not in root:
        print(f"Error: '{zarr_path}' does not contain 'meta/episode_ends'.", file=sys.stderr)
        sys.exit(1)

    ends = root["meta/episode_ends"][:]
    lengths = np.diff(ends, prepend=0)
    return lengths


def plot_episode_lengths(lengths: np.ndarray) -> None:
    """Plot episode lengths as a bar chart."""
    episodes = np.arange(len(lengths))

    plt.figure(figsize=(12, 5))
    plt.bar(episodes, lengths, width=0.8, color="steelblue", edgecolor="none")

    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Lengths")
    plt.xlim(left=-0.5)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot episode lengths from a zarr dataset."
    )
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default="data/opticalfiber/ryhand_opticalfiber_dataset.zarr",
        help="Path to the zarr dataset (default: data/opticalfiber/ryhand_opticalfiber_dataset.zarr)",
    )
    args = parser.parse_args()

    lengths = compute_episode_lengths(args.zarr_path)
    print(f"Loaded {len(lengths)} episodes from '{args.zarr_path}'.")
    print(f"Steps per episode: mean={lengths.mean():.1f}, min={lengths.min()}, max={lengths.max()}")
    plot_episode_lengths(lengths)


if __name__ == "__main__":
    main()
