#!/usr/bin/env python3
"""
Filter out short episodic clips from the first N episodes of a zarr dataset.

Removes all episodes with fewer than ``min_steps`` steps from the first
``check_first`` episodes, while preserving every episode beyond that range.
The result is written to a new zarr file (original path with ``_filtered``
appended before ``.zarr``), leaving the original untouched.

Usage::

    conda activate ssr_huawei
    python scripts/filter_short_episodes.py
    python scripts/filter_short_episodes.py --min-steps 350
    python scripts/filter_short_episodes.py --zarr-path /path/to/dataset.zarr
"""

from __future__ import annotations

import argparse
import os
import sys

import numcodecs
import numpy as np

import zarr


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter episodes shorter than a threshold from the first N episodes "
            "of a zarr dataset.  Creates a new zarr file, the original is never "
            "modified."
        ),
    )
    parser.add_argument(
        "--zarr-path",
        default="data/assembly/ryhand_assembly_dataset.zarr",
        help="Path to the input zarr dataset (default: data/assembly/ryhand_assembly_dataset_filtered.zarr)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=350,
        help="Minimum episode length to keep (default: 350)",
    )
    parser.add_argument(
        "--check-first",
        type=int,
        default=120,
        help="Only filter episodes within the first N (default: 120)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output zarr path (default: <input_base>_filtered.zarr)",
    )
    return parser.parse_args()


def _build_episodes_plan(
    root: zarr.Group,
    min_steps: int,
    check_first: int,
) -> list[tuple[int, int]]:
    """Return list of ``(start, stop)`` slices for episodes to keep."""
    ends = root["meta/episode_ends"][:]
    starts = np.concatenate([[0], ends[:-1]])
    n_total = len(ends)

    plan: list[tuple[int, int]] = []
    n_dropped = 0
    for ep_idx in range(n_total):
        ep_start = int(starts[ep_idx])
        ep_stop = int(ends[ep_idx])
        ep_len = ep_stop - ep_start

        keep = True
        if ep_idx < check_first and ep_len < min_steps:
            keep = False

        if keep:
            plan.append((ep_start, ep_stop))
        else:
            n_dropped += 1
            print(
                f"  Dropping episode {ep_idx} ({ep_len} steps < {min_steps})"
            )

    print(f"\n  Episodes kept: {len(plan)}, dropped: {n_dropped}")
    return plan


def _create_output_zarr(
    output_path: str,
    total_steps: int,
    total_episodes: int,
    ref_root: zarr.Group,
) -> zarr.Group:
    """Allocate the output zarr file with the same schema as the source."""
    compressor = numcodecs.Blosc(
        cname="lz4", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE
    )

    out = zarr.open(
        zarr.DirectoryStore(output_path), mode="w", synchronizer=None
    )

    src_data = ref_root["data"]
    data_out = out.create_group("data")
    meta_out = out.create_group("meta")

    for key in src_data:
        src_arr = src_data[key]
        shape = (total_steps,) + src_arr.shape[1:]
        chunks = src_arr.chunks if src_arr.chunks else (256,) + src_arr.shape[1:]
        # Use source chunks for images (often (1, H, W, C)), numeric default otherwise
        data_out.zeros(
            key,
            shape=shape,
            dtype=src_arr.dtype,
            chunks=chunks,
            compressor=src_arr.compressor or compressor,
        )

    meta_out.zeros(
        "episode_ends",
        shape=(total_episodes,),
        dtype=np.int64,
        compressor=None,
    )

    return out


def main() -> None:
    args = _parse_args()

    # ----------------------------------------------------------------
    # 1. Open source
    # ----------------------------------------------------------------
    src_path = args.zarr_path
    if not os.path.isdir(src_path):
        print(f"Error: zarr path '{src_path}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        root = zarr.open(src_path, mode="r")
    except Exception as exc:
        print(f"Error: cannot open zarr dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    if "meta/episode_ends" not in root or "data" not in root:
        print(
            "Error: dataset missing 'meta/episode_ends' or 'data' group.",
            file=sys.stderr,
        )
        sys.exit(1)

    ends = root["meta/episode_ends"][:]
    print(f"Source: {src_path}")
    print(f"  Total episodes: {len(ends)}, total steps: {int(ends[-1])}")

    # ----------------------------------------------------------------
    # 2. Decide which episodes to keep
    # ----------------------------------------------------------------
    plan = _build_episodes_plan(root, args.min_steps, args.check_first)
    if len(plan) == len(ends):
        print("\nNo episodes to drop — nothing to do.")
        sys.exit(0)

    total_steps = sum(stop - start for start, stop in plan)
    total_episodes = len(plan)
    print(f"  Total steps after filtering: {total_steps}")

    # ----------------------------------------------------------------
    # 3. Create output zarr
    # ----------------------------------------------------------------
    if args.output_path is not None:
        output_path = args.output_path
    else:
        base, ext = os.path.splitext(src_path)
        output_path = f"{base}_filtered.zarr" if ext == ".zarr" else src_path + "_filtered.zarr"

    print(f"\nCreating filtered dataset: {output_path}")
    out = _create_output_zarr(output_path, total_steps, total_episodes, root)
    data_out = out["data"]

    # ----------------------------------------------------------------
    # 4. Copy data
    # ----------------------------------------------------------------
    write_offset = 0
    episode_ends_list: list[int] = []

    for ep_i, (ep_start, ep_stop) in enumerate(plan):
        ep_len = ep_stop - ep_start

        for key in root["data"]:
            data_out[key][write_offset : write_offset + ep_len] = root["data"][key][ep_start:ep_stop]

        write_offset += ep_len
        episode_ends_list.append(write_offset)

        if (ep_i + 1) % 50 == 0 or ep_i == total_episodes - 1:
            pct = 100.0 * (ep_i + 1) / total_episodes
            print(
                f"  [{ep_i + 1}/{total_episodes}] {pct:.0f}% — "
                f"{write_offset}/{total_steps} steps"
            )

    out["meta"]["episode_ends"][:] = np.array(
        episode_ends_list, dtype=np.int64
    )

    print(f"\nDone. Filtered dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
