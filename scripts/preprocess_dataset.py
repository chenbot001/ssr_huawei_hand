#!/usr/bin/env python3
"""
Preprocess collected zarr datasets for diffusion policy training.

Merges raw session ``collected*.zarr`` folders under data/ (by default) into a single zarr store
with the structure expected by diffusion_policy's ReplayBuffer:

  ssr_pickplace_dataset.zarr/
    data/
      arm_eef_pose        (N, 6)           float32   obs
      hand_joint_angles   (N, 15)          float32   obs
      camera_env          (N, 240, 320, 3) uint8     obs  (rgb)
      camera_wrist        (N, 240, 320, 3) uint8     obs  (rgb)
      action              (N, 21)          float32   action  [eef_delta(6) | hand_joints(15)]
    meta/
      episode_ends        (E,)             int64

Key transformations:
  1. Concatenates action_eef_delta (6D) and action_hand_joints (15D) into
     a single "action" array (21D).
  2. Drops the original action_eef_delta / action_hand_joints columns from
     the output (they are encoded inside "action").
  3. Re-computes episode_ends with cumulative offsets across all source zarrs.
  4. Validates the merged dataset with ReplayBuffer.

By default only **raw session** zarr folders are merged: basenames starting with
``collected`` (e.g. ``collected_20260324_*.zarr``). Merged outputs such as
``ssr_pickplace_dataset.zarr``, ``ryhand_dp_dataset.zarr``, and other non-session
stores are ignored. Use ``--all-zarr`` to include every ``*.zarr`` (advanced).

Usage:
    python scripts/preprocess_dataset.py
    python scripts/preprocess_dataset.py --data-dir data --output data/merged.zarr
    python scripts/preprocess_dataset.py --include collected_20260322_*.zarr
    python scripts/preprocess_dataset.py --min-episode-len 30
    python scripts/preprocess_dataset.py --all-zarr
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
dp_path = os.path.join(project_root, "external", "diffusion_policy")
for p in (src_path, project_root, dp_path):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(project_root)

import numcodecs
import numpy as np
import zarr


OBS_KEYS = ["arm_eef_pose", "hand_joint_angles", "camera_env", "camera_wrist"]
ACTION_SRC_KEYS = ["action_eef_delta", "action_hand_joints"]
ACTION_DST_KEY = "action"

IMG_KEYS = {"camera_env", "camera_wrist"}

# Raw teleop session exports (see collect_data.py); merged/training zarrs use other names.
DEFAULT_RAW_ZARR_PREFIX = "collected"


def discover_zarr_folders(
    data_dir: str,
    include: str | None = None,
    *,
    raw_only: bool = True,
    raw_prefix: str = DEFAULT_RAW_ZARR_PREFIX,
) -> list[str]:
    pattern = os.path.join(data_dir, "*.zarr")
    paths = sorted(glob.glob(pattern))
    if raw_only:
        paths = [
            p for p in paths
            if os.path.basename(p).startswith(raw_prefix)
        ]
    if include:
        paths = [p for p in paths if fnmatch.fnmatch(os.path.basename(p), include)]
    return paths


def inspect_source(zarr_path: str) -> dict | None:
    """Quick sanity check on a source zarr. Returns info dict or None on failure."""
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        print(f"  [SKIP] Cannot open {zarr_path}: {e}")
        return None

    if "data" not in root or "meta" not in root:
        print(f"  [SKIP] Missing data/ or meta/ in {zarr_path}")
        return None

    episode_ends = root["meta"]["episode_ends"][:]
    if len(episode_ends) == 0:
        print(f"  [SKIP] Empty dataset: {zarr_path}")
        return None

    for key in OBS_KEYS + ACTION_SRC_KEYS:
        if key not in root["data"]:
            print(f"  [SKIP] Missing data/{key} in {zarr_path}")
            return None

    n_steps = int(episode_ends[-1])
    return {
        "path": zarr_path,
        "root": root,
        "episode_ends": episode_ends,
        "n_episodes": len(episode_ends),
        "n_steps": n_steps,
    }


def create_output_zarr(
    output_path: str,
    total_steps: int,
    total_episodes: int,
    ref_root,
    compressor_numeric,
    compressor_img,
    chunk_time: int,
) -> zarr.Group:
    """Create the output zarr store with pre-allocated arrays."""
    store = zarr.DirectoryStore(output_path)
    out = zarr.group(store, overwrite=True)
    data_group = out.create_group("data")
    meta_group = out.create_group("meta")

    for key in OBS_KEYS:
        src = ref_root["data"][key]
        shape = (total_steps,) + src.shape[1:]
        is_img = key in IMG_KEYS
        cpr = compressor_img if is_img else compressor_numeric
        cks_time = 1 if is_img else chunk_time
        chunks = (cks_time,) + src.shape[1:]
        data_group.zeros(key, shape=shape, dtype=src.dtype,
                         chunks=chunks, compressor=cpr)

    action_dim = sum(ref_root["data"][k].shape[1] for k in ACTION_SRC_KEYS)
    data_group.zeros(ACTION_DST_KEY,
                     shape=(total_steps, action_dim),
                     dtype=np.float32,
                     chunks=(chunk_time, action_dim),
                     compressor=compressor_numeric)

    meta_group.zeros("episode_ends",
                     shape=(total_episodes,),
                     dtype=np.int64,
                     compressor=None)

    return out


def merge_datasets(
    sources: list[dict],
    output_path: str,
    min_episode_len: int = 1,
    chunk_time: int = 256,
) -> str:
    """Merge multiple source zarrs into one output zarr."""

    ref_root = sources[0]["root"]
    compressor_numeric = numcodecs.Blosc(cname="lz4", clevel=5,
                                         shuffle=numcodecs.Blosc.NOSHUFFLE)
    compressor_img = numcodecs.Blosc(cname="lz4", clevel=5,
                                     shuffle=numcodecs.Blosc.NOSHUFFLE)

    # First pass: count totals and filter short episodes
    episodes_plan: list[tuple[zarr.Group, int, int]] = []
    for src in sources:
        root = src["root"]
        ends = src["episode_ends"]
        starts = np.concatenate([[0], ends[:-1]])
        for ep_idx in range(len(ends)):
            ep_start, ep_end = int(starts[ep_idx]), int(ends[ep_idx])
            ep_len = ep_end - ep_start
            if ep_len >= min_episode_len:
                episodes_plan.append((root, ep_start, ep_end))

    total_steps = sum(end - start for _, start, end in episodes_plan)
    total_episodes = len(episodes_plan)

    print(f"\n  Episodes after filtering (min_len={min_episode_len}): {total_episodes}")
    print(f"  Total steps: {total_steps}")

    if total_episodes == 0:
        print("  [ERROR] No episodes to merge.")
        sys.exit(1)

    out = create_output_zarr(
        output_path, total_steps, total_episodes,
        ref_root, compressor_numeric, compressor_img, chunk_time,
    )
    data_out = out["data"]
    meta_out = out["meta"]

    write_offset = 0
    episode_ends_list: list[int] = []

    for ep_i, (root, ep_start, ep_end) in enumerate(episodes_plan):
        ep_len = ep_end - ep_start
        src_data = root["data"]

        for key in OBS_KEYS:
            data_out[key][write_offset : write_offset + ep_len] = \
                src_data[key][ep_start:ep_end]

        eef_delta = src_data["action_eef_delta"][ep_start:ep_end]
        hand_joints = src_data["action_hand_joints"][ep_start:ep_end]
        data_out[ACTION_DST_KEY][write_offset : write_offset + ep_len] = \
            np.concatenate([eef_delta, hand_joints], axis=1)

        write_offset += ep_len
        episode_ends_list.append(write_offset)

        if (ep_i + 1) % 50 == 0 or ep_i == total_episodes - 1:
            pct = 100.0 * (ep_i + 1) / total_episodes
            print(f"  [{ep_i + 1}/{total_episodes}] {pct:.0f}% — "
                  f"{write_offset}/{total_steps} steps written")

    meta_out["episode_ends"][:] = np.array(episode_ends_list, dtype=np.int64)

    return output_path


def validate_output(output_path: str) -> bool:
    """Validate merged zarr with ReplayBuffer and print summary."""
    print(f"\n  Validating {output_path} ...")

    root = zarr.open(output_path, mode="r")
    print("  Arrays in output:")
    for key in root["data"]:
        a = root["data"][key]
        print(f"    data/{key}: {a.shape} {a.dtype}")
    ee = root["meta"]["episode_ends"][:]
    print(f"    meta/episode_ends: ({len(ee)},) int64")

    try:
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        rb = ReplayBuffer.create_from_path(output_path, mode="r")
        lengths = rb.episode_lengths
        print(f"\n  ReplayBuffer OK")
        print(f"    n_episodes : {rb.n_episodes}")
        print(f"    n_steps    : {rb.n_steps}")
        print(f"    keys       : {list(rb.keys())}")
        print(f"    ep lengths : min={lengths.min()}, max={lengths.max()}, "
              f"mean={lengths.mean():.1f}")

        action_arr = rb[ACTION_DST_KEY]
        print(f"    action dim : {action_arr.shape[1]} "
              f"(eef_delta=6 + hand_joints=15)")

        sample = action_arr[:min(1000, action_arr.shape[0])]
        has_nan = np.any(np.isnan(sample))
        has_inf = np.any(np.isinf(sample))
        print(f"    action NaN : {has_nan}, Inf: {has_inf}")

        return True
    except ImportError:
        print("  [WARN] diffusion_policy not importable — skipping ReplayBuffer check")
        return True
    except Exception as e:
        print(f"  [FAIL] ReplayBuffer validation error: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge collected zarr datasets for diffusion policy training"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory containing *.zarr folders (default: <project>/data)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output zarr path (default: <data-dir>/merged.zarr)",
    )
    parser.add_argument(
        "--include", type=str, default=None,
        help="Glob pattern to filter source zarr folders, e.g. 'collected_20260322_*.zarr'",
    )
    parser.add_argument(
        "--all-zarr",
        action="store_true",
        help="Merge all *.zarr under --data-dir (not only raw sessions starting with "
        f"'{DEFAULT_RAW_ZARR_PREFIX}'). Default excludes merged datasets like "
        "ssr_pickplace_dataset.zarr / ryhand_dp_dataset.zarr.",
    )
    parser.add_argument(
        "--raw-prefix",
        type=str,
        default=DEFAULT_RAW_ZARR_PREFIX,
        help=f"Basename prefix for raw session zarrs when --all-zarr is not set (default: {DEFAULT_RAW_ZARR_PREFIX})",
    )
    parser.add_argument(
        "--min-episode-len", type=int, default=1,
        help="Drop episodes shorter than this many steps (default: 1)",
    )
    parser.add_argument(
        "--chunk-time", type=int, default=256,
        help="Chunk size along time axis for numeric arrays (default: 256)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip ReplayBuffer validation after merge",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir or os.path.join(project_root, "data"))
    output_path = args.output or os.path.join(data_dir, "ssr_pickplace_dataset.zarr")

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        return 1

    print("=" * 70)
    print("  Dataset Preprocessor — Merge & Concatenate Actions")
    print("=" * 70)
    print(f"  Source dir : {data_dir}")
    print(f"  Output     : {output_path}")
    if args.all_zarr:
        print("  Sources    : all *.zarr under data-dir (--all-zarr)")
    else:
        print(
            f"  Sources    : only raw sessions (basename starts with "
            f"'{args.raw_prefix}'; use --all-zarr to include merged/other zarrs)"
        )

    zarr_paths = discover_zarr_folders(
        data_dir,
        include=args.include,
        raw_only=not args.all_zarr,
        raw_prefix=args.raw_prefix,
    )
    # never include the output in the source list
    zarr_paths = [p for p in zarr_paths
                  if os.path.abspath(p) != os.path.abspath(output_path)]

    if not zarr_paths:
        if not args.all_zarr:
            print(
                f"[ERROR] No raw session zarr folders (basename starts with "
                f"'{args.raw_prefix}') in {data_dir}"
            )
            print("  Use --all-zarr to merge every *.zarr, or collect data with collect_data.py.")
        else:
            print(f"[ERROR] No .zarr folders found in {data_dir}")
        return 1

    print(f"  Found {len(zarr_paths)} source zarr folder(s)")

    sources: list[dict] = []
    for zp in zarr_paths:
        info = inspect_source(zp)
        if info is not None:
            sources.append(info)
            rel = os.path.relpath(zp, project_root)
            print(f"    + {rel}  ({info['n_episodes']} eps, {info['n_steps']} steps)")

    if not sources:
        print("[ERROR] No valid source datasets found.")
        return 1

    total_src_eps = sum(s["n_episodes"] for s in sources)
    total_src_steps = sum(s["n_steps"] for s in sources)
    print(f"\n  Total sources: {len(sources)} zarrs, "
          f"{total_src_eps} episodes, {total_src_steps} steps")

    t0 = time.time()
    merge_datasets(
        sources=sources,
        output_path=output_path,
        min_episode_len=args.min_episode_len,
        chunk_time=args.chunk_time,
    )
    elapsed = time.time() - t0
    print(f"\n  Merge completed in {elapsed:.1f}s")

    if not args.skip_validation:
        ok = validate_output(output_path)
        if not ok:
            return 1

    print(f"\n{'=' * 70}")
    print(f"  Done. Merged dataset: {output_path}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
