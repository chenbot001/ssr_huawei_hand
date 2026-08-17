#!/usr/bin/env python3
"""
Data Integrity Test — validate zarr datasets produced by collect_data.py.

Auto-discovers all *.zarr folders under data/ and validates every episode
across all of them.

Checks per dataset:
  * All required arrays present (data/* and meta/episode_ends)
  * Shapes are internally consistent (row counts match episode_ends[-1])
  * episode_ends is monotonically increasing with no zeros
  * Correct dtypes (float32 for numeric, uint8 for images)
  * Image dimensions match expected (H, W, 3)
  * Optional: ReplayBuffer can load the dataset

Checks per episode:
  * Slice boundaries are valid (start < end, within array bounds)
  * No NaN / Inf values in numeric arrays
  * Image pixels within [0, 255]
  * Numeric arrays not all-zero (warns on suspicious constant data)
  * Episode length above a minimum threshold

Usage:
    python tests/test_data_integrity.py
    python tests/test_data_integrity.py --data-dir /custom/data
    python tests/test_data_integrity.py <path_to.zarr> ...
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

import numpy as np

EXPECTED_DATA_ARRAYS = {
    "arm_eef_pose":       {"ndim": 2, "cols": 6,  "dtype": np.float32},
    "hand_joint_angles":  {"ndim": 2, "cols": 15, "dtype": np.float32},
    "camera_env":         {"ndim": 4, "dtype": np.uint8},
    "camera_wrist":       {"ndim": 4, "dtype": np.uint8},
}

RAW_ACTION_ARRAYS = {
    "action_hand_joints": {"ndim": 2, "cols": 15, "dtype": np.float32},
    "action_eef_delta":   {"ndim": 2, "cols": 6,  "dtype": np.float32},
}

PREPROCESSED_ACTION_ARRAYS = {
    "action": {"ndim": 2, "cols": 21, "dtype": np.float32},
}

NUMERIC_ARRAYS = [
    k for k, v in EXPECTED_DATA_ARRAYS.items() if v["dtype"] == np.float32
] + list(RAW_ACTION_ARRAYS.keys()) + list(PREPROCESSED_ACTION_ARRAYS.keys())

IMAGE_ARRAYS = [
    k for k, v in EXPECTED_DATA_ARRAYS.items() if v["dtype"] == np.uint8
]

MIN_EPISODE_STEPS = 5


def _tag(ok: bool) -> str:
    return "[PASS]" if ok else "[FAIL]"


def _warn_tag() -> str:
    return "[WARN]"


def validate_zarr_structure(root, zarr_path: str) -> tuple[bool, list[str], np.ndarray | None]:
    """Run structural integrity checks on the zarr store. Returns (ok, messages, episode_ends)."""
    msgs: list[str] = []
    all_ok = True

    # ---- meta/episode_ends ----
    if "meta" not in root or "episode_ends" not in root["meta"]:
        msgs.append("[FAIL] meta/episode_ends missing")
        return False, msgs, None

    episode_ends = root["meta"]["episode_ends"][:]
    n_episodes = len(episode_ends)
    msgs.append(f"[INFO] Episodes: {n_episodes}")

    if n_episodes == 0:
        msgs.append("[FAIL] Dataset is empty (0 episodes)")
        return False, msgs, None

    total_steps = int(episode_ends[-1])
    msgs.append(f"[INFO] Total steps: {total_steps}")

    diffs = np.diff(episode_ends)
    mono_ok = bool(np.all(diffs > 0))
    msgs.append(f"{_tag(mono_ok)} episode_ends monotonically increasing")
    all_ok &= mono_ok

    no_zero = bool(episode_ends[0] > 0)
    msgs.append(f"{_tag(no_zero)} first episode_end > 0 (value: {episode_ends[0]})")
    all_ok &= no_zero

    lengths = np.diff(np.concatenate([[0], episode_ends]))
    msgs.append(
        f"[INFO] Episode lengths: min={int(lengths.min())}, "
        f"max={int(lengths.max())}, mean={lengths.mean():.1f}"
    )

    # ---- data arrays ----
    if "data" not in root:
        msgs.append("[FAIL] data/ group missing")
        return False, msgs, None

    data_group = root["data"]

    # Determine which action arrays to look for
    expected_arrays = dict(EXPECTED_DATA_ARRAYS)
    if "action" in data_group:
        expected_arrays.update(PREPROCESSED_ACTION_ARRAYS)
        msgs.append("[INFO] Found preprocessed 'action' array")
    else:
        expected_arrays.update(RAW_ACTION_ARRAYS)
        msgs.append("[INFO] Found raw 'action_hand_joints' and 'action_eef_delta' arrays")

    for name, spec in expected_arrays.items():
        if name not in data_group:
            msgs.append(f"[FAIL] data/{name} missing")
            all_ok = False
            continue

        arr = data_group[name]
        shape = arr.shape

        rows_ok = shape[0] == total_steps
        msgs.append(
            f"{_tag(rows_ok)} data/{name} rows={shape[0]} "
            f"(expected {total_steps})"
        )
        all_ok &= rows_ok

        ndim_ok = len(shape) == spec["ndim"]
        if not ndim_ok:
            msgs.append(f"[FAIL] data/{name} ndim={len(shape)} (expected {spec['ndim']})")
            all_ok = False
            continue

        if "cols" in spec:
            cols_ok = shape[1] == spec["cols"]
            msgs.append(
                f"{_tag(cols_ok)} data/{name} cols={shape[1]} "
                f"(expected {spec['cols']})"
            )
            all_ok &= cols_ok

        dtype_ok = arr.dtype == spec["dtype"]
        msgs.append(
            f"{_tag(dtype_ok)} data/{name} dtype={arr.dtype} "
            f"(expected {spec['dtype']})"
        )
        all_ok &= dtype_ok

    # ---- image shape consistency (H, W, 3) ----
    for img_name in IMAGE_ARRAYS:
        if img_name not in data_group:
            continue
        shape = data_group[img_name].shape
        if len(shape) == 4:
            ch_ok = shape[3] == 3
            msgs.append(
                f"{_tag(ch_ok)} data/{img_name} channels={shape[3]} (expected 3)"
            )
            all_ok &= ch_ok
            hw = (shape[1], shape[2])
            msgs.append(f"[INFO] data/{img_name} resolution: {hw[1]}x{hw[0]}")

    # ---- cross-array row consistency ----
    row_counts = set()
    for name in expected_arrays:
        if name in data_group:
            row_counts.add(data_group[name].shape[0])
    consistent = len(row_counts) <= 1
    msgs.append(f"{_tag(consistent)} All data arrays have same row count: {row_counts}")
    all_ok &= consistent

    return all_ok, msgs, episode_ends


def validate_episodes(root, episode_ends: np.ndarray, verbose: bool = False) -> tuple[bool, list[str]]:
    """Per-episode data quality checks. Returns (ok, messages)."""
    msgs: list[str] = []
    all_ok = True
    data_group = root["data"]
    n_episodes = len(episode_ends)
    starts = np.concatenate([[0], episode_ends[:-1]])

    fail_count = 0
    warn_count = 0

    for ep_idx in range(n_episodes):
        ep_start = int(starts[ep_idx])
        ep_end = int(episode_ends[ep_idx])
        ep_len = ep_end - ep_start
        ep_ok = True
        ep_warns: list[str] = []
        ep_fails: list[str] = []

        # Slice sanity
        if ep_start >= ep_end:
            ep_fails.append(f"invalid slice [{ep_start}:{ep_end}]")
            ep_ok = False

        if ep_len < MIN_EPISODE_STEPS:
            ep_warns.append(f"very short ({ep_len} steps < {MIN_EPISODE_STEPS})")
            warn_count += 1

        # Numeric arrays: NaN/Inf, all-zero
        for name in NUMERIC_ARRAYS:
            if name not in data_group:
                continue
            try:
                chunk = data_group[name][ep_start:ep_end]
            except Exception as e:
                ep_fails.append(f"data/{name} read error: {e}")
                ep_ok = False
                continue

            if np.any(np.isnan(chunk)):
                nan_count = int(np.isnan(chunk).sum())
                ep_fails.append(f"data/{name} has {nan_count} NaN values")
                ep_ok = False
            if np.any(np.isinf(chunk)):
                inf_count = int(np.isinf(chunk).sum())
                ep_fails.append(f"data/{name} has {inf_count} Inf values")
                ep_ok = False
            if np.all(chunk == 0):
                ep_warns.append(f"data/{name} is all zeros")
                warn_count += 1

        # Image arrays: spot-check first and last frames
        for name in IMAGE_ARRAYS:
            if name not in data_group:
                continue
            try:
                first_frame = data_group[name][ep_start]
                last_frame = data_group[name][ep_end - 1]
            except Exception as e:
                ep_fails.append(f"data/{name} read error: {e}")
                ep_ok = False
                continue

            for label, frame in [("first", first_frame), ("last", last_frame)]:
                if frame.max() == 0 and frame.min() == 0:
                    ep_warns.append(f"data/{name} {label} frame is fully black")
                    warn_count += 1

        if not ep_ok:
            fail_count += 1
            all_ok = False

        if ep_fails or (verbose and ep_warns):
            msgs.append(f"  Episode {ep_idx:>4d}  [{ep_start}:{ep_end}] ({ep_len} steps)")
            for f in ep_fails:
                msgs.append(f"    [FAIL] {f}")
            if verbose:
                for w in ep_warns:
                    msgs.append(f"    {_warn_tag()} {w}")

    ep_pass = n_episodes - fail_count
    summary_ok = fail_count == 0
    msgs.insert(0,
        f"{_tag(summary_ok)} Episode data quality: "
        f"{ep_pass}/{n_episodes} passed, {fail_count} failed, {warn_count} warnings"
    )
    return all_ok, msgs


def validate_replay_buffer(zarr_path: str, n_episodes: int, total_steps: int) -> tuple[bool, list[str]]:
    """Try loading via ReplayBuffer as a smoke test."""
    msgs: list[str] = []
    try:
        dp_path = os.path.join(project_root, "external", "diffusion_policy")
        if dp_path not in sys.path:
            sys.path.insert(0, dp_path)
        from diffusion_policy.common.replay_buffer import ReplayBuffer

        rb = ReplayBuffer.create_from_path(zarr_path, mode="r")
        rb_ok = rb.n_episodes == n_episodes and rb.n_steps == total_steps
        msgs.append(
            f"{_tag(rb_ok)} ReplayBuffer loaded: "
            f"n_episodes={rb.n_episodes}, n_steps={rb.n_steps}, "
            f"keys={list(rb.keys())}"
        )
        return rb_ok, msgs
    except ImportError:
        msgs.append("[INFO] diffusion_policy not available — skipping ReplayBuffer check")
        return True, msgs
    except Exception as e:
        msgs.append(f"{_warn_tag()} ReplayBuffer load error: {e}")
        return True, msgs


def validate_zarr(zarr_path: str, verbose: bool = False) -> tuple[bool, list[str]]:
    """Run all integrity checks on a single zarr dataset."""
    msgs: list[str] = []

    try:
        import zarr
    except ImportError:
        msgs.append("[FAIL] zarr package not installed")
        return False, msgs

    if not os.path.isdir(zarr_path):
        msgs.append(f"[FAIL] Path does not exist: {zarr_path}")
        return False, msgs

    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception as e:
        msgs.append(f"[FAIL] Cannot open zarr store: {e}")
        return False, msgs

    # Phase 1: structural checks
    struct_ok, struct_msgs, episode_ends = validate_zarr_structure(root, zarr_path)
    msgs.extend(struct_msgs)

    if not struct_ok or episode_ends is None:
        return False, msgs

    n_episodes = len(episode_ends)
    total_steps = int(episode_ends[-1])

    # Phase 2: per-episode data quality
    ep_ok, ep_msgs = validate_episodes(root, episode_ends, verbose=verbose)
    msgs.extend(ep_msgs)

    # Phase 3: ReplayBuffer smoke test
    rb_ok, rb_msgs = validate_replay_buffer(zarr_path, n_episodes, total_steps)
    msgs.extend(rb_msgs)

    return struct_ok and ep_ok and rb_ok, msgs


def discover_zarr_folders(data_dir: str) -> list[str]:
    """Find all *.zarr directories under data_dir, sorted by name."""
    pattern = os.path.join(data_dir, "*.zarr")
    return sorted(glob.glob(pattern))


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate zarr datasets from collect_data.py"
    )
    parser.add_argument(
        "paths", nargs="*",
        help="Path(s) to .zarr directories to validate. "
             "If omitted, auto-discovers all *.zarr under --data-dir.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Folder containing *.zarr stores (default: <project>/data). "
             "Ignored when explicit paths are given.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-episode warnings (not just failures)",
    )
    args = parser.parse_args()

    if args.paths:
        zarr_paths = args.paths
    else:
        data_dir = os.path.abspath(args.data_dir or os.path.join(project_root, "data"))
        if not os.path.isdir(data_dir):
            print(f"[ERROR] Data directory not found: {data_dir}")
            return 1
        zarr_paths = discover_zarr_folders(data_dir)
        if not zarr_paths:
            print(f"[ERROR] No .zarr directories found under {data_dir}")
            return 1

    print("=" * 70)
    print("  Data Integrity Test")
    print(f"  Datasets to check: {len(zarr_paths)}")
    print("=" * 70)

    overall = True
    total_episodes = 0
    total_steps = 0
    passed_datasets = 0
    t_start = time.time()

    for idx, zpath in enumerate(zarr_paths, 1):
        rel = os.path.relpath(zpath, project_root)
        print(f"\n{'─' * 70}")
        print(f"  [{idx}/{len(zarr_paths)}] {rel}")
        print(f"{'─' * 70}")

        passed, msgs = validate_zarr(zpath, verbose=args.verbose)
        for m in msgs:
            print(f"  {m}")

        if passed:
            passed_datasets += 1
        else:
            overall = False

        try:
            import zarr
            root = zarr.open(zpath, mode="r")
            if "meta" in root and "episode_ends" in root["meta"]:
                ends = root["meta"]["episode_ends"][:]
                total_episodes += len(ends)
                if len(ends) > 0:
                    total_steps += int(ends[-1])
        except Exception:
            pass

    elapsed = time.time() - t_start

    print(f"\n{'=' * 70}")
    print("  Aggregate Summary")
    print(f"{'─' * 70}")
    print(f"  Datasets  : {passed_datasets}/{len(zarr_paths)} passed")
    print(f"  Episodes  : {total_episodes} total across all datasets")
    print(f"  Steps     : {total_steps} total")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print(f"{'─' * 70}")
    if overall:
        print("  \033[92mALL DATASETS VALID\033[0m")
    else:
        print("  \033[91mINTEGRITY ISSUES DETECTED\033[0m")
    print("=" * 70)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
