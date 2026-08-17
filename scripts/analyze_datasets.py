#!/usr/bin/env python3
"""
Scan zarr datasets under data/ and print a short summary of episodes, step counts,
episode-length statistics, and disk usage.

By default only **raw session** stores are listed (basename starts with ``collected``),
so merged outputs (e.g. ``ssr_pickplace_dataset.zarr``, ``ryhand_dp_dataset.zarr``)
are skipped. Use ``--all-zarr`` to summarize every ``*.zarr`` in the directory.

Usage:
  python scripts/analyze_datasets.py
  python scripts/analyze_datasets.py --data-dir /path/to/data
  python scripts/analyze_datasets.py --all-zarr
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (match replay_data.py)
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")
dp_path = os.path.join(project_root, "external", "diffusion_policy")

for p in (src_path, project_root, dp_path):
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

# Same convention as scripts/preprocess_dataset.py
DEFAULT_RAW_ZARR_PREFIX = "collected"


def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def _fmt_ts(path: str) -> str:
    try:
        m = os.path.getmtime(path)
        return datetime.fromtimestamp(m).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def analyze_one(zarr_path: str) -> dict | None:
    """Return a stats dict, or None if the store is missing / invalid."""
    if not os.path.isdir(zarr_path):
        return None
    try:
        rb = ReplayBuffer.create_from_path(zarr_path, mode="r")
    except Exception as e:
        return {"error": str(e), "path": zarr_path}

    keys = list(rb.keys())
    n_ep = rb.n_episodes
    n_steps = rb.n_steps
    lengths = rb.episode_lengths if n_ep else np.array([], dtype=np.int64)

    out: dict = {
        "path": zarr_path,
        "n_episodes": n_ep,
        "n_steps": n_steps,
        "keys": keys,
        "disk_bytes": _dir_size_bytes(zarr_path),
        "mtime": _fmt_ts(zarr_path),
    }

    if n_ep > 0:
        out["len_min"] = int(lengths.min())
        out["len_max"] = int(lengths.max())
        out["len_mean"] = float(lengths.mean())
        out["len_std"] = float(lengths.std())
        out["len_median"] = float(np.median(lengths))
    else:
        out["len_min"] = out["len_max"] = 0
        out["len_mean"] = out["len_std"] = out["len_median"] = 0.0

    if "camera_env" in rb:
        sh = rb["camera_env"].shape
        if len(sh) >= 3:
            out["rgb_shape"] = f"{sh[1]}x{sh[2]}x{sh[3]}" if len(sh) >= 4 else str(sh[1:])
    return out


def print_report(
    entries: list[dict],
    data_dir: str,
    raw_only: bool,
    raw_prefix: str,
) -> None:
    print("=" * 72)
    print("  Zarr dataset summary")
    print("=" * 72)
    print(f"  Directory : {data_dir}")
    if raw_only:
        print(f"  Filter    : raw sessions only (basename starts with '{raw_prefix}')")
    else:
        print("  Filter    : all *.zarr")
    print()

    ok = [e for e in entries if e and "error" not in e]
    bad = [e for e in entries if e and "error" in e]

    tot_ep = sum(e["n_episodes"] for e in ok)
    tot_steps = sum(e["n_steps"] for e in ok)
    tot_disk = sum(e["disk_bytes"] for e in ok)

    for e in ok:
        rel = os.path.relpath(e["path"], data_dir)
        print(f"  • {rel}")
        print(f"      modified   : {e['mtime']}")
        print(f"      disk       : {_fmt_bytes(e['disk_bytes'])}")
        print(f"      episodes   : {e['n_episodes']}")
        print(f"      steps      : {e['n_steps']}")
        if e["n_episodes"] > 0:
            print(
                f"      ep. steps  : mean={e['len_mean']:.1f}, "
                f"median={e['len_median']:.1f}, "
                f"min={e['len_min']}, max={e['len_max']}, "
                f"std={e['len_std']:.1f}"
            )
        print(f"      data keys  : {', '.join(e['keys'])}")
        if "rgb_shape" in e:
            print(f"      RGB (env)  : HxWxC = {e['rgb_shape']}")
        print()

    for e in bad:
        rel = os.path.relpath(e["path"], data_dir)
        print(f"  ✗ {rel}")
        print(f"      ERROR: {e['error']}\n")

    print("-" * 72)
    print("  Combined (readable stores only)")
    print(f"    Datasets        : {len(ok)}")
    print(f"    Total episodes  : {tot_ep}")
    print(f"    Total steps     : {tot_steps}")
    if tot_ep > 0:
        wmean = tot_steps / tot_ep
        print(f"    Avg steps/ep    : {wmean:.1f}  (weighted by episode count across files)")
    print(f"    Total disk      : {_fmt_bytes(tot_disk)}")
    if bad:
        print(f"    Failed / skipped: {len(bad)}")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize zarr datasets under data/ (raw collected sessions by default)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Folder containing *.zarr stores (default: <project>/data)",
    )
    parser.add_argument(
        "--all-zarr",
        action="store_true",
        help="Include every *.zarr (merged datasets, etc.). Default: only collected*.zarr",
    )
    parser.add_argument(
        "--raw-prefix",
        type=str,
        default=DEFAULT_RAW_ZARR_PREFIX,
        help=f"Basename prefix for raw session zarrs when --all-zarr is not set (default: {DEFAULT_RAW_ZARR_PREFIX})",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir or os.path.join(project_root, "data"))
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Not a directory: {data_dir}")
        sys.exit(1)

    pattern = os.path.join(data_dir, "*.zarr")
    paths = sorted(glob.glob(pattern))
    if not args.all_zarr:
        paths = [p for p in paths if os.path.basename(p).startswith(args.raw_prefix)]

    if not paths:
        hint = "Try --all-zarr to include merged datasets." if not args.all_zarr else ""
        print(f"No matching .zarr directories under {data_dir} {hint}".strip())
        sys.exit(0)

    entries = [analyze_one(p) for p in paths]
    print_report(entries, data_dir, raw_only=not args.all_zarr, raw_prefix=args.raw_prefix)


if __name__ == "__main__":
    main()
