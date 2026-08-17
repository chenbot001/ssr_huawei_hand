#!/usr/bin/env python3
"""
data_overlay_live.py — Overlay training data onion-skin on live env camera feed.

The training-data overlay is composited on top of the live camera frame so you
can visually verify that the camera perspective still matches the dataset.

Usage
-----
  python scripts/data_overlay_live.py
  python scripts/data_overlay_live.py --zarr data/assembly/ryhand_assembly_dataset.zarr
  python scripts/data_overlay_live.py --alpha 0.45

Controls (OpenCV window)
------------------------
  Trackbar "Overlay alpha" — blend weight of the training overlay (0–100 %)
  q                        — quit
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import cv2
import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_PATH = str(PROJECT_ROOT / "src")
for _p in (SRC_PATH, str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ssr.hardware.realsense_worker import RealSenseWorker
from ssr.config import get_hardware_config
from ssr.utils.camera_utils import get_video_index_by_id, find_rgb_video_index_for_usb

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ZARR = "data/assembly/ryhand_assembly_dataset.zarr"
DEFAULT_ALPHA = 0.40   # overlay weight (0 = fully live, 1 = fully overlay)


# ---------------------------------------------------------------------------
# Build onion-skin overlay from zarr dataset
# ---------------------------------------------------------------------------
def build_overlay(zarr_path: str) -> np.ndarray:
    """Return a uint8 RGB image that is the equal-weight blend of every
    episode's first frame from *camera_env*."""
    root = zarr.open(zarr_path, mode="r")
    episode_ends = root["meta/episode_ends"][:120]
    first_indices = [0] + [int(i) + 1 for i in episode_ends[:-1]]

    camera_env = root["data/camera_env"] 
    num_frames = len(camera_env)

    frames: list[np.ndarray] = []
    for idx in first_indices:
        if 0 <= idx < num_frames:
            frames.append(camera_env[idx])
        else:
            print(f"[overlay] Frame {idx} out of bounds — skipped")

    if not frames:
        raise RuntimeError("No valid frames found in the dataset.")

    acc = np.zeros_like(frames[0], dtype=np.float32)
    for f in frames:
        acc += f.astype(np.float32) / 255.0
    acc /= len(frames)

    overlay = np.clip(acc * 255, 0, 255).astype(np.uint8)
    print(f"[overlay] Built from {len(frames)} episode start frames  shape={overlay.shape}")
    return overlay   # H x W x 3, RGB


# ---------------------------------------------------------------------------
# Camera init (mirrors deploy_policy_earbud_demo.py)
# ---------------------------------------------------------------------------
def init_env_camera(hw_config: dict, img_w: int, img_h: int) -> RealSenseWorker | None:
    rs_configs = hw_config.get("cameras", {}).get("realsense", [])
    rs_by_name = {cfg.get("name", ""): cfg for cfg in rs_configs}
    cfg = rs_by_name.get("rs_env")
    if cfg is None:
        print("[camera] No 'rs_env' entry in hardware config.")
        return None

    cam_id     = cfg.get("id", "")
    cam_offset = cfg.get("offset", 0)
    cam_zoom   = cfg.get("zoom", 1.0)
    serial     = (cfg.get("serial") or cfg.get("serial_number") or "").strip()

    if serial:
        try:
            worker = RealSenseWorker(width=img_w, height=img_h, serial_number=serial)
        except (ValueError, ImportError) as e:
            print(f"[camera] RealSense SDK init failed: {e}")
            return None
    else:
        video_idx = get_video_index_by_id(cam_id, cam_offset)
        if video_idx is None and cam_id:
            video_idx, _ = find_rgb_video_index_for_usb(cam_id, img_w, img_h)
        if video_idx is None:
            print("[camera] Cannot find video device for rs_env.")
            return None
        worker = RealSenseWorker(camera_index=video_idx, width=img_w, height=img_h)

    worker.set_zoom(cam_zoom)
    worker.daemon = True
    worker.start()

    for _ in range(50):
        if worker.get_latest_frame() is not None:
            break
        time.sleep(0.1)

    if worker.get_latest_frame() is None:
        print("[camera] Camera did not produce frames — aborting.")
        worker.stop()
        return None

    print(f"[camera] Env camera ready (serial={serial or 'OpenCV'})")
    return worker


# ---------------------------------------------------------------------------
# Resize a frame to target (h, w) — matches field of view without cropping
# ---------------------------------------------------------------------------
def resize_to_overlay(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if frame.shape[0] == target_h and frame.shape[1] == target_w:
        return frame
    return cv2.resize(frame, (target_w, target_h))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Overlay training data on live env camera.")
    parser.add_argument("--zarr",  default=DEFAULT_ZARR,
                        help="Path to the zarr dataset (default: %(default)s)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Initial overlay blend weight 0-1 (default: %(default)s)")
    args = parser.parse_args()

    zarr_path = str(PROJECT_ROOT / args.zarr) if not pathlib.Path(args.zarr).is_absolute() else args.zarr

    # -- Build training overlay --------------------------------------------------
    print(f"[overlay] Loading zarr from: {zarr_path}")
    overlay_rgb = build_overlay(zarr_path)
    overlay_h, overlay_w = overlay_rgb.shape[:2]

    # -- Init camera ------------------------------------------------------------
    hw_config = get_hardware_config()

    # Request the full native camera resolution so we have room to crop.
    # The overlay dimensions define the crop target.
    native_w, native_h = 640, 480
    rs_env = init_env_camera(hw_config, native_w, native_h)
    if rs_env is None:
        print("[camera] Could not open env camera. Exiting.")
        sys.exit(1)

    # -- OpenCV window with trackbar --------------------------------------------
    win = "Overlay Alignment Check  (q = quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    TRACKBAR = "Overlay alpha %"
    alpha_pct = int(args.alpha * 100)
    cv2.createTrackbar(TRACKBAR, win, alpha_pct, 100, lambda v: None)

    # Convert overlay to a contour image for sharp alignment reference.
    # Edges are drawn in cyan on a black background.
    _ov_bgr  = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    _ov_gray = cv2.cvtColor(_ov_bgr, cv2.COLOR_BGR2GRAY)
    _ov_blur = cv2.GaussianBlur(_ov_gray, (3, 3), 0)
    _edges   = cv2.Canny(_ov_blur, threshold1=30, threshold2=80)
    # Colour the contours cyan on black
    overlay_contour = np.zeros((*_edges.shape, 3), dtype=np.uint8)
    overlay_contour[_edges > 0] = (0, 255, 255)   # cyan in BGR
    overlay_bgr = overlay_contour.astype(np.float32)

    print("[run] Window open — adjust the trackbar to change blend weight. Press 'q' to quit.")

    while True:
        # Grab live frame
        raw = rs_env.get_latest_frame()   # BGR, native resolution
        if raw is None:
            time.sleep(0.02)
            continue

        # Resize to match overlay dimensions
        live_bgr = resize_to_overlay(raw, overlay_h, overlay_w).astype(np.float32)

        # Blend: contour pixels are composited additively over the live image
        # so the live view is always fully visible underneath.
        alpha = cv2.getTrackbarPos(TRACKBAR, win) / 100.0
        blended = np.clip(live_bgr + overlay_bgr * alpha, 0, 255).astype(np.uint8)
        display = blended

        # Annotate
        label = f"contour overlay {alpha*100:.0f}%"
        cv2.putText(display, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 0), 1, cv2.LINE_AA)

        display = cv2.resize(display, (overlay_w * 2, overlay_h * 2), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    rs_env.stop()
    cv2.destroyAllWindows()
    print("[run] Done.")


if __name__ == "__main__":
    main()
