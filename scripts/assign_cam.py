#!/usr/bin/env python3
"""
Interactive RealSense Camera Binding Tool

Discovers all connected cameras, displays feeds one at a time, and lets
you assign each to a role (Env / Wrist) via on-screen buttons.
Results are saved back to configs/hardware_config.yaml.

Edge cases handled:
  - T265 tracking cameras are automatically filtered out (pose-only, no RGB)
  - Each physical camera exposes multiple /dev/videoX nodes (depth, IR,
    metadata, colour); only the RGB/colour node is used
  - Cameras that fail to produce frames are skipped

Usage:
    python scripts/sys_test/assign_cam.py
"""

import sys
import os
import re
import time
import subprocess

import cv2
import numpy as np
import yaml

# ============================================================================
# Path setup
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, "src")

if src_path not in sys.path:
    sys.path.append(src_path)
if project_root not in sys.path:
    sys.path.append(project_root)

os.chdir(project_root)

# ============================================================================
# Constants
# ============================================================================
WINDOW_NAME = "Camera Binding Tool"
CANVAS_W, CANVAS_H = 960, 640

# Pixel‐format families — used to distinguish colour nodes from depth / IR
COLOR_FORMATS = {"YUYV", "MJPG", "RGB3", "BGR3", "NV12", "NV21", "UYVY", "RGBP", "BA24"}
DEPTH_IR_FORMATS = {"Z16", "Y16", "Y8", "GREY", "PAIR", "Y12I", "Y8I", "INZI", "INVI"}


# ============================================================================
# V4L2 device discovery
# ============================================================================
def parse_v4l2_devices():
    """
    Run ``v4l2-ctl --list-devices`` and group /dev/videoX nodes by device.

    Returns a list of dicts::

        [{"name": str, "usb_id": str, "nodes": ["/dev/videoX", ...]}, ...]
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=False,
        )
        output = result.stdout
    except FileNotFoundError:
        print("[错误] v4l2-ctl 未安装，请运行: sudo apt install v4l-utils")
        return []

    devices = []
    cur_name = None
    cur_usb_id = None
    cur_nodes = []

    def _flush():
        if cur_name and cur_nodes:
            devices.append({"name": cur_name, "usb_id": cur_usb_id,
                            "nodes": list(cur_nodes)})

    for line in output.splitlines():
        if line.startswith("\t") or line.startswith(" "):
            node = line.strip()
            if node.startswith("/dev/video"):
                cur_nodes.append(node)
        elif "(" in line and "):" in line:
            _flush()
            cur_nodes = []
            cur_name = line.split("(")[0].strip().rstrip(":")
            parts = re.findall(r"\(([^)]+)\)", line)
            usb_parts = [p for p in parts if "usb-" in p or "pci-" in p]
            cur_usb_id = usb_parts[-1] if usb_parts else (parts[-1] if parts else None)
        else:
            _flush()
            cur_name = cur_usb_id = None
            cur_nodes = []
    _flush()

    return devices


def _node_pixel_formats(node_path):
    """Return the set of pixel‐format FourCC codes a V4L2 node advertises."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", f"--device={node_path}", "--list-formats-ext"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=False,
        )
        return set(re.findall(r"'([A-Z0-9 ]{4})'", result.stdout))
    except Exception:
        return set()


def _is_t265(name):
    """Return True if the device name indicates a T265 / tracking camera."""
    low = name.lower()
    return "tracking" in low or "t265" in low


def _node_index(node_path):
    m = re.search(r"video(\d+)", node_path)
    return int(m.group(1)) if m else None


def _try_read_frame(video_index, timeout_frames=15):
    """Open a V4L2 node, try to grab a BGR frame. Returns frame or None."""
    cap = cv2.VideoCapture(video_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(timeout_frames):
        ret, f = cap.read()
        if ret and f is not None and len(f.shape) == 3 and f.shape[2] == 3:
            frame = f
            break
        time.sleep(0.05)
    cap.release()
    return frame


def discover_rgb_cameras():
    """
    Full discovery pipeline:

    1. Enumerate V4L2 device groups
    2. Filter out T265 (pose‐only, not RGB)
    3. For each device, find the colour/RGB node using pixel‐format probing
    4. Verify the node produces a real frame

    Returns a list of camera‐info dicts.
    """
    print("[发现] 正在枚举 V4L2 视频设备...")
    devices = parse_v4l2_devices()

    if not devices:
        print("[错误] 未发现任何视频设备。请检查相机连接。")
        return []

    print(f"[发现] 检测到 {len(devices)} 个设备组:")
    for d in devices:
        print(f"  - {d['name']} ({d['usb_id']}) -> {d['nodes']}")

    cameras = []
    for dev in devices:
        if _is_t265(dev["name"]):
            print(f"  [跳过] {dev['name']} (T265 追踪相机，非 RGB 输入)")
            continue

        print(f"  [探测] {dev['name']} ...", end=" ", flush=True)

        # Walk through nodes, find one whose advertised formats are colour‐only
        found = False
        for offset, node in enumerate(dev["nodes"]):
            fmts = _node_pixel_formats(node)
            stripped = {f.strip() for f in fmts}

            has_color = bool(stripped & COLOR_FORMATS)
            has_depth_ir = bool(stripped & DEPTH_IR_FORMATS)

            if not has_color or has_depth_ir:
                continue  # depth / IR / metadata node — skip

            vid_idx = _node_index(node)
            if vid_idx is None:
                continue

            # Final verification: can we actually read a colour frame?
            frame = _try_read_frame(vid_idx, timeout_frames=15)
            if frame is None:
                continue

            cameras.append({
                "name": dev["name"],
                "usb_id": dev["usb_id"],
                "video_index": vid_idx,
                "node": node,
                "offset": offset,
                "all_nodes": dev["nodes"],
            })
            print(f"RGB 可用 ({node}, 格式: {stripped & COLOR_FORMATS})")
            found = True
            break

        if not found:
            print("未找到可用 RGB 节点，跳过")

    print(f"\n[发现] 共找到 {len(cameras)} 个可用 RGB 相机\n")
    return cameras


# ============================================================================
# Interactive binding UI
# ============================================================================
_buttons = {}        # label -> (x, y, w, h)
_click_result = [None]  # mutable box for mouse callback


def _mouse_cb(event, x, y, _flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for label, (bx, by, bw, bh) in _buttons.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                _click_result[0] = label
                break


def _draw_button(canvas, x, y, w, h, label, *,
                 color=(70, 70, 70), text_color=(255, 255, 255),
                 highlight=False, disabled=False):
    """Draw a flat button and register its hit‐box."""
    if disabled:
        color = (35, 35, 35)
        text_color = (90, 90, 90)
    elif highlight:
        color = (45, 140, 60)

    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.55, 2
    tw, th = cv2.getTextSize(label, font, scale, thick)[0]
    cv2.putText(canvas, label, (x + (w - tw) // 2, y + (h + th) // 2),
                font, scale, text_color, thick)

    _buttons[label] = (x, y, w, h)


def run_binding_ui(cameras):
    """
    Show camera feeds one at a time with clickable buttons.

    Returns ``{"env": cam_info, "wrist": cam_info}`` (possibly partial).
    """
    if not cameras:
        print("[错误] 没有可用的相机")
        return {}

    bindings = {}      # role -> camera list‐index
    cur = [0]          # current camera index (mutable for inner funcs)
    cap = [None]       # current VideoCapture

    def open_cam(idx):
        if cap[0] is not None:
            cap[0].release()
        c = cv2.VideoCapture(cameras[idx]["video_index"], cv2.CAP_V4L2)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap[0] = c

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_cb)
    open_cam(0)

    print("[操作指南]")
    print("  点击 [Assign: Env]   → 将当前画面绑定为环境相机")
    print("  点击 [Assign: Wrist] → 将当前画面绑定为手腕相机")
    print("  点击 [<< Prev] / [Next >>]  切换相机")
    print("  点击 [Save & Exit]   → 保存绑定至配置文件")
    print("  按 'q'               → 退出（不保存）\n")

    while True:
        _buttons.clear()
        _click_result[0] = None

        cam = cameras[cur[0]]

        # --- Read frame ---
        frame = None
        if cap[0] is not None and cap[0].isOpened():
            ret, frame = cap[0].read()
            if not ret:
                frame = None

        # --- Build canvas ---
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

        # Header
        hdr_h = 55
        cv2.rectangle(canvas, (0, 0), (CANVAS_W, hdr_h), (40, 40, 40), -1)
        cv2.putText(canvas,
                    f"Camera {cur[0] + 1}/{len(cameras)}: {cam['name']}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas,
                    f"USB: {cam['usb_id']}  |  Node: {cam['node']}  |  "
                    f"video{cam['video_index']}",
                    (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        # Which role (if any) is this camera assigned to?
        this_role = None
        for role, idx in bindings.items():
            if idx == cur[0]:
                this_role = role
                break
        if this_role:
            badge = f"[Assigned: {this_role.upper()}]"
            cv2.putText(canvas, badge, (CANVAS_W - 260, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2)

        # Feed area
        feed_x, feed_y = 10, hdr_h + 5
        feed_w = CANVAS_W - 20
        feed_h = CANVAS_H - hdr_h - 125

        if frame is not None:
            display = cv2.resize(frame, (feed_w, feed_h))
            canvas[feed_y:feed_y + feed_h, feed_x:feed_x + feed_w] = display
        else:
            cv2.rectangle(canvas, (feed_x, feed_y),
                          (feed_x + feed_w, feed_y + feed_h), (30, 30, 30), -1)
            cv2.putText(canvas, "No frame", (feed_x + feed_w // 2 - 60,
                        feed_y + feed_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        # --- Button row ---
        btn_y = feed_y + feed_h + 10
        btn_h = 38

        _draw_button(canvas, 10, btn_y, 105, btn_h, "<< Prev",
                     disabled=(cur[0] == 0))
        _draw_button(canvas, 125, btn_y, 195, btn_h, "Assign: Env",
                     highlight=(bindings.get("env") == cur[0]))
        _draw_button(canvas, 330, btn_y, 195, btn_h, "Assign: Wrist",
                     highlight=(bindings.get("wrist") == cur[0]))
        _draw_button(canvas, 535, btn_y, 105, btn_h, "Next >>",
                     disabled=(cur[0] >= len(cameras) - 1))

        all_assigned = ("env" in bindings and "wrist" in bindings
                        and bindings["env"] != bindings["wrist"])
        _draw_button(canvas, CANVAS_W - 195, btn_y, 180, btn_h,
                     "Save & Exit",
                     color=(30, 130, 210) if all_assigned else (35, 35, 35),
                     disabled=not all_assigned)

        # --- Status bar ---
        st_y = btn_y + btn_h + 12
        cv2.rectangle(canvas, (0, st_y - 5), (CANVAS_W, CANVAS_H),
                      (25, 25, 25), -1)

        def _status(role):
            if role in bindings:
                c = cameras[bindings[role]]
                return f"{role.capitalize()}: Camera {bindings[role]+1} ({c['name']}, {c['usb_id']})"
            return f"{role.capitalize()}: [not assigned]"

        cv2.putText(canvas, _status("env"), (12, st_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(canvas, _status("wrist"), (12, st_y + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if not all_assigned:
            hint = "Assign both Env and Wrist cameras to enable Save"
            cv2.putText(canvas, hint, (CANVAS_W - 430, st_y + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            print("[退出] 未保存绑定")
            bindings.clear()
            break

        click = _click_result[0]
        if click == "<< Prev" and cur[0] > 0:
            cur[0] -= 1
            open_cam(cur[0])
        elif click == "Next >>" and cur[0] < len(cameras) - 1:
            cur[0] += 1
            open_cam(cur[0])
        elif click == "Assign: Env":
            # Toggle off if clicking the already‐assigned camera
            if bindings.get("env") == cur[0]:
                del bindings["env"]
                print(f"[取消] Env 绑定已移除")
            else:
                # If this camera was the wrist, unassign wrist first
                if bindings.get("wrist") == cur[0]:
                    del bindings["wrist"]
                bindings["env"] = cur[0]
                print(f"[绑定] Env -> Camera {cur[0]+1}: {cam['name']} ({cam['usb_id']})")
        elif click == "Assign: Wrist":
            if bindings.get("wrist") == cur[0]:
                del bindings["wrist"]
                print(f"[取消] Wrist 绑定已移除")
            else:
                if bindings.get("env") == cur[0]:
                    del bindings["env"]
                bindings["wrist"] = cur[0]
                print(f"[绑定] Wrist -> Camera {cur[0]+1}: {cam['name']} ({cam['usb_id']})")
        elif click == "Save & Exit" and all_assigned:
            break

    if cap[0] is not None:
        cap[0].release()
    cv2.destroyAllWindows()

    return {role: cameras[idx] for role, idx in bindings.items()}


# ============================================================================
# Save to hardware_config.yaml
# ============================================================================
def save_bindings(bindings):
    """Write camera bindings into the ``cameras.realsense`` list."""
    config_path = os.path.join(project_root, "configs", "hardware_config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    rs_list = []
    for role in ("env", "wrist"):
        if role in bindings:
            cam = bindings[role]
            rs_list.append({
                "name": f"rs_{role}",
                "id": cam["usb_id"],
                "offset": cam["offset"],
                "zoom": 1.0,
            })

    config.setdefault("cameras", {})["realsense"] = rs_list

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False,
                  allow_unicode=True, sort_keys=False)

    print(f"\n[保存] 相机绑定已写入: {config_path}")
    for role, cam in bindings.items():
        print(f"  rs_{role}: id={cam['usb_id']}, offset={cam['offset']}, "
              f"node={cam['node']}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  RealSense 相机交互式绑定工具")
    print("=" * 60)

    cameras = discover_rgb_cameras()

    if not cameras:
        print("\n没有找到可用的 RGB 相机。请检查连接后重试。")
        return

    bindings = run_binding_ui(cameras)

    if bindings:
        save_bindings(bindings)
        print("\n绑定完成！")
    else:
        print("\n未绑定任何相机，配置文件未修改。")


if __name__ == "__main__":
    main()
