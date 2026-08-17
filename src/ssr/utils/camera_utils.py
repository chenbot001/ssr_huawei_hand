import subprocess
import re
import os
import time

def list_v4l2_devices():
    """
    Parses v4l2-ctl --list-devices to return a mapping of 
    Stable ID (USB Path) -> List of Video Nodes.
    """
    try:
        # Use subprocess.run to capture output even if exit code is non-zero
        # some v4l2-ctl versions exit with 1 if they can't open a specific device (like /dev/video0)
        # but they still print the other devices to stdout.
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"], 
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, # Suppress "Cannot open device" errors
            text=True, 
            check=False
        )
        output = result.stdout
    except FileNotFoundError:
        return {}

    devices = {}
    current_id = None
    
    # Example segments:
    # HD WebCam: HD WebCam (usb-0000:00:14.0-5.1.2):
    #         /dev/video1
    #         /dev/video3
    
    lines = output.splitlines()
    for line in lines:
        if line.startswith('\t') or line.startswith(' '): # It's a video node
            if current_id:
                node = line.strip()
                if node.startswith('/dev/video'):
                    devices[current_id].append(node)
        elif '(' in line and '):' in line:
            # Find the USB ID which is usually in the last set of parentheses
            # e.g., "HD WebCam (usb-0000:00:14.0-5.1.2):"
            # or "Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-5.2):"
            parts = re.findall(r'\(([^)]+)\)', line)
            if parts:
                # Look for the part that starts with 'usb-' or 'pci-'
                usb_indices = [i for i, p in enumerate(parts) if 'usb-' in p or 'pci-' in p]
                if usb_indices:
                    current_id = parts[usb_indices[-1]]
                else:
                    current_id = parts[-1] # Fallback to last one
                devices[current_id] = []
        else:
            current_id = None
            
    return devices

def get_video_index_by_id(stable_id, offset=0):
    """
    Returns the integer index for a camera given its stable USB ID.
    If offset is provided, it returns the N-th node associated with that ID.
    """
    devices = list_v4l2_devices()
    if stable_id in devices and len(devices[stable_id]) > offset:
        node = devices[stable_id][offset]
        # Extract index from /dev/videoN
        return int(re.search(r'video(\d+)', node).group(1))
    
    return None


def find_rgb_video_index_for_usb(stable_id, width=640, height=480, timeout_reads=30):
    """
    在指定 USB 稳定 ID 下的所有 /dev/video* 节点上依次尝试读取一帧彩色图，
    找到第一个能稳定出 BGR 画面的节点。用于重启后 ``offset`` 与节点顺序变化时的回退。

    Returns:
        (video_index, offset) 或 (None, None)
    """
    try:
        import cv2
    except ImportError:
        return None, None

    devices = list_v4l2_devices()
    if stable_id not in devices:
        return None, None

    nodes = devices[stable_id]
    for offset, node in enumerate(nodes):
        m = re.search(r"video(\d+)", node)
        if not m:
            continue
        idx = int(m.group(1))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        ok = False
        for _ in range(timeout_reads):
            ret, frame = cap.read()
            if (
                ret
                and frame is not None
                and len(frame.shape) == 3
                and frame.shape[2] == 3
                and frame.size > 0
            ):
                ok = True
                break
            time.sleep(0.05)
        cap.release()
        if ok:
            return idx, offset

    return None, None


if __name__ == "__main__":
    # Test
    print("Detected Devices:")
    devs = list_v4l2_devices()
    for sid, nodes in devs.items():
        print(f"  {sid}: {nodes}")
