#!/usr/bin/env python3
"""
Live MANUS glove data monitor — prints incoming ZMQ data in a labelled,
human-readable format.

Prerequisites:
  1. Run the MANUS SDK:  ./SDKClient_Linux.out  (press 1 for standalone)
  2. Run this script:    python tests/test_manus_live.py

Options:
  --host HOST     ZMQ host  (default: localhost)
  --port PORT     ZMQ port  (default: 8000)
  --rate HZ       Max print rate in Hz — 0 for unlimited  (default: 5)
  --full          Show all 25 joints; otherwise shows fingertips only
  --ergo          Show raw ergonomics angle vector when received
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import zmq

# ---------------------------------------------------------------------------
# Glove serial numbers
# ---------------------------------------------------------------------------
LEFT_GLOVE_SN  = "4848debd"
RIGHT_GLOVE_SN = "db397317"

# ---------------------------------------------------------------------------
# Layout constants (must match SDKClient.cpp stream format)
# ---------------------------------------------------------------------------
NUM_JOINTS     = 25
VALUES_PER_JOINT = 7   # x, y, z, qx, qy, qz, qw per joint
PAYLOAD_LEN    = 1 + NUM_JOINTS * VALUES_PER_JOINT   # 176 per hand

# Named joint index map (FINGER_CONNECTIONS order from visualize_skeleton.py)
JOINT_NAMES = {
    0:  "Wrist",
    1:  "Thumb  MCP", 2:  "Thumb  PIP", 3:  "Thumb  DIP", 4:  "Thumb  Tip",
    5:  "Index  MCP", 6:  "Index  PIP", 7:  "Index  DIP", 8:  "Index  Tip",
    9:  "Middle MCP", 10: "Middle PIP", 11: "Middle DIP", 12: "Middle Tip",
    13: "Ring   MCP", 14: "Ring   PIP", 15: "Ring   DIP", 16: "Ring   Tip",
    17: "Pinky  MCP", 18: "Pinky  PIP", 19: "Pinky  DIP", 20: "Pinky  Tip",
    21: "Palm   A",   22: "Palm   B",   23: "Palm   C",   24: "Palm   D",
}

FINGERTIP_NODES = [4, 8, 12, 16, 20]   # one per finger, thumb → pinky


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_hand(fields: list[str], full: bool) -> None:
    """Print labelled data for one 176-element hand payload."""
    if len(fields) < PAYLOAD_LEN:
        print(f"  [WARN] short payload: {len(fields)} fields (expected {PAYLOAD_LEN})")
        return

    serial = fields[0]
    sn_label = (
        "LEFT " if serial == LEFT_GLOVE_SN else
        "RIGHT" if serial == RIGHT_GLOVE_SN else
        "?????"
    )

    joints_to_print = range(NUM_JOINTS) if full else FINGERTIP_NODES

    print(f"  ┌── {sn_label}  SN={serial}")
    for j in joints_to_print:
        base = 1 + j * VALUES_PER_JOINT
        x,  y,  z  = float(fields[base]),     float(fields[base+1]), float(fields[base+2])
        qx, qy, qz = float(fields[base+3]),   float(fields[base+4]), float(fields[base+5])
        qw          = float(fields[base+6])
        name        = JOINT_NAMES.get(j, f"Joint{j:02d}")
        print(f"  │  [{j:02d}] {name:<12}  pos=({x:+.4f}, {y:+.4f}, {z:+.4f})"
              f"  rot=({qx:+.3f}, {qy:+.3f}, {qz:+.3f}, {qw:+.3f})")
    print(f"  └─────────────────────────────────────────────────────")


def _parse_ergonomics(fields: list[str]) -> None:
    """Print the 20+20 ergonomics angle vector."""
    floats = list(map(float, fields[:40]))
    print("  ┌── ERGONOMICS  (joint flex angles, both hands)")
    print(f"  │  Left  [{', '.join(f'{v:+.3f}' for v in floats[:20])}]")
    print(f"  │  Right [{', '.join(f'{v:+.3f}' for v in floats[20:])}]")
    print(f"  └─────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rate", type=float, default=5.0,
                        help="Max display rate in Hz (0 = unlimited)")
    parser.add_argument("--full", action="store_true",
                        help="Show all 25 joints instead of fingertips only")
    parser.add_argument("--ergo", action="store_true",
                        help="Also print ergonomics angle packets when received")
    args = parser.parse_args()

    address = f"tcp://{args.host}:{args.port}"
    min_interval = (1.0 / args.rate) if args.rate > 0 else 0.0

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.CONFLATE, 1)   # keep only latest
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(address)

    mode = "all 25 joints" if args.full else "fingertips only"
    print(f"\n[MANUS live monitor]  {address}  |  {mode}  |  max {args.rate} Hz")
    print(f"  Left  SN : {LEFT_GLOVE_SN}")
    print(f"  Right SN : {RIGHT_GLOVE_SN}")
    print("  Waiting for data  (Ctrl-C to quit)...\n")

    frame     = 0
    last_print = 0.0

    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.Again:
                print("[WARN] No data received — is the SDK running?")
                continue

            now = time.monotonic()
            if now - last_print < min_interval:
                continue
            last_print = now

            message = raw.decode("utf-8")
            fields  = message.split(",")
            n       = len(fields)

            frame += 1
            print(f"\n── Frame {frame:06d}  ({n} fields) {'─'*40}")

            if n == 40 and args.ergo:
                _parse_ergonomics(fields)
            elif n == PAYLOAD_LEN:                      # 176: single hand
                _parse_hand(fields, args.full)
            elif n == PAYLOAD_LEN * 2:                  # 352: two hands
                _parse_hand(fields[:PAYLOAD_LEN],         args.full)
                _parse_hand(fields[PAYLOAD_LEN:],         args.full)
            else:
                print(f"  [SKIP] unrecognised payload length: {n}")

    except KeyboardInterrupt:
        print(f"\n[done]  {frame} frames displayed.")
    finally:
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
