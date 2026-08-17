#!/usr/bin/env bash
set -euo pipefail

interface="${1:-can0}"

sudo ip link set "$interface" down 2>/dev/null || true
sudo ip link set "$interface" up type can bitrate 1000000
ip -details link show "$interface"
