sudo ip link set can0 down

sudo ip link set can0 up type can bitrate 1000000

ip -details link show can0