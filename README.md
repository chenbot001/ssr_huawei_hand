# SSR UR5 + RYHand Teleoperation

This repository contains one teleoperation path:

- A serial-selected Vive tracker controls a UR5 in Cartesian space through
  clutch-relative SteamVR motion.
- Manus Core 3.1.1 skeletons are retargeted through PyBullet IK to a RYHand.

After a live mode starts, fresh MANUS samples control the RYHand continuously.
The UR starts with its clutch engaged, so tracker motion is ignored and the TCP
is held. Press **Space** to release the clutch: the current Vive and TCP poses
become references, and subsequent Vive pose deltas drive the UR. Press Space
again to engage the clutch and hold the arm while hand tracking continues.
Stale Vive input engages the UR clutch and requires another explicit release;
stale MANUS input holds only the RYHand and resumes automatically when fresh.

The local visual console adds System, Manus, RYHand, and Teleoperation tabs.
Every tab uses the same four-panel layout: visual monitor, terminal event
stream, controls, and status. The Manus tab adapts EgoTac4D's live dual-hand
skeleton view, SDK skeleton settings, glove metadata, and official guided
MANUS calibration workflow to this project. The RYHand tab contains the
PyBullet thumb/index retargeting calibration.

## Setup

The supported runtime is Linux with Python 3.10 or newer.

```bash
python -m pip install -e .
```

Optional Intel RealSense RGB previews are installed separately:

```bash
python -m pip install -e '.[rgb]'
```

Place the vendor libraries at:

```text
src/hardware/lib/libRyhand64.so
src/hardware/manus_sdk/lib/libManusSDK_Integrated.so
```

Both shared libraries are intentionally ignored by Git. The official RYHand
URDF and its 26 meshes are bundled in `src/assets/ruihand15z`; no Bidex
submodule or Manus 2.4 bridge is used. The Manus client is derived from the
working EgoTac4D 3.1.1 integration and runs the Linux Integrated SDK as a
managed subprocess. Its use is governed by
`src/hardware/manus_sdk/MANUS_SDK_LICENSE.txt`.

Update the robot IP, Vive tracker serials, CAN interface, Manus UDP address, and
optional numeric glove IDs in `configs/hardware_config.yaml`. An ID of `0`
accepts and latches the first SDK-reported glove on that side for the process
lifetime. Set an explicit ID when multiple same-side gloves may be present.
Add optional RGB cameras by serial under `rgb_cameras`; an empty list keeps all
camera code inactive:

```yaml
rgb_cameras:
  - name: Workspace
    serial: "123456789"
```
Teleoperation timing, relative-motion limits, and servoL values are in
`configs/teleop_config.yaml`.

The Vive input uses SteamVR's background OpenVR interface and starts a headless
`vrserver` on Linux when one is not already running. Default teleoperation uses
the configured left tracker and left Manus glove. `--vive-right` and
`--manus-right` select them independently. Vive poses are never fed into the
Manus SDK and Manus wrist poses are never used by UR control. Every clutch
release captures the current Vive and UR TCP poses, so SteamVR's absolute
origin is never sent to the robot.

Bring up SocketCAN before using the hand:

```bash
bash scripts/ryhand_init.sh
```

The script accepts an optional interface name, for example
`bash scripts/ryhand_init.sh can1`.

## Commands

Start the loopback-only console:

```bash
ssr-console
```

It binds to `127.0.0.1:8768`, opens the browser, and starts with no task and no
motion armed. The System tab performs read-only readiness checks and shows a
compact hardware inventory. UR is connected only after a real RTDE receive
session returns a valid TCP pose, robot mode, and safety mode; port reachability
alone is insufficient. Enter a different robot IPv4 address and press
**Connect UR** to use it for the current console/teleop session.

**Run RYHand Init** explicitly executes `scripts/ryhand_init.sh` for the
configured CAN interface. The console uses non-interactive sudo so the browser
cannot hang on a hidden password prompt. If authorization is required, run
`sudo -v` in a terminal immediately before pressing the button. The console
owns one shared MANUS bridge and routes its calibrated skeleton to the active
tab. Only the Manus tab can change SDK skeleton settings or run official glove
calibration. The RYHand and Teleop tabs are read-only MANUS consumers; changing
tabs never starts a second SDK process or UDP receiver. RYHand calibration uses
PyBullet, with physical hand output disabled by default. The Teleop tab provides:

- **Full system:** Vive → physical UR5 and Manus → physical RYHand.
- **UR + Vive:** arm teleoperation without opening Manus or RYHand.
- **RYHand + Manus:** hand teleoperation without opening Vive or UR RTDE.
- **Virtual UR + RYHand:** both live inputs control simulated outputs and the
  controller; no actuator connection is opened. Use this when both physical
  outputs are unavailable. The Teleop visual shows the live UR-base-frame
  translation and rotation deltas relative to the TCP pose captured when the
  clutch was released. All six values read zero while the clutch is engaged.

Calibration and teleoperation are mutually exclusive. Space toggles the UR
clutch only while the Teleop tab is active. MANUS-to-RYHand commands remain
independent of that clutch. Moving away from a MANUS-driven Teleop session
revokes its skeleton input and holds the RYHand until the tab owns the stream
again. Moving away from RYHand calibration similarly holds its last hand pose.
RealSense preview is explicit and never starts on page load. Stop/Ctrl+C stops
UR servo motion and leaves the RYHand at its last pose.

Opening the Teleop tab starts independent read-only previews of the selected
Vive tracker and calibrated MANUS skeleton before a mode is started. The two
side selectors are independent: Vive controls only UR, while MANUS controls
only RYHand. Choose the mode from the dropdown; the compact settings below it
change with the selected output path. **Save Configuration** validates and atomically updates
`configs/hardware_config.yaml` and `configs/teleop_config.yaml`. Saved settings
apply to subsequent teleoperation sessions and survive console restarts.

Run the non-moving readiness check first:

```bash
ssr-check
```

It checks Python dependencies, the official URDF/meshes, both local shared
libraries, CAN state, UR RTDE receive, Vive pose input, and the configured
Manus glove. It never invokes `sudo` or sends robot motion commands.

Validate the live input and mapping path without connecting actuators:

```bash
ssr-teleop --dry-run
```

Run live teleoperation:

```bash
ssr-teleop
ssr-teleop --vive-right
ssr-teleop --manus-right
ssr-teleop --vive-right --manus-right
```

Calibrate the enabled thumb and index mapping:

```bash
ssr-calibrate-hand --dry-run
ssr-calibrate-hand
```

Calibration opens PyBullet sliders by default and saves atomically to
`configs/manus_calibration.json` on exit. Middle, ring, and pinky outputs are
intentionally held at zero; thumb swing is fixed at 10 degrees and index swing
at zero.

## Troubleshooting

- `libRyhand64.so is missing`: copy the RYHand vendor library to the fixed path above.
- `libManusSDK_Integrated.so is missing`: copy it from the Manus 3.1.1 Linux SDK.
- `RYHand URDF is missing`: reinstall the repository so its bundled assets are present.
- `CAN interface is not UP`: run `scripts/ryhand_init.sh` and verify the adapter.
- `UR RTDE failed`: confirm the configured IP, Remote Control mode, and that no
  other RTDE control client is connected.
- `No Manus sample`: check the managed SDK error reported by `ssr-check`, the
  Manus dongle, UDP bind address, and any configured glove ID.
- `No pose from Vive tracker`: confirm SteamVR sees the configured serial,
  Lighthouse tracking is valid, and no stale `vrserver` process is failing.
- Unexpected robot direction: stop teleoperation and verify
  `VIVE_WORLD_TO_UR_BASE` against the installed robot/base-station orientation
  before live use.
