from __future__ import annotations

from copy import deepcopy
from ipaddress import IPv4Address, ip_address
import math
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

_hardware_config: dict[str, Any] | None = None
_teleop_config: dict[str, Any] | None = None
_config_lock = threading.Lock()

_EDITABLE_TELEOP_FIELDS = frozenset(
    {
        "ur_ip",
        "can_port",
        "translation_scale",
        "servo_speed",
        "servo_acceleration",
        "update_rate",
        "input_timeout",
        "hand_motor_speed",
        "max_linear_speed",
        "max_angular_speed",
    }
)


def load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return data


def _require(config: dict[str, Any], path: tuple[str, ...], filename: str) -> None:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(path)
            raise ValueError(f"Missing required setting {dotted!r} in {filename}")
        value = value[key]


def get_hardware_config() -> dict[str, Any]:
    global _hardware_config
    if _hardware_config is None:
        config = load_yaml("hardware_config.yaml")
        for required in (
            ("ur_arm", "ip"),
            ("vive_tracker", "left_serial"),
            ("vive_tracker", "right_serial"),
            ("manus_glove", "address"),
            ("manus_glove", "left_id"),
            ("manus_glove", "right_id"),
            ("ruiyan_hand", "port"),
        ):
            _require(config, required, "hardware_config.yaml")
        cameras = config.get("rgb_cameras", [])
        if not isinstance(cameras, list):
            raise ValueError("rgb_cameras must be a list in hardware_config.yaml")
        for index, camera in enumerate(cameras):
            if not isinstance(camera, dict) or not isinstance(camera.get("serial"), str):
                raise ValueError(
                    f"rgb_cameras[{index}] must contain a string serial"
                )
            camera.setdefault("name", f"RGB Camera {index + 1}")
        config["rgb_cameras"] = cameras
        _hardware_config = config
    return _hardware_config


def get_teleop_config() -> dict[str, Any]:
    global _teleop_config
    if _teleop_config is None:
        config = load_yaml("teleop_config.yaml")
        for required in (
            ("servo", "speed"),
            ("servo", "acceleration"),
            ("servo", "dt"),
            ("servo", "lookahead_time"),
            ("servo", "gain"),
            ("vive", "translation_scale"),
            ("control", "update_rate"),
            ("control", "input_timeout"),
            ("control", "hand_motor_speed"),
            ("safety", "max_linear_speed"),
            ("safety", "max_angular_speed"),
            ("safety", "max_translation_from_reference"),
            ("safety", "max_rotation_from_reference"),
            ("safety", "max_tracker_translation_jump"),
            ("safety", "max_tracker_rotation_jump"),
        ):
            _require(config, required, "teleop_config.yaml")
        _teleop_config = config
    return _teleop_config


def get_console_teleop_config() -> dict[str, Any]:
    hardware = get_hardware_config()
    teleop = get_teleop_config()
    return {
        "ur_ip": str(hardware["ur_arm"]["ip"]),
        "can_port": str(hardware["ruiyan_hand"]["port"]),
        "translation_scale": float(teleop["vive"]["translation_scale"]),
        "servo_speed": float(teleop["servo"]["speed"]),
        "servo_acceleration": float(teleop["servo"]["acceleration"]),
        "update_rate": float(teleop["control"]["update_rate"]),
        "input_timeout": float(teleop["control"]["input_timeout"]),
        "hand_motor_speed": int(teleop["control"]["hand_motor_speed"]),
        "max_linear_speed": float(teleop["safety"]["max_linear_speed"]),
        "max_angular_speed": float(teleop["safety"]["max_angular_speed"]),
    }


def _bounded_number(
    values: dict[str, Any], key: str, minimum: float, maximum: float
) -> float:
    if isinstance(values[key], bool):
        raise ValueError(f"{key} must be numeric")
    try:
        value = float(values[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be numeric") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum:g} and {maximum:g}")
    return value


def _write_yaml_atomically(path: Path, values: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            yaml.safe_dump(values, stream, sort_keys=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_console_teleop_config(values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(values, dict) or not values:
        raise ValueError("At least one teleoperation setting is required")
    unknown = set(values) - _EDITABLE_TELEOP_FIELDS
    if unknown:
        raise ValueError(f"Unknown teleoperation setting: {sorted(unknown)[0]}")

    with _config_lock:
        hardware = deepcopy(get_hardware_config())
        teleop = deepcopy(get_teleop_config())

        if "ur_ip" in values:
            try:
                parsed = ip_address(str(values["ur_ip"]).strip())
            except ValueError as exc:
                raise ValueError("ur_ip must be a valid IPv4 address") from exc
            if not isinstance(parsed, IPv4Address):
                raise ValueError("ur_ip must be a valid IPv4 address")
            hardware["ur_arm"]["ip"] = str(parsed)
        if "can_port" in values:
            if not isinstance(values["can_port"], str):
                raise ValueError("can_port must be a valid Linux interface name")
            port = values["can_port"].strip()
            if re.fullmatch(r"[A-Za-z0-9_.-]{1,15}", port) is None:
                raise ValueError("can_port must be a valid Linux interface name")
            hardware["ruiyan_hand"]["port"] = port
        if "translation_scale" in values:
            teleop["vive"]["translation_scale"] = _bounded_number(
                values, "translation_scale", 0.05, 3.0
            )
        if "servo_speed" in values:
            teleop["servo"]["speed"] = _bounded_number(
                values, "servo_speed", 0.01, 2.0
            )
        if "servo_acceleration" in values:
            teleop["servo"]["acceleration"] = _bounded_number(
                values, "servo_acceleration", 0.01, 5.0
            )
        if "update_rate" in values:
            update_rate = _bounded_number(values, "update_rate", 10.0, 250.0)
            teleop["control"]["update_rate"] = update_rate
            teleop["servo"]["dt"] = 1.0 / update_rate
        if "input_timeout" in values:
            teleop["control"]["input_timeout"] = _bounded_number(
                values, "input_timeout", 0.05, 2.0
            )
        if "hand_motor_speed" in values:
            speed = _bounded_number(values, "hand_motor_speed", 1, 65535)
            if not speed.is_integer():
                raise ValueError("hand_motor_speed must be an integer")
            teleop["control"]["hand_motor_speed"] = int(speed)
        if "max_linear_speed" in values:
            teleop["safety"]["max_linear_speed"] = _bounded_number(
                values, "max_linear_speed", 0.01, 1.0
            )
        if "max_angular_speed" in values:
            teleop["safety"]["max_angular_speed"] = _bounded_number(
                values, "max_angular_speed", 0.05, 5.0
            )
        if {"ur_ip", "can_port"} & values.keys():
            _write_yaml_atomically(CONFIG_DIR / "hardware_config.yaml", hardware)
        if values.keys() - {"ur_ip", "can_port"}:
            _write_yaml_atomically(CONFIG_DIR / "teleop_config.yaml", teleop)

        global _hardware_config, _teleop_config
        _hardware_config = hardware
        _teleop_config = teleop
        return get_console_teleop_config()


def clear_config_cache() -> None:
    """Clear cached YAML values. Intended for tests and calibration reloads."""
    global _hardware_config, _teleop_config
    _hardware_config = None
    _teleop_config = None
