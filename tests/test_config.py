from pathlib import Path

import pytest
import yaml

import config as config_module
from config import get_hardware_config, get_teleop_config


def test_hardware_config_contains_only_retained_devices():
    config = get_hardware_config()
    assert set(config) == {
        "ur_arm",
        "vive_tracker",
        "manus_glove",
        "ruiyan_hand",
        "rgb_cameras",
    }
    assert set(config["ur_arm"]) == {"ip"}
    assert set(config["vive_tracker"]) == {"left_serial", "right_serial"}
    assert all(
        str(config["vive_tracker"][key]).startswith("LHR-")
        for key in ("left_serial", "right_serial")
    )
    assert set(config["ruiyan_hand"]) == {"port"}
    assert set(config["manus_glove"]) == {"address", "left_id", "right_id"}
    assert config["manus_glove"]["address"].startswith("udp://")
    assert isinstance(config["rgb_cameras"], list)


def test_teleop_config_contains_only_runtime_settings():
    config = get_teleop_config()
    assert set(config) == {"servo", "vive", "control", "safety"}
    assert config["control"]["update_rate"] == 80.0
    assert config["control"]["input_timeout"] == 0.25
    assert config["servo"]["dt"] == 1.0 / config["control"]["update_rate"]
    assert set(config["vive"]) == {"translation_scale"}


def test_console_teleop_config_saves_validated_yaml_atomically(monkeypatch, tmp_path: Path):
    hardware = {
        "ur_arm": {"ip": "192.168.0.2"},
        "vive_tracker": {"left_serial": "LHR-LEFT", "right_serial": "LHR-RIGHT"},
        "manus_glove": {"address": "udp://127.0.0.1:9001", "left_id": 0, "right_id": 0},
        "ruiyan_hand": {"port": "can0"},
        "rgb_cameras": [],
    }
    teleop = {
        "servo": {"speed": 0.5, "acceleration": 0.5, "dt": 0.0125, "lookahead_time": 0.1, "gain": 300},
        "vive": {"translation_scale": 1.0},
        "control": {"update_rate": 80.0, "input_timeout": 0.25, "hand_motor_speed": 1000},
        "safety": {
            "max_linear_speed": 0.2,
            "max_angular_speed": 0.8,
            "max_translation_from_reference": 0.3,
            "max_rotation_from_reference": 1.0472,
            "max_tracker_translation_jump": 0.05,
            "max_tracker_rotation_jump": 0.35,
        },
    }
    (tmp_path / "hardware_config.yaml").write_text(yaml.safe_dump(hardware), encoding="utf-8")
    (tmp_path / "teleop_config.yaml").write_text(yaml.safe_dump(teleop), encoding="utf-8")
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)
    config_module.clear_config_cache()
    try:
        saved = config_module.save_console_teleop_config(
            {
                "ur_ip": "192.168.0.9",
                "can_port": "can7",
                "update_rate": 100,
                "input_timeout": 0.3,
                "hand_motor_speed": 1200,
            }
        )
        on_disk_hardware = yaml.safe_load((tmp_path / "hardware_config.yaml").read_text())
        on_disk_teleop = yaml.safe_load((tmp_path / "teleop_config.yaml").read_text())

        assert saved["ur_ip"] == "192.168.0.9"
        assert on_disk_hardware["ur_arm"]["ip"] == "192.168.0.9"
        assert on_disk_hardware["ruiyan_hand"]["port"] == "can7"
        assert on_disk_teleop["control"]["update_rate"] == 100.0
        assert on_disk_teleop["servo"]["dt"] == pytest.approx(0.01)
        assert on_disk_teleop["vive"] == {"translation_scale": 1.0}
        assert not list(tmp_path.glob("*.tmp"))

        with pytest.raises(ValueError, match="Unknown teleoperation setting"):
            config_module.save_console_teleop_config({"unknown": 1})
        with pytest.raises(ValueError, match="input_timeout"):
            config_module.save_console_teleop_config({"input_timeout": float("nan")})
        with pytest.raises(ValueError, match="Unknown teleoperation setting"):
            config_module.save_console_teleop_config({"mount_correction": {}})
    finally:
        config_module.clear_config_cache()
