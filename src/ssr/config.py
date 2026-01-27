import os
import yaml
from pathlib import Path

# Base paths
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

def load_yaml(filename):
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Load configurations once
_hardware_conf = None
_teleop_conf = None

def get_hardware_config():
    global _hardware_conf
    if _hardware_conf is None:
        _hardware_conf = load_yaml("hardware_config.yaml")
    return _hardware_conf

def get_teleop_config():
    global _teleop_conf
    if _teleop_conf is None:
        _teleop_conf = load_yaml("teleop_config.yaml")
    return _teleop_conf

# Helper accessors for backward compatibility or ease of use
def get_ur_ip():
    return get_hardware_config()['ur_arm']['ip']

def get_gello_port():
    return get_hardware_config()['gello']['port']
