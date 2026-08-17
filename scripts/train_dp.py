"""
Custom script to run training for the SSR Pick and Place system using the
diffusion_policy framework from the fork, but keeping the configuration files
and dataset definitions local to this project repo.
"""

import sys
import os
import pathlib

# Ensure we're running from the root of the project to match typical paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Inject the diffusion_policy fork into the path so it can be imported as a top-level module
DP_PATH = str(PROJECT_ROOT / "external" / "diffusion_policy")
sys.path.insert(0, DP_PATH)

# Inject the src/ directory so that `ssr.dataset.*` can be resolved by Hydra
SRC_PATH = str(PROJECT_ROOT / "src")
sys.path.insert(0, SRC_PATH)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf

# We import BaseWorkspace from the injected diffusion_policy path
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Define the path to our local config folder relative to this script
# Hydra's config_path is relative to the directory containing this script.
@hydra.main(
    version_base=None,
    config_path="../configs/dp",
    config_name="config/dp_opticalfiber_config"
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
