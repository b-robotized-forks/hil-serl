#!/usr/bin/env python3
"""
Smoke test for RobotEnv using a mock adapter.

Origin: New file for ROS2 migration
Modified: Yes
Changes:
  - Switched to shared MockRobotAdapter helpers in serl_framework.testing.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

HIL_SERL_ROOT = Path(__file__).resolve().parents[2]
for package_path in (HIL_SERL_ROOT / "serl_framework",):
    if str(package_path) not in sys.path:
        sys.path.insert(0, str(package_path))

from serl_framework.envs.robot_env import DefaultEnvConfig, RobotEnv
from serl_framework.testing.mock_adapter import MockRobotAdapter, make_mock_snapshot


def _load_config(path: Optional[str]) -> DefaultEnvConfig:
    """
    Load the RobotEnv config, ensuring at least one camera entry.

    Args:
        path: Optional YAML path.

    Returns:
        Config instance with camera defaults set.
    """
    config = DefaultEnvConfig.from_yaml(path) if path else DefaultEnvConfig()
    if not config.CAMERAS:
        config.CAMERAS = {"front": {}}
    config.DISPLAY_IMAGE = False
    config.ENABLE_KEYBOARD_LISTENER = False
    return config


def main() -> None:
    """
    Execute a minimal RobotEnv smoke test.
    """
    parser = argparse.ArgumentParser(description="RobotEnv smoke test")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=3, help="Number of steps to run")
    parser.add_argument("--hz", type=int, default=10, help="Control frequency")
    args = parser.parse_args()

    config = _load_config(args.config)
    camera_names = list(config.CAMERAS.keys())
    snapshot = make_mock_snapshot(camera_names, config.ROBOT_DOF)
    adapter = MockRobotAdapter(snapshot)

    env = RobotEnv(hz=args.hz, fake_env=False, save_video=False, config=config, adapter=adapter)
    action = np.zeros((7,), dtype=np.float32)

    for step_idx in range(args.steps):
        obs, reward, done, truncated, info = env.step(action)
        print(
            f"step {step_idx}: reward={reward}, done={done}, "
            f"images={list(obs['images'].keys())}, "
            f"state_keys={list(obs['state'].keys())}"
        )
        if done or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
