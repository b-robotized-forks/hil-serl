"""
Smoke test utility for the serl_ros2 adapter.

Spins up a ROS2 node, instantiates the adapter, and prints the first valid
observation received. Use this to verify adapter connectivity and topic wiring.

Usage:
    ros2 run serl_ros2 serl_ros2_smoke --config config/adapter_example.yaml
"""

import argparse
import time
from typing import Any

import rclpy
from rclpy.node import Node

from serl_ros2.robot_adapter import RobotAdapter


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="serl_ros2 adapter smoke test")
    parser.add_argument("--config", default=None, help="Path to adapter YAML config")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for observation")
    parser.add_argument("--max-age", type=float, default=0.5, help="Max snapshot age (seconds)")
    return parser.parse_args()


def main() -> None:
    """Run the smoke test."""
    args = parse_args()
    rclpy.init()
    node = Node("serl_ros2_smoke")
    adapter = RobotAdapter(node=node, config_path=args.config)

    node.get_logger().info(f"Waiting up to {args.timeout}s for observation...")

    start = time.time()
    obs: dict[str, Any] | None = None
    while rclpy.ok() and (time.time() - start) < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (time.time() - start) < 0.1:
            # wait at least a little while in order to make sure also the images have arrived.
            continue
        obs = adapter.get_observation(max_age_s=args.max_age)
        if obs is not None and obs["state"] is not None:
            break

    if obs is not None and obs["state"] is not None:
        node.get_logger().info("Observation received:")
        print(f"  timestamp: {obs['timestamp']}")
        print(f"  state keys: {list(obs['state'].keys())}")
        if obs["images"]:
            print(f"  image keys: {list(obs['images'].keys())}")
        print(f"  state: {obs['state']}")
    else:
        node.get_logger().warn("No fresh observation received within timeout.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
