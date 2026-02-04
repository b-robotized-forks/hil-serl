"""
Cube demo RobotEnv wrapper for ROS2 experiments.

This module provides a thin subclass that can manage adapter lifecycle for
the cube_demo_ros2 experiment.
"""

from serl_framework.envs.robot_env import RobotEnv
from serl_framework.robot_adapter import RobotAdapter


class CubeDemoEnv(RobotEnv):
    """
    Minimal RobotEnv subclass for the cube demo task.
    """

    def __init__(
        self,
        hz: int = 10,
        fake_env: bool = False,
        save_video: bool = False,
        config=None,
        set_load: bool = False,
        adapter: RobotAdapter | None = None,
        owns_adapter: bool = False,
    ) -> None:
        """
        Initialize the cube demo environment.

        Args:
            hz: Control frequency for the environment step loop.
            fake_env: If True, skip adapter interaction.
            save_video: If True, store video frames per reset.
            config: EnvConfig instance.
            set_load: If True, apply payload parameters at startup.
            adapter: RobotAdapter instance for robot communication.
            owns_adapter: If True, stop the adapter on close().
        """
        super().__init__(
            hz=hz,
            fake_env=fake_env,
            save_video=save_video,
            config=config,
            set_load=set_load,
            adapter=adapter,
        )
        self._owns_adapter = owns_adapter

    def close(self) -> None:
        """
        Close env resources and stop the adapter if this env owns it.
        """
        super().close()
        if self._owns_adapter and self.adapter is not None:
            self.adapter.stop()
