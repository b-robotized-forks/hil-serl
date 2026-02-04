"""
Relative frame wrapper for RobotEnv-style environments.

Origin: hil-serl/serl_robot_infra/franka_env/envs/relative_env.py
Modified: Yes
Changes:
  - Removed dual-arm wrapper support to keep the single-arm-only scope.
  - Updated imports to serl_framework utilities (no franka_env dependency).
  - adjusted import `from gym import Env` to use `gymnasium` instead.
"""
import copy
from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gymnasium import Env
from serl_framework.utils.transformations import (
    construct_transform_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.transform_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update transform matrix
        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['original_state_obs'] = copy.deepcopy(obs['state'])

        # Update transform matrix
        self.transform_matrix = construct_transform_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the transform matrix
        """
        transform_inv = np.linalg.inv(self.transform_matrix)
        obs["state"]["tcp_vel"] = transform_inv @ obs["state"]["tcp_vel"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into into spatial(base) frame
        using the transform matrix. 
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.transform_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the transform matrix.
        """
        action = np.array(action)
        action[:6] = np.linalg.inv(self.transform_matrix) @ action[:6]
        return action
