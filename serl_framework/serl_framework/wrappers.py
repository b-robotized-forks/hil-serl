import logging
import time
from gymnasium import Env, spaces
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
import copy
from scipy.spatial.transform import Rotation as R
from typing import Any, Callable, List
from serl_framework.teleop_adapter import TeleopAdapter
from serl_framework.utils.teleop import transform_delta_to_base

sigmoid = lambda x: 1 / (1 + np.exp(-x))
LOGGER = logging.getLogger(__name__)

class HumanClassifierWrapper(gym.Wrapper):
    """
    Prompts the user for success/failure when an episode ends.
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
    - No changes needed; robot-agnostic.
    """
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        if done:
            while True:
                try:
                    rew = int(input("Success? (1/0)"))
                    assert rew == 0 or rew == 1
                    break
                except:
                    continue
        info['succeed'] = rew
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
    
class MultiCameraBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    This wrapper uses the camera images to compute the reward,
    which is not part of the observation space
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - Extended reward function contract to optionally return debug info.
    """

    def __init__(
        self,
        env: Env,
        reward_classifier_func: Callable[[dict], int | tuple[int, dict[str, Any]]] | None,
        target_hz: float | None = None,
    ):
        """
        Initialize binary classifier reward wrapper.

        Args:
            env: Wrapped env.
            reward_classifier_func: Reward callback taking `obs` and returning either:
                - `int`/`bool` reward (legacy behavior), or
                - `(reward, info_dict)` where `info_dict` is merged into step info.
            target_hz: Optional wrapper-side pacing limit.
        """
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.target_hz = target_hz

    def compute_reward_and_info(self, obs: dict) -> tuple[int | float, dict[str, Any]]:
        """
        Compute reward and optional info payload from classifier callback.

        Returns:
            `(reward, reward_info)` where `reward_info` defaults to `{}`.
        """
        if self.reward_classifier_func is not None:
            out = self.reward_classifier_func(obs)
            if isinstance(out, tuple):
                rew = out[0]
                extra_info = out[1] if len(out) > 1 and isinstance(out[1], dict) else {}
                return rew, extra_info
            return out, {}
        return 0, {}

    def step(self, action):
        start_time = time.time()
        obs, rew, done, truncated, info = self.env.step(action)
        rew, reward_info = self.compute_reward_and_info(obs)
        rew = int(rew)
        done = done or bool(rew)
        info['succeed'] = bool(rew)
        info.update(reward_info)
        if self.target_hz is not None:
            time.sleep(max(0, 1/self.target_hz - (time.time() - start_time)))
            
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['succeed'] = False
        return obs, info
    
    
class MultiStageBinaryRewardClassifierWrapper(gym.Wrapper):
    """
    Chains multiple binary classifiers to gate multi-stage success.
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; robot-agnostic.
    """
    def __init__(self, env: Env, reward_classifier_func: List[callable]):
        super().__init__(env)
        self.reward_classifier_func = reward_classifier_func
        self.received = [False] * len(reward_classifier_func)
    
    def compute_reward(self, obs):
        rewards = [0] * len(self.reward_classifier_func)
        for i, classifier_func in enumerate(self.reward_classifier_func):
            if self.received[i]:
                continue

            logit = classifier_func(obs).item()
            if sigmoid(logit) >= 0.75:
                self.received[i] = True
                rewards[i] = 1

        reward = sum(rewards)
        return reward

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        rew = self.compute_reward(obs)
        done = (done or all(self.received)) # either environment done or all rewards satisfied
        info['succeed'] = all(self.received)
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.received = [False] * len(self.reward_classifier_func)
        info['succeed'] = False
        return obs, info


class Quat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; robot-agnostic.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["tcp_pose"]
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation


class Quat2R2Wrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to rotation matrix
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; assumes tcp_pose quaternion.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(9,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        r = R.from_quat(tcp_pose[3:]).as_matrix()
        observation["state"]["tcp_pose"] = np.concatenate(
            (tcp_pose[:3], r[..., :2].flatten())
        )
        return observation


class DualQuat2EulerWrapper(gym.ObservationWrapper):
    """
    Convert the quaternion representation of the tcp pose to euler angles
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; assumes dual tcp_pose quaternions.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        assert env.observation_space["state"]["left/tcp_pose"].shape == (7,)
        assert env.observation_space["state"]["right/tcp_pose"].shape == (7,)
        # from xyz + quat to xyz + euler
        self.observation_space["state"]["left/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )
        self.observation_space["state"]["right/tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(6,)
        )

    def observation(self, observation):
        # convert tcp pose from quat to euler
        tcp_pose = observation["state"]["left/tcp_pose"]
        observation["state"]["left/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        tcp_pose = observation["state"]["right/tcp_pose"]
        observation["state"]["right/tcp_pose"] = np.concatenate(
            (tcp_pose[:3], R.from_quat(tcp_pose[3:]).as_euler("xyz"))
        )
        return observation
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

class GripperCloseEnv(gym.ActionWrapper):
    """
    Drops the gripper dimension to enforce closed-gripper tasks.
    Use this wrapper to task that requires the gripper to be closed.
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; assumes 7D action with gripper.
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:6]
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    
class TeleopIntervention(gym.ActionWrapper):
    """
    Overrides actions with teleop input during interventions.

    This wrapper is device-agnostic and works with any TeleopAdapter implementation
    (e.g., SpaceMouseTeleop, JoyTeleopAdapter).

    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - replace SpaceMouseExpert with TeleopAdapter.get_teleop().
      - frame-aware teleop routing based on the TeleopAdapter-provided frame id.
    """
    def __init__(
        self,
        env,
        teleop_adapter: TeleopAdapter,
        action_indices=None,
        base_frame_id: str = "base",
        tcp_frame_id: str = "tcp",
    ):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        self.teleop_adapter = teleop_adapter
        self.left, self.right = False, False
        self.action_indices = action_indices
        self.base_frame_id = str(base_frame_id)
        self.tcp_frame_id = str(tcp_frame_id)
        if not self.base_frame_id or not self.tcp_frame_id:
            raise ValueError("TeleopIntervention requires non-empty base/tcp frame ids.")
        self._warned_unknown_frames: set[str] = set()

    def _resolve_command_frame(
        self,
        delta_xyz: np.ndarray,
        delta_rotvec: np.ndarray,
        input_frame_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map teleop deltas into the base frame using frame metadata.

        Args:
            delta_xyz: Translation delta from teleop input.
            delta_rotvec: Rotation delta from teleop input.
            input_frame_id: Frame id attached to the teleop sample.

        Returns:
            Tuple of (delta_xyz_base, delta_rotvec_base).
        """
        frame_id = str(input_frame_id).strip()

        # Temporary workaround: ROS2 joy_node publishes frame_id='joy'.
        # Treat it as TCP-frame teleop for now.
        if frame_id == "joy":
            frame_id = self.tcp_frame_id

        # Missing frame metadata defaults to TCP frame.
        if not frame_id:
            frame_id = self.tcp_frame_id

        if frame_id == self.base_frame_id:
            return delta_xyz, delta_rotvec

        # Unknown frames also default to TCP behavior for safety.
        if frame_id != self.tcp_frame_id:
            if frame_id not in self._warned_unknown_frames:
                LOGGER.warning(
                    "Unknown teleop frame '%s'; defaulting to tcp-frame transform.",
                    frame_id,
                )
                self._warned_unknown_frames.add(frame_id)
            frame_id = self.tcp_frame_id

        robot_env = self.env.unwrapped
        adapter = getattr(robot_env, "adapter", None)
        if adapter is None:
            return delta_xyz, delta_rotvec

        obs = adapter.get_observation()
        if obs is None:
            return delta_xyz, delta_rotvec

        state = obs.get("state", {})
        tcp_pose = state.get("tcp_pose", None)
        if tcp_pose is None or len(tcp_pose) < 7:
            return delta_xyz, delta_rotvec

        return transform_delta_to_base(
            delta_xyz,
            delta_rotvec,
            np.array(tcp_pose[3:], dtype=np.float32),
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: teleop action if nonzero; else, policy action
        """
        xyz, rpy, buttons, frame_id = self.teleop_adapter.get_teleop()
        delta_xyz = np.array(xyz, dtype=np.float32)
        delta_rpy = np.array(rpy, dtype=np.float32)
        delta_xyz, delta_rpy = self._resolve_command_frame(
            delta_xyz,
            delta_rpy,
            frame_id,
        )
        expert_a = np.array(list(delta_xyz) + list(delta_rpy), dtype=np.float32)
        if expert_a.shape[0] > 6:
            expert_a = expert_a[:6]
        buttons = list(buttons)
        while len(buttons) < 2:
            buttons.append(0)
        self.left, self.right = tuple(buttons[:2])
        intervened = False
        
        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if self.gripper_enabled:
            if self.left:  # close gripper
                gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right:  # open gripper
                gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                gripper_action = np.zeros((1,))
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if intervened:
            return expert_a, True

        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

class DualTeleopIntervention(gym.ActionWrapper):
    """
    Dual-arm intervention wrapper that injects teleop actions.

    This wrapper is device-agnostic and works with any TeleopAdapter implementation
    (e.g., SpaceMouseTeleop, JoyTeleopAdapter).

    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - replace SpaceMouseExpert with TeleopAdapter.get_teleop().
    """
    def __init__(
        self,
        env,
        teleop_left: TeleopAdapter,
        teleop_right: TeleopAdapter,
        action_indices=None,
        gripper_enabled=True,
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled

        self.teleop_left = teleop_left
        self.teleop_right = teleop_right
        self.left1, self.left2, self.right1, self.right2 = False, False, False, False
        self.action_indices = action_indices

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: teleop action if nonzero; else, policy action
        """
        intervened = False
        left_xyz, left_rpy, left_buttons, _left_frame_id = self.teleop_left.get_teleop()
        right_xyz, right_rpy, right_buttons, _right_frame_id = self.teleop_right.get_teleop()
        left_action = list(left_xyz + left_rpy)[:6]
        right_action = list(right_xyz + right_rpy)[:6]
        expert_a = np.array(left_action + right_action, dtype=np.float32)
        left_buttons = list(left_buttons)
        right_buttons = list(right_buttons)
        while len(left_buttons) < 2:
            left_buttons.append(0)
        while len(right_buttons) < 2:
            right_buttons.append(0)
        self.left1, self.left2 = tuple(left_buttons[:2])
        self.right1, self.right2 = tuple(right_buttons[:2])


        if self.gripper_enabled:
            if self.left1:  # close gripper
                left_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.left2:  # open gripper
                left_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                left_gripper_action = np.zeros((1,))

            if self.right1:  # close gripper
                right_gripper_action = np.random.uniform(-1, -0.9, size=(1,))
                intervened = True
            elif self.right2:  # open gripper
                right_gripper_action = np.random.uniform(0.9, 1, size=(1,))
                intervened = True
            else:
                right_gripper_action = np.zeros((1,))
            expert_a = np.concatenate(
                (expert_a[:6], left_gripper_action, expert_a[6:], right_gripper_action),
                axis=0,
            )

        if self.action_indices is not None:
            filtered_expert_a = np.zeros_like(expert_a)
            filtered_expert_a[self.action_indices] = expert_a[self.action_indices]
            expert_a = filtered_expert_a

        if np.linalg.norm(expert_a) > 0.001:
            intervened = True

        if intervened:
            return expert_a, True
        return action, False

    def step(self, action):

        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info["left1"] = self.left1
        info["left2"] = self.left2
        info["right1"] = self.right1
        info["right2"] = self.right2
        return obs, rew, done, truncated, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class GripperPenaltyWrapper(gym.RewardWrapper):
    """
    Penalizes frequent gripper toggling to stabilize training.
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; assumes 7D action with gripper index 6.
    """
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (7,)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos > 0.95) or (
            action[6] > 0.5 and self.last_gripper_pos < 0.95
        ):
            return reward - self.penalty
        else:
            return reward

    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        self.last_gripper_pos = observation["state"][0, 0]
        return observation, reward, terminated, truncated, info

class DualGripperPenaltyWrapper(gym.RewardWrapper):
    """
    Penalizes gripper toggling for both arms independently.
    Copy from original `serl_robot_infra/franka_env/envs/wrappers.py` with modifications:
      - No changes needed; assumes dual gripper indices 6 and 13.
    """
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        assert env.action_space.shape == (14,)
        self.penalty = penalty
        self.last_gripper_pos_left = 0 #TODO: this assume gripper starts opened
        self.last_gripper_pos_right = 0 #TODO: this assume gripper starts opened
    
    def reward(self, reward: float, action) -> float:
        if (action[6] < -0.5 and self.last_gripper_pos_left==0):
            reward -= self.penalty
            self.last_gripper_pos_left = 1
        elif (action[6] > 0.5 and self.last_gripper_pos_left==1):
            reward -= self.penalty
            self.last_gripper_pos_left = 0
        if (action[13] < -0.5 and self.last_gripper_pos_right==0):
            reward -= self.penalty
            self.last_gripper_pos_right = 1
        elif (action[13] > 0.5 and self.last_gripper_pos_right==1):
            reward -= self.penalty
            self.last_gripper_pos_right = 0
        return reward
    
    def step(self, action):
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        if "intervene_action" in info:
            action = info["intervene_action"]
        reward = self.reward(reward, action)
        return observation, reward, terminated, truncated, info
