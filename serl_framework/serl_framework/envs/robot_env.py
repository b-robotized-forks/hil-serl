"""
Robot-agnostic Gymnasium environment for HIL-SERL.

Origin: hil-serl/serl_robot_infra/franka_env/envs/franka_env.py
Modified: Yes
Changes:
  - Replaced HTTP/Flask requests with RobotAdapter calls.
  - Removed direct camera capture; images come from adapter snapshots.
  - Added configuration for DOF, image channel order, and observation freshness.
"""

import os
import time
import copy
import queue
import threading
from datetime import datetime
from typing import Any, Callable, Dict

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

from serl_framework.robot_adapter import RobotAdapter
from serl_framework.utils.config import apply_config_overrides, load_yaml_dict
from serl_framework.utils.motion import move_to_pose_interpolated
from serl_framework.utils.rotations import euler2quat
from serl_framework.utils.teleop import transform_translation_to_tip


class ImageDisplayer(threading.Thread):
    """
    Background thread that displays concatenated camera images for debugging.
    """

    def __init__(self, image_queue: queue.Queue, name: str) -> None:
        """
        Initialize the image display thread.

        Args:
            image_queue: Queue that provides image dictionaries for display.
            name: Window name for the OpenCV display.
        """
        super().__init__(daemon=True)
        self.queue = image_queue
        self.name = name

    def run(self) -> None:
        """
        Continuously read images from the queue and render them.
        """
        while True:
            img_array = self.queue.get()
            if img_array is None:
                break

            frame = np.concatenate(
                [cv2.resize(v, (128, 128)) for k, v in img_array.items() if "full" not in k],
                axis=1,
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


##############################################################################


ENV_CONFIG_ALIASES = {
    "CAMERAS": "CAMERAS",
    "cameras": "CAMERAS",
}

ENV_ARRAY_KEYS = {
    "TARGET_POSE",
    "GRASP_POSE",
    "REWARD_THRESHOLD",
    "ACTION_SCALE",
    "FREE_ACTION_SCALE",
    "RESET_POSE",
    "ABS_POSE_LIMIT_HIGH",
    "ABS_POSE_LIMIT_LOW",
    "REL_POSE_LIMIT_LOW",
    "REL_POSE_LIMIT_HIGH",
}

# Exponential moving-average factor for inter-step Hz reporting.
STEP_INTERCALL_EMA_ALPHA = 0.2


class DefaultEnvConfig:
    """
    Default configuration for RobotEnv.

    Attributes are intended to be overridden by task-specific configs.
    """

    ROBOT_DOF: int = 7
    # Control loop frequency in Hz (sleep duration in RobotEnv.step()).
    # Set to None to use the default.
    HZ: int = 10
    CAMERAS: Dict[str, Dict] = {}
    IMAGE_CROP: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    IMAGE_CHANNEL_ORDER: str = "bgr"
    OBSERVATION_MAX_AGE_S: float | None = 1.0
    # Max age of the state when the current state is queried after a command was sent.
    # If this is None, the time since dispatching the command will be used as max age.
    # float("inf"): disable this "freshness gate".
    # You should only set this if you want to allow older states than this.
    STATE_MAX_AGE_S: float | None = float("inf")
    VIDEO_PREFIX: str = "robot"
    ENABLE_KEYBOARD_LISTENER: bool = True
    GRIPPER_OPEN_THRESHOLD: float = 0.85

    TARGET_POSE: np.ndarray = np.zeros((6,))
    GRASP_POSE: np.ndarray = np.zeros((6,))
    REWARD_THRESHOLD: np.ndarray = np.zeros((6,))
    ACTION_SCALE = np.zeros((3,))
    FREE_ACTION_SCALE = None  # Falls back to ACTION_SCALE when not configured.
    RESET_POSE = np.zeros((6,))
    RANDOM_RESET = False
    RANDOM_XY_RANGE = (0.0,)
    RANDOM_RZ_RANGE = (0.0,)
    ABS_POSE_LIMIT_HIGH = np.zeros((6,))
    ABS_POSE_LIMIT_LOW = np.zeros((6,))

    # Optional relative workspace around the post-reset TCP pose.
    # 3-D (XYZ only) or 6-D (XYZ + RPY) offsets.
    #   - XYZ: metres, axis-aligned in base_link.
    #   - RPY: radians, relative to the post-reset TCP euler angles.
    # After each reset, the effective workspace is the intersection of the
    # absolute fence (ABS_POSE_LIMIT_*) and (post_reset_pose + these).
    # Set to None to use only absolute limits.
    REL_POSE_LIMIT_LOW: np.ndarray | None = None
    REL_POSE_LIMIT_HIGH: np.ndarray | None = None

    COMPLIANCE_PARAM: Dict[str, float] = {}
    RESET_PARAM: Dict[str, float] = {}
    PRECISION_PARAM: Dict[str, float] = {}
    # Action-magnitude boost: extra wrench when action is near saturation.
    # Translation only. Additive to persistent feedforward_wrench.
    ACTION_BOOST: Dict[str, Any] = {
        "enabled": False,
        "threshold": 0.9,
        "max_force": 2.0,
    }
    LOAD_PARAM: Dict[str, float] = {
        "mass": 0.0,
        "F_x_center_load": [0.0, 0.0, 0.0],
        "load_inertia": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
    PAYLOAD_PARAM: Dict[str, float] | None = None

    DISPLAY_IMAGE: bool = True
    GRIPPER_SLEEP: float = 0.6
    MAX_EPISODE_LENGTH: int = 100
    JOINT_RESET_PERIOD: int = 0

    # Timeouts for the optional ``reset_world`` service call in ``go_to_reset()``.
    # Override these in task configs that need a longer scene reset.
    RESET_WORLD_TIMEOUT_SEC: float = 0.5
    RESET_WORLD_SERVICE_TIMEOUT_SEC: float = 0.01
    # Extra step budget, in seconds, for reset interpolation moves.
    RESET_MOVE_TIMEOUT_SEC: float = 1.0

    @classmethod
    def from_yaml(cls, path: str) -> "DefaultEnvConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to a YAML config file.

        Returns:
            Instance of DefaultEnvConfig with overrides applied.
        """
        data = load_yaml_dict(path)
        config = cls()
        apply_config_overrides(
            config=config,
            overrides=data,
            aliases=ENV_CONFIG_ALIASES,
            array_keys=ENV_ARRAY_KEYS,
        )
        return config


##############################################################################


class RobotEnv(gym.Env):
    """
    Robot-agnostic Gymnasium environment backed by a RobotAdapter.

    The adapter supplies synchronized state and image snapshots. This env converts
    them into the HIL-SERL observation format and issues pose/gripper commands.
    """

    @staticmethod
    def _validate_rel_pose_limit(
        value: np.ndarray | None,
        *,
        name: str,
    ) -> np.ndarray | None:
        """Validate one relative workspace limit vector (3-D or 6-D).

        Accepts either a 3-element XYZ offset or a 6-element XYZ+RPY offset.
        """
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float64)
        if arr.shape not in ((3,), (6,)):
            raise ValueError(
                f"{name} must be a 3-element (XYZ) or 6-element (XYZ+RPY) "
                f"offset vector, got shape {arr.shape}."
            )
        return arr

    @staticmethod
    def _clip_angle_to_interval(
        angle: float,
        low: float,
        high: float,
    ) -> float:
        """Clip one Euler angle into a base-frame interval with wrap handling.

        The interval is interpreted in base-frame Euler coordinates.  The input
        angle is first wrapped to the representation nearest the interval
        centre, then clipped into ``[low, high]``.
        """
        center = 0.5 * (low + high)
        wrapped = center + ((angle - center + np.pi) % (2.0 * np.pi) - np.pi)
        return float(np.clip(wrapped, low, high))

    def __init__(
        self,
        hz: int | None = None,
        fake_env: bool = False,
        save_video: bool = False,
        config: DefaultEnvConfig | str | None = None,
        set_load: bool = False,
        adapter: RobotAdapter | None = None,
    ) -> None:
        """
        Initialize the environment.

        Args:
            hz: Control frequency in Hz for the step loop. If None, uses config.HZ.
            fake_env: If True, skip adapter interaction and hardware calls.
            save_video: If True, record cropped image frames to disk on reset.
            config: Environment configuration object.
            set_load: If True, apply payload parameters at startup.
            adapter: RobotAdapter instance that provides observations and commands.
                Required when fake_env is False.
        """
        if isinstance(config, str):
            config = DefaultEnvConfig.from_yaml(config)
        if config is None:
            config = DefaultEnvConfig()

        self.adapter = adapter
        self.config = config
        self.hz = config.HZ if hz is None else hz
        self.action_scale = config.ACTION_SCALE
        # Resolve free-space action scale once: None falls back to the task scale.
        free = getattr(config, "FREE_ACTION_SCALE", None)
        self._free_action_scale = free if free is not None else config.ACTION_SCALE
        self._task_action_scale = config.ACTION_SCALE
        self._TARGET_POSE = config.TARGET_POSE
        self._fixed_reset_pose_euler = self.get_fixed_reset_pose_euler()
        # Backwards-compatible alias for older task envs that still read this
        # attribute directly.
        self._RESET_POSE = self._fixed_reset_pose_euler
        self._REWARD_THRESHOLD = config.REWARD_THRESHOLD
        self.max_episode_length = config.MAX_EPISODE_LENGTH
        self.display_image = config.DISPLAY_IMAGE
        self.gripper_sleep = config.GRIPPER_SLEEP
        self.randomreset = config.RANDOM_RESET
        self.random_xy_range = config.RANDOM_XY_RANGE
        self.random_rz_range = config.RANDOM_RZ_RANGE
        self.joint_reset_cycle = config.JOINT_RESET_PERIOD
        self.robot_dof = config.ROBOT_DOF
        self.observation_max_age_s = config.OBSERVATION_MAX_AGE_S
        self.state_max_age_s = config.STATE_MAX_AGE_S
        self.video_prefix = config.VIDEO_PREFIX
        self.enable_keyboard_listener = config.ENABLE_KEYBOARD_LISTENER
        self.gripper_open_threshold = config.GRIPPER_OPEN_THRESHOLD
        self.image_channel_order = config.IMAGE_CHANNEL_ORDER.lower()
        self.expected_image_keys = self._camera_keys(config.CAMERAS)
        self.save_video = save_video

        self.curr_path_length = 0
        self.cycle_count = 0
        self.last_gripper_act = time.time()
        self._latest_images_raw: Dict[str, np.ndarray] | None = None

        # Step-to-step timing tracker (raw + exponentially smoothed).
        self._step_intercall_ema_alpha = STEP_INTERCALL_EMA_ALPHA
        self._last_step_call_s: float | None = None
        self._step_intercall_ema_s: float | None = None

        # Fixed reset pose is optional for envs that fully override go_to_reset().
        if self._fixed_reset_pose_euler is None:
            self.resetpos = None
        else:
            # convert last 3 elements from euler to quat, from size (6,) to (7,)
            self.resetpos = np.concatenate(
                [
                    self._fixed_reset_pose_euler[:3],
                    euler2quat(self._fixed_reset_pose_euler[3:]),
                ]
            )

        # Absolute safety fence (always-on outer boundary).
        self.xyz_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[:3],
            config.ABS_POSE_LIMIT_HIGH[:3],
            dtype=np.float64,
        )
        self.rpy_bounding_box = gym.spaces.Box(
            config.ABS_POSE_LIMIT_LOW[3:],
            config.ABS_POSE_LIMIT_HIGH[3:],
            dtype=np.float64,
        )

        # Optional relative workspace around post-reset TCP position.
        self.rel_pose_limit_low = self._validate_rel_pose_limit(
            config.REL_POSE_LIMIT_LOW,
            name="REL_POSE_LIMIT_LOW",
        )
        self.rel_pose_limit_high = self._validate_rel_pose_limit(
            config.REL_POSE_LIMIT_HIGH,
            name="REL_POSE_LIMIT_HIGH",
        )
        if (self.rel_pose_limit_low is None) != (self.rel_pose_limit_high is None):
            raise ValueError(
                "REL_POSE_LIMIT_LOW and REL_POSE_LIMIT_HIGH must either both be set "
                "or both be None."
            )
        if (
            self.rel_pose_limit_low is not None
            and self.rel_pose_limit_high is not None
        ):
            if self.rel_pose_limit_low.shape != self.rel_pose_limit_high.shape:
                raise ValueError(
                    "REL_POSE_LIMIT_LOW and REL_POSE_LIMIT_HIGH must have the same "
                    f"shape, got {self.rel_pose_limit_low.shape} and "
                    f"{self.rel_pose_limit_high.shape}."
                )
            if not np.all(self.rel_pose_limit_low < self.rel_pose_limit_high):
                dims = "XYZ axes" if self.rel_pose_limit_low.shape[0] == 3 else "XYZ+RPY axes"
                raise ValueError(
                    "REL_POSE_LIMIT_LOW must be strictly less than "
                    f"REL_POSE_LIMIT_HIGH on all {dims}."
                )
        # Effective workspace used by clip_safety_box().
        # Starts as the absolute fence; narrowed after reset if relative
        # limits are configured.
        self._effective_xyz_low = np.array(self.xyz_bounding_box.low, copy=True)
        self._effective_xyz_high = np.array(self.xyz_bounding_box.high, copy=True)
        self._effective_rpy_low = np.array(self.rpy_bounding_box.low, copy=True)
        self._effective_rpy_high = np.array(self.rpy_bounding_box.high, copy=True)

        # Action/Observation Space
        self.action_space = gym.spaces.Box(
            np.ones((7,), dtype=np.float32) * -1,
            np.ones((7,), dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action_delta_debug: Dict[str, float] | None = None

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(-np.inf, np.inf, shape=(7,)),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "gripper_pose": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "q": gym.spaces.Box(-np.inf, np.inf, shape=(self.robot_dof,)),
                        "dq": gym.spaces.Box(-np.inf, np.inf, shape=(self.robot_dof,)),
                    }
                ),
                "images": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for key in self.expected_image_keys
                    }
                ),
            }
        )

        if fake_env:
            if self.save_video:
                self.recording_frames = []
            self.terminate = False
            self._initialize_fake_state()
            return

        if self.adapter is None:
            raise ValueError("adapter is required when fake_env is False")

        if self.save_video:
            print("Saving videos!")
            self.recording_frames = []

        self._update_currpos()

        if self.display_image and self.expected_image_keys:
            self.img_queue = queue.Queue()
            self.displayer = ImageDisplayer(self.img_queue, self.video_prefix)
            self.displayer.start()

        if set_load:
            self._set_payload_interactive()

        self.terminate = False
        if self.enable_keyboard_listener:
            from pynput import keyboard
            self.terminate = False
            def on_press(key):
                if key == keyboard.Key.esc:
                    self.terminate = True
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()

        print("Initialized RobotEnv")

    def get_fixed_reset_pose_euler(self) -> np.ndarray | None:
        """Return the env's fixed reset pose, or ``None`` if not configured."""
        pose = getattr(self.config, "RESET_POSE", None)
        if pose is None:
            return None
        return np.asarray(pose, dtype=np.float64)

    def _require_fixed_reset_pose(self, context: str) -> np.ndarray:
        """Return the configured fixed reset pose or raise a clear error."""
        if self._fixed_reset_pose_euler is None or self.resetpos is None:
            raise RuntimeError(
                f"{context} requires config.RESET_POSE, but this env does not "
                "define a fixed reset pose."
            )
        return self._fixed_reset_pose_euler

    def now(self) -> float:
        """Return the current time in seconds from the adapter's clock.

        Uses the adapter's clock (sim-time when available). Falls back to
        wall-clock ``time.time()`` when no adapter is set (e.g. fake_env).
        """
        if self.adapter is not None:
            return self.adapter.now()
        return time.time()

    def _update_effective_workspace(self, reference_tcp_pose: np.ndarray) -> None:
        """Recompute the effective workspace around a reference pose.

        If ``REL_POSE_LIMIT_LOW/HIGH`` are configured, the effective
        workspace is the intersection of the absolute safety fence and
        ``reference_pose + rel_limits``.  The relative limits are in
        base-frame coordinates (XYZ metres, RPY radians).

        The relative limits may be 3-D (XYZ only — RPY stays absolute)
        or 6-D (XYZ + RPY).

        Called after each reset once the post-reset TCP pose is known.
        """
        abs_xyz_low = np.array(self.xyz_bounding_box.low)
        abs_xyz_high = np.array(self.xyz_bounding_box.high)
        abs_rpy_low = np.array(self.rpy_bounding_box.low)
        abs_rpy_high = np.array(self.rpy_bounding_box.high)

        if self.rel_pose_limit_low is None or self.rel_pose_limit_high is None:
            self._effective_xyz_low = abs_xyz_low.copy()
            self._effective_xyz_high = abs_xyz_high.copy()
            self._effective_rpy_low = abs_rpy_low.copy()
            self._effective_rpy_high = abs_rpy_high.copy()
            return

        # --- XYZ (always present in 3-D or 6-D limits) ---
        ref_xyz = np.asarray(reference_tcp_pose[:3], dtype=np.float64)
        cand_xyz_low = ref_xyz + self.rel_pose_limit_low[:3]
        cand_xyz_high = ref_xyz + self.rel_pose_limit_high[:3]
        eff_xyz_low = np.maximum(abs_xyz_low, cand_xyz_low)
        eff_xyz_high = np.minimum(abs_xyz_high, cand_xyz_high)

        if np.all(eff_xyz_low < eff_xyz_high):
            self._effective_xyz_low = eff_xyz_low
            self._effective_xyz_high = eff_xyz_high
        else:
            print(
                "WARNING: relative XYZ workspace does not intersect with "
                f"absolute safety fence. ref_xyz={ref_xyz}, "
                f"rel_low={self.rel_pose_limit_low[:3]}, "
                f"rel_high={self.rel_pose_limit_high[:3]}. "
                "Falling back to absolute XYZ limits."
            )
            self._effective_xyz_low = abs_xyz_low.copy()
            self._effective_xyz_high = abs_xyz_high.copy()

        # --- RPY (only if 6-D limits provided) ---
        if self.rel_pose_limit_low.shape[0] >= 6:
            ref_rpy = Rotation.from_quat(reference_tcp_pose[3:]).as_euler("xyz")
            cand_rpy_low = ref_rpy + self.rel_pose_limit_low[3:]
            cand_rpy_high = ref_rpy + self.rel_pose_limit_high[3:]
            eff_rpy_low = np.maximum(abs_rpy_low, cand_rpy_low)
            eff_rpy_high = np.minimum(abs_rpy_high, cand_rpy_high)

            if np.all(eff_rpy_low < eff_rpy_high):
                self._effective_rpy_low = eff_rpy_low
                self._effective_rpy_high = eff_rpy_high
            else:
                print(
                    "WARNING: relative RPY workspace does not intersect with "
                    f"absolute safety fence. ref_rpy={ref_rpy}, "
                    f"rel_low={self.rel_pose_limit_low[3:]}, "
                    f"rel_high={self.rel_pose_limit_high[3:]}. "
                    "Falling back to absolute RPY limits."
                )
                self._effective_rpy_low = abs_rpy_low.copy()
                self._effective_rpy_high = abs_rpy_high.copy()
        else:
            self._effective_rpy_low = abs_rpy_low.copy()
            self._effective_rpy_high = abs_rpy_high.copy()

    def clip_safety_box(self, pose: np.ndarray) -> np.ndarray:
        """
        Clip the pose to be within the effective workspace.

        Both translation and rotation are clipped to the effective limits
        (intersection of absolute fence and optional relative workspace
        around the post-reset TCP pose).

        Args:
            pose: 7D pose array [x, y, z, qx, qy, qz, qw].

        Returns:
            Clipped pose array within bounds.
        """
        pose[:3] = np.clip(
            pose[:3], self._effective_xyz_low, self._effective_xyz_high
        )
        euler = Rotation.from_quat(pose[3:]).as_euler("xyz")
        euler = np.array(
            [
                self._clip_angle_to_interval(
                    angle,
                    low,
                    high,
                )
                for angle, low, high in zip(
                    euler,
                    self._effective_rpy_low,
                    self._effective_rpy_high,
                )
            ],
            dtype=np.float64,
        )
        pose[3:] = Rotation.from_euler("xyz", euler).as_quat()

        return pose

    def step(self, action: np.ndarray) -> tuple:
        """
        Execute a single control step in the environment.

        Args:
            action: 7D action vector (xyz delta, rot delta, gripper).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # Pace using inter-call timing: if caller invokes step() too quickly,
        # sleep the remaining time to maintain target frequency.
        # Uses the adapter clock (sim-time when available) so pacing stays
        # correct even when the simulator runs slower than real-time.
        step_call_s = self.now()
        target_period_s = 1.0 / float(self.hz)
        inter_call_dt = None
        if self._last_step_call_s is not None:
            inter_call_dt = step_call_s - self._last_step_call_s
            if inter_call_dt < target_period_s:
                # Poll the sim clock in small wall-time increments so the
                # pacing is correct regardless of the real-time factor.
                target_time = self._last_step_call_s + target_period_s
                while self.now() < target_time:
                    time.sleep(0.001)
            # Measure the effective inter-step period after pacing sleep.
            step_call_s = self.now()
            inter_call_dt = step_call_s - self._last_step_call_s
            if self._step_intercall_ema_s is None:
                self._step_intercall_ema_s = inter_call_dt
            else:
                alpha = self._step_intercall_ema_alpha
                self._step_intercall_ema_s = (
                    (1.0 - alpha) * self._step_intercall_ema_s + alpha * inter_call_dt
                )
        self._last_step_call_s = step_call_s

        start_time = self.now()
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        self.last_action_delta_debug = {
            "action_norm": float(np.linalg.norm(action[:6])),
            "trans_m": float(np.linalg.norm(action[:3] * float(self.action_scale[0]))),
            "rot_rad": float(np.linalg.norm(action[3:6] * float(self.action_scale[1]))),
        }
        xyz_delta = action[:3]

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta * self.action_scale[0]

        # GET ORIENTATION FROM ACTION
        self.nextpos[3:] = (
            Rotation.from_rotvec(action[3:6] * self.action_scale[1])
            * Rotation.from_quat(self.currpos[3:])
        ).as_quat()

        gripper_action = action[6] * self.action_scale[2]

        self._send_gripper_command(gripper_action)

        # Action-magnitude boost: extra wrench when translation action is near saturation.
        boost_cfg = getattr(self.config, "ACTION_BOOST", {})
        if boost_cfg.get("enabled", False) and self.adapter is not None:
            from serl_framework.utils.config import compute_action_boost
            action_tip = transform_translation_to_tip(action[:3], self.currpos[3:])
            self.adapter.set_step_wrench_boost(
                compute_action_boost(action_tip.tolist(), boost_cfg)
            )

        self._send_pos_command(self.clip_safety_box(self.nextpos))

        self.curr_path_length += 1
        elapsed = self.now() - start_time
        max_age = self.state_max_age_s
        if max_age is None:
            max_age = elapsed
        if not self._update_currpos(max_in_the_past_s=max_age):
            # Truncated indicates episode ended due to a time limit or external interruption,
            # not because the task reached a terminal success/failure condition.
            return {}, 0, False, True, {
                "timeout": True,
                "action_delta_debug": copy.deepcopy(self.last_action_delta_debug),
            }

        ob = self._get_obs()
        reward = self.compute_reward(ob)
        if reward:
            print("\n======== SUCCESS ============")
        done = self.curr_path_length >= self.max_episode_length or reward or self.terminate
        return ob, int(reward), done, False, {
            "succeed": reward,
            "action_delta_debug": copy.deepcopy(self.last_action_delta_debug),
        }

    def compute_reward(self, obs: Dict[str, Dict[str, np.ndarray]]) -> bool:
        """
        Compute a sparse success reward based on pose distance.

        Args:
            obs: Observation dictionary containing current state.

        Returns:
            True if the target pose is reached within thresholds.
        """
        current_pose = obs["state"]["tcp_pose"]
        current_rot = Rotation.from_quat(current_pose[3:]).as_matrix()
        target_rot = Rotation.from_euler("xyz", self._TARGET_POSE[3:]).as_matrix()
        diff_rot = current_rot.T @ target_rot
        diff_euler = Rotation.from_matrix(diff_rot).as_euler("xyz")
        delta = np.abs(np.hstack([current_pose[:3] - self._TARGET_POSE[:3], diff_euler]))
        return bool(np.all(delta < self._REWARD_THRESHOLD))

    def get_im(self) -> Dict[str, np.ndarray]:
        """
        Build resized observation images from the latest adapter snapshot.

        Returns:
            Dict mapping camera names to uint8 RGB images.
        """
        if self._latest_images_raw is None:
            raise RuntimeError("No images available; call _update_currpos first.")

        images: Dict[str, np.ndarray] = {}
        display_images: Dict[str, np.ndarray] = {}
        full_res_images: Dict[str, np.ndarray] = {}

        missing_keys = [key for key in self.expected_image_keys if key not in self._latest_images_raw]
        if missing_keys:
            raise RuntimeError(f"Missing images for cameras: {missing_keys}")

        for key in self.expected_image_keys:
            raw = self._latest_images_raw[key]
            cropped = self.config.IMAGE_CROP[key](raw) if key in self.config.IMAGE_CROP else raw
            target_shape = self.observation_space["images"][key].shape[:2][::-1]
            resized = cv2.resize(cropped, target_shape)

            obs_image, display_image = self._convert_image_channels(resized)
            images[key] = obs_image
            display_images[key] = display_image
            display_images[key + "_full"] = self._convert_image_channels(cropped)[1]
            full_res_images[key] = self._convert_image_channels(cropped)[1].copy()

        # Store full resolution cropped images separately
        if self.save_video:
            self.recording_frames.append(full_res_images)

        if self.display_image and display_images:
            self.img_queue.put(display_images)
        return images

    def interpolate_move(
        self,
        goal: np.ndarray,
        timeout: float,
        in_euler: bool,
        pos_tol_m: float | None = None,
        rot_tol_rad: float | None = None,
    ) -> None:
        """
        Closed-loop linear point-to-point move ("carrot on a stick").

        Each cycle reads the actual TCP, places a target one delta ahead
        along the current-to-goal direction, and sends it.  Pacing uses
        the adapter clock (sim-time when available) so motion speed is
        consistent in the simulation world regardless of the real-time factor.

        The delta is the Euclidean norm of the translational action scale,
        i.e. the maximum 3-D displacement one policy step can produce.

        Args:
            goal: Target pose as 6D euler or 7D quaternion array (depending on setting of in_euler).
            timeout: Time budget in adapter-clock seconds.
            pos_tol_m: Optional translational convergence tolerance override
                (metres). ``None`` keeps the default
                ``max(0.001, min(0.003, 0.5 * delta_m))``.
            rot_tol_rad: Optional rotational convergence tolerance override
                (radians). ``None`` keeps the default
                ``max(0.01, min(0.03, 0.5 * delta_rad))``.
        """
        def read_current_pose() -> np.ndarray:
            if not self._update_currpos():
                raise RuntimeError(
                    "ERROR: could not interpolate move because updating the current pos failed."
                )
            return np.array(self.currpos, dtype=np.float32)

        def send_pose_command(pose: np.ndarray) -> None:
            self._send_pos_command(pose)
            self.nextpos = np.array(pose, copy=True)

        result = move_to_pose_interpolated(
            goal=goal,
            timeout=timeout,
            in_euler=in_euler,
            delta_m=float(self.action_scale[0]) * np.sqrt(3),
            delta_rad=float(self.action_scale[1]) * np.sqrt(3),
            hz=float(self.hz),
            read_current_pose=read_current_pose,
            send_pose_command=send_pose_command,
            clip_pose=lambda pose: self.clip_safety_box(np.array(pose, dtype=np.float32)),
            now=self.now,
            sleep=time.sleep,
            pos_tol_m=pos_tol_m,
            rot_tol_rad=rot_tol_rad,
        )
        self.nextpos = result.goal_pose.copy()
        if result.reached:
            return
        print(
            "WARNING: interpolate_move did not reach the goal pose within the allowed step budget; "
            f"continuing. pos_error={result.pos_error_m:.4f}m rot_error={result.rot_error_rad:.4f}rad"
        )

    def _pose_error_norms(
        self,
        current_pose: np.ndarray,
        target_pose: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute translational and rotational pose error magnitudes.

        Args:
            current_pose: Current pose [x, y, z, qx, qy, qz, qw].
            target_pose: Target pose [x, y, z, qx, qy, qz, qw].

        Returns:
            Tuple `(pos_error_m, rot_error_rad)`.
        """
        pos_error_m = float(np.linalg.norm(target_pose[:3] - current_pose[:3]))
        r_curr = Rotation.from_quat(current_pose[3:])
        r_tgt = Rotation.from_quat(target_pose[3:])
        rot_error_rad = float((r_curr.inv() * r_tgt).magnitude())
        return pos_error_m, rot_error_rad

    def _wait_until_pose_reached(
        self,
        target_pose: np.ndarray,
        timeout_s: float,
        pos_tol_m: float,
        rot_tol_rad: float,
    ) -> bool:
        """
        Wait until the current pose reaches a target pose within tolerances.

        Args:
            target_pose: Waypoint pose [x, y, z, qx, qy, qz, qw].
            timeout_s: Maximum waiting time in seconds.
            pos_tol_m: Position tolerance in meters.
            rot_tol_rad: Rotation tolerance in radians.

        Returns:
            True if reached within timeout, False otherwise.
        """
        deadline = self.now() + max(timeout_s, 1.0 / self.hz)
        while self.now() < deadline:
            if self._update_currpos():
                pos_error_m, rot_error_rad = self._pose_error_norms(
                    current_pose=self.currpos,
                    target_pose=target_pose,
                )
                if pos_error_m <= pos_tol_m and rot_error_rad <= rot_tol_rad:
                    return True
            time.sleep(1.0 / self.hz)
        return False

    def go_to_reset(self, joint_reset: bool = False) -> None:
        """
        Execute the reset routine to move the robot to its reset pose.

        The concrete steps to perform reset should be
        implemented each subclass for the specific task.
        Should override this method if custom reset procedure is needed.

        Args:
            joint_reset: If True, request a joint reset via the adapter.
        """
        if not self._update_currpos():
            raise RuntimeError("ERROR: could not go to reset because updating the current pos failed.")

        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        self._set_freespace_params()
        time.sleep(0.5)

        # Optional integration hook: reset external world objects (e.g. simulation scene).
        # If no listener/service is running, this call is intentionally silent.
        self._call_optional_service(
            "reset_world",
            timeout_sec=self.config.RESET_WORLD_TIMEOUT_SEC,
            service_timeout_sec=self.config.RESET_WORLD_SERVICE_TIMEOUT_SEC,
        )

        # Perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            self._call_service("reset_robot")
            time.sleep(6.0)
        else:
            print("SOFT RESET")

        if self.randomreset:
            fixed_reset_pose = self._require_fixed_reset_pose("RobotEnv.go_to_reset(random_reset)")
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = fixed_reset_pose[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler2quat(euler_random)
            self.interpolate_move(
                reset_pose,
                timeout=float(getattr(self.config, "RESET_MOVE_TIMEOUT_SEC", 1.0)),
                in_euler=False,
            )
        else:
            self._require_fixed_reset_pose("RobotEnv.go_to_reset")
            reset_pose = self.resetpos.copy()
            self.interpolate_move(
                reset_pose,
                timeout=float(getattr(self.config, "RESET_MOVE_TIMEOUT_SEC", 1.0)),
                in_euler=False,
            )

        # Reset teleop-side helpers after the robot reached the episode start
        # pose so scripted intervention adapters begin each episode fresh.
        teleop_adapter = getattr(self, "_teleop_adapter", None)
        if teleop_adapter is not None and hasattr(teleop_adapter, "reset"):
            teleop_adapter.reset()

        # Optional integration hook: notify teleop-side helpers that reset completed.
        # If no listener/service is running, this call is intentionally silent.
        self._call_optional_service(
            "reset_signal",
            timeout_sec=0.2,
            service_timeout_sec=0.01,
        )

        self._set_task_params()

    def reset(self, joint_reset: bool = False, **kwargs) -> tuple:
        """
        Reset the environment and return the initial observation.

        Args:
            joint_reset: If True, perform a joint reset during the reset routine.
            **kwargs: Unused kwargs for Gym compatibility.

        Returns:
            Tuple of (obs, info).
        """
        self.last_gripper_act = time.time()
        self._set_task_params()
        if self.save_video:
            self.save_video_recording()

        self.cycle_count += 1
        if self.joint_reset_cycle != 0 and self.cycle_count % self.joint_reset_cycle == 0:
            self.cycle_count = 0
            joint_reset = True

        # Reset effective workspace to absolute fence before go_to_reset().
        # The robot may traverse the full workspace during the reset move.
        self._effective_xyz_low = np.array(self.xyz_bounding_box.low, copy=True)
        self._effective_xyz_high = np.array(self.xyz_bounding_box.high, copy=True)
        self._effective_rpy_low = np.array(self.rpy_bounding_box.low, copy=True)
        self._effective_rpy_high = np.array(self.rpy_bounding_box.high, copy=True)

        self._recover()
        self.go_to_reset(joint_reset=joint_reset)
        self._recover()
        self.curr_path_length = 0

        if not self._update_currpos():
            raise RuntimeError("ERROR: could not reset because updating the current pos failed.")

        # Narrow the effective workspace around the actual post-reset TCP
        # pose (if relative limits are configured).
        self._update_effective_workspace(self.currpos)

        obs = self._get_obs()
        self.terminate = False
        return obs, {"succeed": False}

    def save_video_recording(self) -> None:
        """
        Persist recorded image frames to mp4 files.
        """
        try:
            if self.recording_frames:
                if not os.path.exists('./videos'):
                    os.makedirs('./videos')

                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                for camera_key in self.recording_frames[0].keys():
                    video_path = f"./videos/{self.video_prefix}_{camera_key}_{timestamp}.mp4"
                    # Get the shape of the first frame for this camera
                    first_frame = self.recording_frames[0][camera_key]
                    height, width = first_frame.shape[:2]

                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        10,
                        (width, height),
                    )

                    for frame_dict in self.recording_frames:
                        video_writer.write(frame_dict[camera_key])

                    video_writer.release()
                    print(f"Saved video for camera {camera_key} at {video_path}")

            self.recording_frames.clear()
        except Exception as e:
            print(f"Failed to save video: {e}")

    def _recover(self) -> None:
        """Recover the robot from error state."""
        self._call_service("clear_error")

    def _send_pos_command(self, pos: np.ndarray) -> None:
        """
        Send an absolute pose command to the robot.

        Args:
            pos: 7D pose array [x, y, z, qx, qy, qz, qw].
        """
        self._recover()
        arr = np.array(pos).astype(np.float32)
        self.adapter.send_pose_command(arr.tolist())

    def _send_gripper_command(self, pos: float, mode: str = "binary") -> None:
        """
        Issue gripper commands based on the action input.

        Args:
            pos: Scalar gripper action value.
            mode: Control mode; currently supports "binary" only.
        """
        if mode == "binary":
            gripper_value = float(self.curr_gripper_pos[0])
            now = time.time()
            if (
                (pos <= -0.5)
                and (gripper_value > self.gripper_open_threshold)
                and (now - self.last_gripper_act > self.gripper_sleep)
            ):
                self._call_service("set_gripper", mode=1)
                self.last_gripper_act = now
                time.sleep(self.gripper_sleep)
            elif (
                (pos >= 0.5)
                and (gripper_value < self.gripper_open_threshold)
                and (now - self.last_gripper_act > self.gripper_sleep)
            ):
                self._call_service("set_gripper", mode=0)
                self.last_gripper_act = now
                time.sleep(self.gripper_sleep)
        elif mode == "continuous":
            raise NotImplementedError("Continuous gripper control is optional")

    def _update_currpos(self, max_in_the_past_s: float | None = None) -> bool:
        """
        Refresh the cached state and images from the adapter snapshot.

        Args:
            max_in_the_past_s (float): freshness gate - only a state max this old will be accepted.
                Can be used to override instance settings;
                If None, instance settings will be used.
                Can be float('inf') to effectively disable. 
        Returns:
            True if fresh state was received, False on timeout.
        """
        if max_in_the_past_s is None:
            # use instance-wide default
            max_in_the_past_s = self.state_max_age_s
        if max_in_the_past_s == float('inf'):
            # default is "disable": use `None` (RobotAdapter uses None to disable)
            max_in_the_past_s = None
        try:
            self.adapter.wait_for_state(timeout_s=5, max_in_the_past_s=max_in_the_past_s)
        except TimeoutError as exc:
            print(f"WARNING: The robot state has not arrived. {exc}")
            return False
        snapshot = self.adapter.get_observation(max_age_s=self.observation_max_age_s)
        if snapshot is None or snapshot.get("state") is None:
            raise RuntimeError("No robot state available from adapter.")

        state = snapshot["state"]
        self.currpos = np.array(state.get("tcp_pose", np.zeros((7,))), dtype=np.float32)
        self.currvel = np.array(state.get("tcp_vel", np.zeros((6,))), dtype=np.float32)
        self.currforce = np.array(state.get("tcp_force", np.zeros((3,))), dtype=np.float32)
        self.currtorque = np.array(state.get("tcp_torque", np.zeros((3,))), dtype=np.float32)

        gripper_pose = state.get("gripper_pose", 0.0)
        if isinstance(gripper_pose, (list, tuple, np.ndarray)):
            gripper_value = float(gripper_pose[0]) if gripper_pose else 0.0
        else:
            gripper_value = float(gripper_pose)
        self.curr_gripper_pos = np.array([gripper_value], dtype=np.float32)

        self.q = np.array(state.get("q", np.zeros((self.robot_dof,))), dtype=np.float32)
        self.dq = np.array(state.get("dq", np.zeros((self.robot_dof,))), dtype=np.float32)
        if self.q.shape[0] != self.robot_dof or self.dq.shape[0] != self.robot_dof:
            raise ValueError(
                f"Expected q/dq length {self.robot_dof}, got {self.q.shape[0]} and {self.dq.shape[0]}"
            )

        images = snapshot.get("images")
        self._latest_images_raw = images if images is not None else {}
        return True

    def update_currpos(self) -> None:
        """
        Compatibility wrapper for updating robot state.
        """
        if not self._update_currpos():
            raise RuntimeError("Failed to retrieve the current robot state.")

    def _get_obs(self) -> dict:
        """
        Compose the observation dict from cached state and images.

        Returns:
            Observation dict with "images" and "state" entries.
        """
        images = self.get_im()
        state_observation = {
            "tcp_pose": self.currpos,
            "tcp_vel": self.currvel,
            "gripper_pose": self.curr_gripper_pos,
            "tcp_force": self.currforce,
            "tcp_torque": self.currtorque,
            "q": self.q,
            "dq": self.dq,
        }
        return copy.deepcopy(dict(images=images, state=state_observation))

    def close(self) -> None:
        """
        Release resources associated with the environment.
        """
        if hasattr(self, "listener"):
            self.listener.stop()
        if self.display_image and hasattr(self, "img_queue"):
            self.img_queue.put(None)
            cv2.destroyAllWindows()
            self.displayer.join()

    def _initialize_fake_state(self) -> None:
        """
        Initialize placeholder state for fake environments.
        """
        self.currpos = np.zeros((7,), dtype=np.float32)
        self.currvel = np.zeros((6,), dtype=np.float32)
        self.currforce = np.zeros((3,), dtype=np.float32)
        self.currtorque = np.zeros((3,), dtype=np.float32)
        self.q = np.zeros((self.robot_dof,), dtype=np.float32)
        self.dq = np.zeros((self.robot_dof,), dtype=np.float32)
        self.curr_gripper_pos = np.zeros((1,), dtype=np.float32)
        self._latest_images_raw = {}

    def _camera_keys(self, cameras: Dict[str, Dict]) -> list:
        """
        Resolve the expected camera keys from config.

        Args:
            cameras: Mapping of camera names to configuration dicts.

        Returns:
            List of camera names.
        """
        return list(cameras.keys())

    def _convert_image_channels(self, image: np.ndarray) -> tuple:
        """
        Convert image channel order to RGB for observations and BGR for display.

        Args:
            image: Input image array in the configured channel order.

        Returns:
            Tuple of (obs_image_rgb, display_image_bgr).
        """
        if self.image_channel_order == "bgr":
            return image[..., ::-1], image
        if self.image_channel_order == "rgb":
            return image, image[..., ::-1]
        raise ValueError(f"Unsupported IMAGE_CHANNEL_ORDER: {self.image_channel_order}")

    def _call_service(self, service_name: str, **kwargs) -> Dict[str, object]:
        """
        Call a named service on the adapter.

        Args:
            service_name: Service identifier.
            **kwargs: Service arguments.

        Returns:
            Adapter response dict with "ok" and "message" fields.
        """
        response = self.adapter.call_service(service_name, **kwargs)
        if not isinstance(response, dict):
            print(f"WARNING: service '{service_name}' returned non-dict response: {response}")
            return response
        ok = response.get("ok", None)
        if ok is False:
            message = response.get("message", "")
            print(f"WARNING: service '{service_name}' failed: {message}")
        elif ok is None:
            print(f"WARNING: service '{service_name}' response missing 'ok': {response}")
        return response

    def _call_optional_service(self, service_name: str, **kwargs) -> Dict[str, object]:
        """
        Call an optional service on the adapter.

        Missing/unready optional services are expected in some setups and will
        not be logged as warnings. Any other explicit failure is still logged.

        Args:
            service_name: Optional service identifier.
            **kwargs: Service arguments.

        Returns:
            Adapter response dict with "ok" and "message" fields.
        """
        response = self.adapter.call_service(service_name, **kwargs)
        if not isinstance(response, dict):
            print(f"WARNING: optional service '{service_name}' returned non-dict response: {response}")
            return response

        if response.get("ok", None) is True:
            return response

        message = str(response.get("message", ""))
        message_lower = message.lower()
        if "not available" in message_lower or "not ready" in message_lower:
            return response

        # print(f"WARNING: optional service '{service_name}' failed: {message}")
        return response

    def _set_compliance(self, params: Dict[str, Any]) -> None:
        """
        Send compliance parameters to the adapter.

        Resolves ``damping_ratio`` to ``cartesian_damping`` if present.

        Args:
            params: Compliance parameter mapping.
        """
        if params:
            from serl_framework.utils.config import resolve_compliance_params
            self._call_service("set_compliance", params=resolve_compliance_params(params))

    def _set_action_scale(self, scale) -> None:
        """Switch the active action scale.

        Unlike ``_set_compliance`` this is purely internal state -- the
        controller side does not need to be informed.

        Args:
            scale: 3-element action scale (translation_m, rotation_rad, gripper).
        """
        self.action_scale = scale

    def _set_freespace_params(self) -> None:
        """Activate free-space impedance and action scale (for reset moves)."""
        self._set_compliance(self.config.PRECISION_PARAM)
        self._set_action_scale(self._free_action_scale)

    def _set_task_params(self) -> None:
        """Activate task-phase impedance and action scale (for policy rollout)."""
        self._set_compliance(self.config.COMPLIANCE_PARAM)
        self._set_action_scale(self._task_action_scale)

    def _set_payload_interactive(self) -> None:
        """
        Prompt operator before applying payload parameters.
        """
        input("Put arm into programing mode and press enter.")
        self._set_payload()
        input("Put arm into execution mode and press enter.")
        for _ in range(2):
            self._recover()
            time.sleep(1)

    def _set_payload(self) -> None:
        """
        Apply payload parameters using the adapter service.
        """
        payload = self.config.PAYLOAD_PARAM or self.config.LOAD_PARAM
        mass = payload.get("mass", 0.0)
        center_of_mass = payload.get("center_of_mass", payload.get("F_x_center_load", [0.0, 0.0, 0.0]))
        inertia = payload.get("inertia", payload.get("load_inertia", [0.0] * 9))
        self._call_service(
            "set_payload",
            mass=mass,
            center_of_mass=center_of_mass,
            inertia=inertia,
        )
