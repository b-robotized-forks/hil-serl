"""
Cube demo experiment configuration for the ROS2 RobotAdapter path.

This experiment keeps the task intentionally simple: move the TCP to a fixed
target pose above a cube on the table. It uses the serl_framework RobotEnv and
instantiates the adapter via serl_ros2 (no serl_robot_infra dependencies).
"""

import atexit
import os
import jax
import jax.numpy as jnp
import numpy as np

from serl_framework.envs.relative_env import RelativeFrame
from serl_framework.envs.robot_env import DefaultEnvConfig
from serl_framework.wrappers import (
    GripperCloseEnv,
    MultiCameraBinaryRewardClassifierWrapper,
    Quat2EulerWrapper,
    TeleopIntervention,
)
from serl_launcher.networks.reward_classifier import load_classifier_func
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

from serl_framework.train.config import DefaultTrainingConfig
from experiments.cube_demo_ros2.wrapper import CubeDemoEnv


class EnvConfig(DefaultEnvConfig):
    """
    Environment configuration for the cube demo.

    Update TARGET_POSE and safety limits to match the cube's position on your table.
    """

    # Camera names used by RobotEnv; must match adapter_config.yaml "cameras".
    CAMERAS = {
        "front": {},
        "wrist": {},
    }
    # Optional image crops per camera, e.g. {"front": lambda img: img[100:500, 200:900]}
    IMAGE_CROP = {}
    # "bgr" if camera images arrive as BGR, "rgb" if they are already RGB.
    IMAGE_CHANNEL_ORDER = "bgr"
    # Prefix for saved video files (when save_video=True).
    VIDEO_PREFIX = "cube_demo_ros2"

    # Target pose the TCP should reach (xyz + rpy in radians).
    # Ursina sim default cube is at (3, 0, 0); orientation is unused in the sim.
    TARGET_POSE = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Reset pose used at episode start (xyz + rpy in radians).
    # Ursina sim starts the ball at (0.5, 0.0, 0.4) with identity orientation.
    RESET_POSE = np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0])
    # Optional grasp pose (unused in this demo but kept for config parity).
    GRASP_POSE = RESET_POSE.copy()
    # Reward success thresholds for pose deltas (xyz + rpy in radians).
    REWARD_THRESHOLD = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0])

    # Absolute pose safety bounds (xyz + rpy). Set to safe workspace limits.
    # Wider bounds to cover the Ursina sim workspace.
    ABS_POSE_LIMIT_LOW = np.array([0.0, -1.0, -1.0, -np.pi, -np.pi, -np.pi])
    ABS_POSE_LIMIT_HIGH = np.array([3.5, 1.0, 1.0, np.pi, np.pi, np.pi])

    # Action scaling for xyz, rotvec, and gripper commands.
    ACTION_SCALE = (0.1, 0.1, 1.0)
    # Show a live camera window for debugging.
    DISPLAY_IMAGE = True
    # Episode timeout in steps.
    MAX_EPISODE_LENGTH = 200

    # Compliance and precision parameters forwarded to the adapter (optional).
    COMPLIANCE_PARAM = {}
    PRECISION_PARAM = {}

    # Maximum age for state. When this is None, a very "young" state only will be
    # acceptable (the time since the command was sent). This is overwritten now because
    # the full extent of this still has to be tested. We should use a generouse time.
    STATE_MAX_AGE_S = 0.5


class TrainConfig(DefaultTrainingConfig):
    """
    Training configuration for the cube demo ROS2 experiment.
    """

    # Image keys must match EnvConfig.CAMERAS.
    image_keys = ["front", "wrist"]
    # Reward classifier image keys; set to match the camera list if using a classifier.
    # IMPORTANT: set to empty if no classifier is used
    classifier_keys = None # ["front", "wrist"]
    # Proprioception keys to include in the flattened state vector.
    proprio_keys = ["tcp_pose", "tcp_vel", "gripper_pose"]
    # Encoder choice for pixel policies.
    encoder_type = "resnet-pretrained"
    # Fixed-gripper setup keeps the action space at 6D (no learned gripper).
    setup_mode = "single-arm-fixed-gripper"

    # Training limits - adjust based on task complexity
    # Simple tasks like this demo often converge in 20-50k steps
    max_steps: int = 100000

    # Checkpoint settings - save model every N steps (0 = disabled)
    # Checkpoints are saved to the --checkpoint_path directory (e.g., first_run/)
    checkpoint_period: int = 5000

    # Adapter config used by serl_ros2 RobotAdapter.
    adapter_config_path = os.path.join(
        os.path.dirname(__file__), "adapter_config.yaml"
    )
    # Set True to enable teleop for interventions and data collection.
    # Teleop device options:
    #   - "spacemouse": Direct SpaceMouse input (requires pyspacemouse)
    #   - "joy": ROS2 Joy message subscriber (for keyboard teleop or gamepad via ROS2)
    use_teleop = True
    teleop_device = "joy"  # "spacemouse" or "joy"
    teleop_joy_topic = "teleop_joy"  # Joy topic name when teleop_device="joy"
    # Teleop frame ids used by TeleopIntervention for automatic frame routing.
    teleop_base_frame_id = "base"
    teleop_tcp_frame_id = "tcp"
    # Used only when teleop_device == "spacemouse". Ignored for Joy teleop.
    teleop_spacemouse_frame_id = "tcp"

    def get_environment(
        self,
        fake_env: bool = False,
        save_video: bool = False,
        classifier: bool = False,
    ):
        """
        Build the cube demo environment using RobotEnv and serl_ros2.

        Args:
            fake_env: If True, skip adapter wiring for learner runs.
            save_video: If True, store video clips on reset.
            classifier: Unused for this demo (no reward classifier).

        Returns:
            Configured Gymnasium environment.
        """
        adapter = None
        teleop = None
        owns_adapter = False
        if not fake_env:
            # Lazy import keeps learner-only runs ROS2-independent.
            from serl_ros2.robot_adapter import RobotAdapter
            if self.use_teleop:
                if self.teleop_device == "joy":
                    from serl_ros2.joy_teleop_adapter import JoyTeleopAdapter
                    teleop = JoyTeleopAdapter(topic=self.teleop_joy_topic)
                else:  # default to spacemouse
                    from serl_framework.spacemouse_teleop import SpaceMouseTeleop
                    teleop = SpaceMouseTeleop(
                        frame_id=self.teleop_spacemouse_frame_id
                    )
            adapter = RobotAdapter.create(
                config_path=self.adapter_config_path,
                teleop_adapter=teleop,
            )
            adapter.start()
            adapter.wait_for_state(timeout_s=10.0)
            atexit.register(adapter.stop)
            owns_adapter = True

        env = CubeDemoEnv(
            hz=30,
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
            adapter=adapter,
            owns_adapter=owns_adapter,
        )
        env = GripperCloseEnv(env)
        if teleop is not None:
            env = TeleopIntervention(
                env,
                teleop_adapter=teleop,
                base_frame_id=self.teleop_base_frame_id,
                tcp_frame_id=self.teleop_tcp_frame_id,
            )
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier and self.classifier_keys:
            classifier_fn = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
                checkpoint_path=os.path.abspath("classifier_ckpt/"),
            )

            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                prob = float(np.asarray(sigmoid(classifier_fn(obs))).reshape(-1)[0])
                return int(prob > 0.75)

            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        return env
