"""
Ursina-based fake robot that mirrors /hilserl/command_pose into /hilserl/tcp_pose.

This node provides the ROS2 topics expected by RobotAdapter while rendering
a simple scene (ball + cube). The ball position follows the latest /hilserl/command_pose.
"""

import argparse
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from ursina import *  # noqa: F403
from panda3d.core import ClockObject, GraphicsOutput, PerspectiveLens, Texture as P3DTexture
from ursina.prefabs.editor_camera import EditorCamera
import cv2

from serl_framework.utils.rotations import quat_to_angvel


@dataclass
class PoseState:
    """
    Thread-safe holder for the commanded pose.
    """

    pose: np.ndarray
    stamp: float


@dataclass
class ImageState:
    """
    Thread-safe holder for the latest camera images.
    """

    images: dict[str, np.ndarray]


@dataclass
class CameraStream:
    """
    Offscreen camera stream configuration.
    """

    name: str
    tex: Texture
    width: int
    height: int
    camera: object




class UrsinaRobotNode(Node):
    """
    ROS2 node that mirrors command_pose into tcp_pose and renders a ball + cube.
    """

    def __init__(
        self,
        pose_state: PoseState,
        lock: threading.Lock,
        image_state: ImageState,
        image_lock: threading.Lock,
        camera_names: list[str],
        image_width: int,
        image_height: int,
        robot_dof: int,
        frame_id: str,
        publish_rate_hz: float,
    ) -> None:
        super().__init__("serl_ros2_ursina_sim")
        self.pose_state = pose_state
        self.lock = lock
        self.image_state = image_state
        self.image_lock = image_lock
        self.camera_names = camera_names
        self.image_width = image_width
        self.image_height = image_height
        self.robot_dof = robot_dof
        self.frame_id = frame_id
        self.publish_rate_hz = publish_rate_hz

        self._pose_pub = self.create_publisher(PoseStamped, "/hilserl/tcp_pose", 10)
        self._twist_pub = self.create_publisher(TwistStamped, "/hilserl/tcp_twist", 10)
        self._joint_pub = self.create_publisher(JointState, "/hilserl/joint_states", 10)
        self._gripper_pub = self.create_publisher(Float32, "/hilserl/gripper_pos", 10)
        self._image_pubs = {
            name: self.create_publisher(Image, f"/hilserl/camera/{name}/image", 10)
            for name in self.camera_names
        }

        self._command_sub = self.create_subscription(
            PoseStamped, "/hilserl/command_pose", self._on_command_pose, 10
        )
        self._clear_error_srv = self.create_service(
            Trigger, "/hilserl/clear_error", self._on_clear_error
        )

        self._last_pose = pose_state.pose.copy()
        self._last_stamp = pose_state.stamp
        self._fallback_image = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        self._joint_names = [f"joint_{idx}" for idx in range(self.robot_dof)]

        period = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0 else 0.05
        self._timer = self.create_timer(period, self._publish_state)

    def _on_command_pose(self, msg: PoseStamped) -> None:
        with self.lock:
            self.pose_state.pose = np.array(
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ],
                dtype=np.float32,
            )
            self.pose_state.stamp = time.time()

    def _on_clear_error(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = "ok"
        return response

    def _publish_state(self) -> None:
        with self.lock:
            pose = self.pose_state.pose.copy()
            stamp = self.pose_state.stamp

        dt = max(1e-6, stamp - self._last_stamp)
        lin_vel = (pose[:3] - self._last_pose[:3]) / dt
        ang_vel = quat_to_angvel(self._last_pose[3:], pose[3:], dt)

        self._last_pose = pose
        self._last_stamp = stamp

        ros_stamp = self.get_clock().now().to_msg()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = ros_stamp
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(pose[0])
        pose_msg.pose.position.y = float(pose[1])
        pose_msg.pose.position.z = float(pose[2])
        pose_msg.pose.orientation.x = float(pose[3])
        pose_msg.pose.orientation.y = float(pose[4])
        pose_msg.pose.orientation.z = float(pose[5])
        pose_msg.pose.orientation.w = float(pose[6])
        self._pose_pub.publish(pose_msg)

        twist_msg = TwistStamped()
        twist_msg.header.stamp = ros_stamp
        twist_msg.header.frame_id = self.frame_id
        twist_msg.twist.linear.x = float(lin_vel[0])
        twist_msg.twist.linear.y = float(lin_vel[1])
        twist_msg.twist.linear.z = float(lin_vel[2])
        twist_msg.twist.angular.x = float(ang_vel[0])
        twist_msg.twist.angular.y = float(ang_vel[1])
        twist_msg.twist.angular.z = float(ang_vel[2])
        self._twist_pub.publish(twist_msg)

        joint_msg = JointState()
        joint_msg.header.stamp = ros_stamp
        joint_msg.name = self._joint_names
        joint_msg.position = [0.0] * self.robot_dof
        joint_msg.velocity = [0.0] * self.robot_dof
        self._joint_pub.publish(joint_msg)

        gripper_msg = Float32()
        gripper_msg.data = 1.0
        self._gripper_pub.publish(gripper_msg)

        with self.image_lock:
            image_cache = dict(self.image_state.images)

        for name, pub in self._image_pubs.items():
            image = image_cache.get(name, self._fallback_image)
            if (
                image is None
                or image.ndim != 3
                or image.shape[0] != self.image_height
                or image.shape[1] != self.image_width
                or image.shape[2] != 3
            ):
                image = self._fallback_image

            image_msg = Image()
            image_msg.header.stamp = ros_stamp
            image_msg.header.frame_id = name
            image_msg.height = self.image_height
            image_msg.width = self.image_width
            image_msg.encoding = "bgr8"
            image_msg.is_bigendian = False
            image_msg.step = self.image_width * 3
            image_msg.data = np.ascontiguousarray(image).tobytes()
            pub.publish(image_msg)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="serl_ros2 Ursina fake robot")
    parser.add_argument("--rate", type=float, default=50.0, help="Publish rate in Hz.")
    parser.add_argument(
        "--cameras",
        type=str,
        default="front,wrist",
        help="Comma-separated camera names to publish.",
    )
    parser.add_argument("--image-width", type=int, default=128, help="Dummy image width.")
    parser.add_argument("--image-height", type=int, default=128, help="Dummy image height.")
    parser.add_argument("--robot-dof", type=int, default=7, help="Number of joints.")
    parser.add_argument("--frame-id", type=str, default="base_link", help="Frame ID.")
    parser.add_argument(
        "--show-images",
        action="store_true",
        help="Display camera images in separate windows.",
    )
    parser.add_argument(
        "--publish-images",
        action="store_true",
        help="Publish camera images on ROS topics.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without Ursina rendering; publish state only.",
    )
    return parser.parse_args()


def _create_stream(
    name: str,
    parent: object,
    width: int,
    height: int,
    fov: float,
    position: tuple[float, float, float],
    look_at: object | None,
) -> CameraStream:
    """
    Create an offscreen camera stream with RAM-accessible texture.

    Args:
        name: Stream name.
        parent: Parent node for the camera.
        width: Image width.
        height: Image height.
        fov: Field of view in degrees.
        position: Camera position relative to the parent.
        look_at: Optional target to look at.

    Returns:
        CameraStream with texture and camera handle.
    """
    tex = P3DTexture(f"{name}_tex")
    tex.setup_2d_texture(
        width, height, P3DTexture.T_unsigned_byte, P3DTexture.F_rgba
    )
    tex.setCompression(P3DTexture.CM_off)

    buffer = base.win.makeTextureBuffer(f"{name}_buffer", width, height, tex, to_ram=True)
    buffer.addRenderTexture(tex, GraphicsOutput.RTMCopyRam)

    cam = base.makeCamera(buffer)
    cam.reparentTo(parent)
    cam.setPos(*position)
    if look_at is not None:
        cam.lookAt(look_at)

    lens = PerspectiveLens()
    lens.setFov(fov)
    cam.node().setLens(lens)

    return CameraStream(name=name, tex=tex, width=width, height=height, camera=cam)


def main() -> None:
    args = _parse_args()
    camera_names = [name.strip() for name in args.cameras.split(",") if name.strip()]

    rclpy.init()
    pose_state = PoseState(
        pose=np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        stamp=time.time(),
    )
    image_state = ImageState(images={})
    lock = threading.Lock()
    image_lock = threading.Lock()
    node = UrsinaRobotNode(
        pose_state=pose_state,
        lock=lock,
        image_state=image_state,
        image_lock=image_lock,
        camera_names=camera_names,
        image_width=args.image_width,
        image_height=args.image_height,
        robot_dof=args.robot_dof,
        frame_id=args.frame_id,
        publish_rate_hz=args.rate,
    )

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    if args.headless:
        app = Ursina(window_type="none")
    else:
        app = Ursina()
    globalClock.setMode(ClockObject.MLimited)
    globalClock.setFrameRate(args.rate)


    ball = Entity(model="sphere", scale=0.5, color=color.red)
    axis_length = 1.0
    axis_thickness = 0.03
    Entity(
        model="cube",
        parent=ball,
        position=(axis_length / 2.0, 0.0, 0.0),
        scale=(axis_length, axis_thickness, axis_thickness),
        color=color.red,
    )
    Entity(
        model="cube",
        parent=ball,
        position=(0.0, 0.0, axis_length / 2.0),
        scale=(axis_thickness, axis_thickness, axis_length),
        color=color.green,
    )
    Entity(
        model="cube",
        parent=ball,
        position=(0.0, axis_length / 2.0, 0.0),
        scale=(axis_thickness, axis_length, axis_thickness),
        color=color.blue,
    )

    def _ros_to_ursina_pos(pos: tuple[float, float, float]) -> tuple[float, float, float]:
        return (pos[0], pos[2], pos[1])

    cube = Entity(
        model="cube",
        position=_ros_to_ursina_pos((3.0, 0.0, 0.0)),
        rotation=(-90.0, 0.0, 0.0),
        color=color.azure,
    )
    Entity(model="plane", scale=20, y=-1, color=color.gray)
    DirectionalLight().look_at(Vec3(1, -1, -1))
    AmbientLight(color=color.rgba(120, 120, 120, 255))
    if not args.headless:
        EditorCamera()

    streams: dict[str, CameraStream] = {}
    if not args.headless and (args.publish_images or args.show_images):
        camera_specs: dict[str, dict[str, object]] = {}
        wrist_names = [name for name in sorted(camera_names) if name.startswith("wrist")]
        fixed_names = [name for name in sorted(camera_names) if not name.startswith("wrist")]

        wrist_radius = 0.3
        wrist_height = 0.1
        for idx, name in enumerate(wrist_names):
            angle = (2.0 * np.pi * idx) / max(len(wrist_names), 1)
            position = (
                wrist_radius * np.cos(angle),
                wrist_radius * np.sin(angle),
                wrist_height,
            )
            camera_specs[name] = {
                "parent": ball,
                "fov": 90,
                "position": position,
                "look_at": None,
            }

        fixed_radius = 2.5
        fixed_height = 1.5
        for idx, name in enumerate(fixed_names):
            angle = (2.0 * np.pi * idx) / max(len(fixed_names), 1)
            position = (
                fixed_radius * np.cos(angle),
                fixed_radius * np.sin(angle),
                fixed_height,
            )
            camera_specs[name] = {
                "parent": scene,
                "fov": 60,
                "position": position,
                "look_at": cube,
            }

        for name, spec in camera_specs.items():
            streams[name] = _create_stream(
                name=name,
                parent=spec["parent"],
                width=args.image_width,
                height=args.image_height,
                fov=spec["fov"],
                position=spec["position"],
                look_at=spec["look_at"],
            )

    last_fps_print = 0.0

    def _tick() -> None:
        nonlocal last_fps_print
        with lock:
            pose = pose_state.pose.copy()
        ball.position = Vec3(pose[0], pose[2], pose[1])
        # if time.dt > 0 and (time.time() - last_fps_print) >= 1.0:
        #     print(f"fps ~ {1.0 / max(time.dt, 1e-6):.1f}")
        #     last_fps_print = time.time()

        if not args.headless and (args.publish_images or args.show_images):
            latest_images: dict[str, np.ndarray] = {}
            for name, stream in streams.items():
                if not stream.tex.hasRamImage():
                    continue
                rgba = stream.tex.getRamImage()
                if rgba is None:
                    continue
                img = np.frombuffer(rgba, dtype=np.uint8).reshape(
                    (stream.height, stream.width, 4)
                )
                img = np.flipud(img)
                bgr = img[:, :, :3][:, :, ::-1]
                latest_images[name] = bgr

            if latest_images and args.publish_images:
                with image_lock:
                    image_state.images.update(latest_images)
            if latest_images and args.show_images:
                for name, image in latest_images.items():
                    cv2.imshow(f"{name} camera", image)
                cv2.waitKey(1)

    # Ursina runs update() on entities; attach tick to a helper entity.
    updater = Entity()
    updater.update = _tick

    app.run()
    if args.show_images:
        cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
