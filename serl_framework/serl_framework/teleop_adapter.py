"""
Teleop adapter interfaces for HIL-SERL.

This module defines a device-agnostic TeleopAdapter that produces the teleop
signals HIL-SERL expects:
translation deltas `xyz`, rotation vector `rpy`, button states, and source
frame id metadata.
Concrete subclasses handle device-specific IO.
"""

from abc import ABC, abstractmethod
from enum import IntEnum
import multiprocessing
import time


class TeleopButton(IntEnum):
    """
    Canonical teleop button indices.

    Adapters should return button states in this order so consumers are device-agnostic.
    """

    GRIPPER_CLOSE = 0
    GRIPPER_OPEN = 1


class TeleopAdapter(ABC):
    """
    Abstract base class for teleop adapters.

    Teleop adapters publish device input as translation deltas (xyz) and
    rotation-vector deltas (rx, ry, rz) per device, plus associated button
    states. The rotation vector components are interpreted as angles around the
    corresponding axes (rotation around x, y, and z --> RPY).
    Subclasses implement device IO while the base class manages the background
    read loop and shared state.
    """

    def __init__(self) -> None:
        """
        Initialize the adapter.
        Does not start the background read loop - use start() for that.
        """
        xyz, rpy, buttons, frame_id = self._default_teleop()
        self._manager = multiprocessing.Manager()
        self._shared = self._manager.dict()
        self._shared["xyz"] = list(xyz)
        self._shared["rpy"] = list(rpy)
        self._shared["buttons"] = list(buttons)
        self._shared["frame_id"] = str(frame_id)

        self._stop: multiprocessing.Event | None = None
        self._process: multiprocessing.Process | None = None

    def start(self) -> None:
        """
        Start the background read loop if it is not already running.

        This method is idempotent. It can be called again after close() to
        restart the reader loop.
        """
        if self._process is not None and self._process.is_alive():
            return
        self._stop = multiprocessing.Event()
        self._process = multiprocessing.Process(target=self._read_loop)
        self._process.daemon = True
        self._process.start()

    def _read_loop(self) -> None:
        """
        Continuously read device state and update the shared teleop cache.
        """
        if self._stop is None:
            return
        period_s = 1.0 / 60.0
        self._open_device()
        try:
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                devices = self._read_devices()
                if not devices:
                    sleep_for = period_s - (time.perf_counter() - loop_start)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    continue

                xyz_all: list[float] = []
                rpy_all: list[float] = []
                buttons_all: list[int] = []
                frame_id = ""
                for device in devices:
                    if len(device) == 4:
                        xyz, rpy, buttons, device_frame_id = device
                    elif len(device) == 3:
                        xyz, rpy, buttons = device
                        device_frame_id = ""
                    else:
                        raise ValueError(
                            "Teleop devices must report (xyz, rpy, buttons[, frame_id])."
                        )
                    if len(xyz) != 3 or len(rpy) != 3:
                        raise ValueError(
                            "Teleop devices must report xyz and rpy vectors of length 3."
                        )
                    xyz_all.extend(xyz)
                    rpy_all.extend(rpy)
                    buttons_all.extend(buttons)
                    device_frame_id = str(device_frame_id)
                    if not frame_id:
                        frame_id = device_frame_id
                    elif device_frame_id and device_frame_id != frame_id:
                        frame_id = ""

                self._shared["xyz"] = xyz_all
                self._shared["rpy"] = rpy_all
                self._shared["buttons"] = buttons_all
                self._shared["frame_id"] = frame_id
                sleep_for = period_s - (time.perf_counter() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self._close_device()

    def get_teleop(self) -> tuple[list[float], list[float], list[int], str]:
        """
        Return the latest teleop reading.

        Returns:
            Tuple of (xyz, rpy, buttons, frame_id).

            Axis conventions:
                - xyz: translation deltas along x/y/z axes.
                - rpy: rotation-vector deltas around x/y/z axes.
                - buttons: states ordered by TeleopButton indices.
                - frame_id: source command frame. This may be an empty string; in
                  that case consumers should default to the configured TCP frame.
        """
        return (
            list(self._shared["xyz"]),
            list(self._shared["rpy"]),
            list(self._shared["buttons"]),
            str(self._shared.get("frame_id", "")),
        )

    def get_button(self, button: TeleopButton) -> int:
        """
        Return the state of a canonical teleop button.

        Args:
            button: TeleopButton enum value.

        Returns:
            1 if pressed, 0 otherwise.
        """
        _xyz, _rpy, buttons, _frame_id = self.get_teleop()
        idx = int(button)
        if idx < 0 or idx >= len(buttons):
            return 0
        return int(bool(buttons[idx]))

    def close(self) -> None:
        """
        Stop the background reader and close the device.
        """
        if self._process is None:
            return
        if self._stop is not None:
            self._stop.set()
        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._process = None
        self._stop = None

    @abstractmethod
    def _default_teleop(self) -> tuple[list[float], list[float], list[int], str]:
        """
        Provide default (xyz, rpy, buttons, frame_id) values before the first read.
        """

    @abstractmethod
    def _read_devices(self) -> list[tuple[list[float], list[float], list[int], str]]:
        """
        Read device state and return a list of (xyz, rpy, buttons, frame_id) tuples.

        `frame_id` may be empty if the source does not provide frame metadata.
        """

    @abstractmethod
    def _open_device(self) -> None:
        """
        Open the underlying device connection.
        """

    @abstractmethod
    def _close_device(self) -> None:
        """
        Close the underlying device connection.
        """
