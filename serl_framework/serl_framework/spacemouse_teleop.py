"""
SpaceMouse teleop adapter implementation using pyspacemouse
(See https://spacemouse.kubaandrysek.cz/).
"""

from serl_framework.teleop_adapter import TeleopAdapter, TeleopButton
import pyspacemouse


class SpaceMouseTeleop(TeleopAdapter):
    """
    Teleop adapter that reads SpaceMouse state via pyspacemouse.
    """

    DEFAULT_BUTTON_MAPPING = {
        TeleopButton.GRIPPER_CLOSE: 0,
        TeleopButton.GRIPPER_OPEN: 14,
    }

    # Default axis mapping: [source_attr, sign]
    # Maps pyspacemouse state attributes to xyz/rpy outputs.
    DEFAULT_XYZ_AXES = [("x", -1.0), ("y", 1.0), ("z", 1.0)]
    DEFAULT_RPY_AXES = [("pitch", 1.0), ("roll", 1.0), ("yaw", 1.0)]

    # The two frame ids used for toggling. Press both buttons to swap.
    FRAME_IDS = ("tcp", "base")

    def __init__(
        self,
        button_mapping: dict[TeleopButton, int] | None = None,
        frame_id: str = "tcp",
        axis_mapping: dict[str, dict] | None = None,
    ) -> None:
        """
        Initialize the SpaceMouse teleop adapter.

        Args:
            button_mapping: Optional dict mapping TeleopButton to SpaceMouse button indices.
            frame_id: Initial command frame id ("tcp" or "base"). Press both
                open/close gripper buttons simultaneously to toggle.
            axis_mapping: Per-frame axis mapping. Dict keyed by frame id, each
                value is a dict with optional "xyz_axes" and "rpy_axes" lists.
                Example::

                    {
                        "tcp":  {"xyz_axes": [["x", 1], ["y", -1], ["z", -1]],
                                 "rpy_axes": [["pitch", -1], ["roll", -1], ["yaw", 1]]},
                        "base": {"xyz_axes": [["x", -1], ["y", 1], ["z", -1]],
                                 "rpy_axes": [["pitch", 1], ["roll", 1], ["yaw", 1]]},
                    }

                If a frame id is missing, or the entire argument is None,
                DEFAULT_XYZ_AXES / DEFAULT_RPY_AXES are used for that frame.
        """

        self._pyspacemouse = pyspacemouse
        self._device = None
        self._button_mapping = button_mapping or self.DEFAULT_BUTTON_MAPPING.copy()
        self._frame_id = str(frame_id).strip()
        if not self._frame_id:
            raise ValueError("SpaceMouseTeleop frame_id must not be empty.")
        self._axis_mapping = axis_mapping or {}
        self._prev_both_pressed = False
        self._toggle_cooldown_remaining = 0
        super().__init__()

    def _get_axes(self) -> tuple[list, list]:
        """Return (xyz_axes, rpy_axes) for the current frame."""
        frame_cfg = self._axis_mapping.get(self._frame_id, {})
        xyz = frame_cfg.get("xyz_axes", self.DEFAULT_XYZ_AXES)
        rpy = frame_cfg.get("rpy_axes", self.DEFAULT_RPY_AXES)
        return xyz, rpy

    def _default_teleop(self) -> tuple[list[float], list[float], list[int], str]:
        """
        Default teleop state for a single SpaceMouse.
        """
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0], self._frame_id

    def _open_device(self) -> None:
        """
        Open the SpaceMouse device.
        """
        self._device = self._pyspacemouse.open()

    def _close_device(self) -> None:
        """
        Close the SpaceMouse device.
        """
        if self._device is None:
            return
        if hasattr(self._device, "close"):
            self._device.close()
        self._device = None

    def _read_devices(self) -> list[tuple[list[float], list[float], list[int], str]]:
        """
        Read SpaceMouse devices and map to HIL-SERL axis conventions.
        """
        if self._device is None:
            return []
        state = self._device.read()
        if not state:
            return []

        xyz_axes, rpy_axes = self._get_axes()
        xyz = [getattr(state, attr) * sign for attr, sign in xyz_axes]
        rpy = [getattr(state, attr) * sign for attr, sign in rpy_axes]
        raw_buttons = [int(button) for button in state.buttons]

        # Detect both-button combo for frame toggle.
        close_idx = self._button_mapping.get(TeleopButton.GRIPPER_CLOSE, -1)
        open_idx = self._button_mapping.get(TeleopButton.GRIPPER_OPEN, -1)
        close_pressed = 0 <= close_idx < len(raw_buttons) and raw_buttons[close_idx]
        open_pressed = 0 <= open_idx < len(raw_buttons) and raw_buttons[open_idx]
        both_pressed = close_pressed and open_pressed

        if both_pressed and not self._prev_both_pressed:
            # Toggle on rising edge of both-button press
            a, b = self.FRAME_IDS
            self._frame_id = b if self._frame_id == a else a
            print(f"SpaceMouse frame toggled to: {self._frame_id}")
            # Suppress button actions for a short cooldown after toggle,
            # so releasing one button slightly before the other doesn't
            # trigger a spurious gripper action.
            self._toggle_cooldown_remaining = 10  # ~10 read cycles
        self._prev_both_pressed = both_pressed

        # Suppress individual button actions while both are held or during cooldown
        buttons = [0] * len(TeleopButton)
        if self._toggle_cooldown_remaining > 0:
            self._toggle_cooldown_remaining -= 1
        elif not both_pressed:
            for button, idx in self._button_mapping.items():
                if 0 <= idx < len(raw_buttons):
                    buttons[int(button)] = int(raw_buttons[idx])

        return [(xyz, rpy, buttons, self._frame_id)]
