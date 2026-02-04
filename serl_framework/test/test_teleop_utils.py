"""
Unit tests for teleop frame-conversion helpers.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from serl_framework.utils.teleop import (  # noqa: E402
    transform_delta_to_base,
    transform_translation_to_tip,
)


def test_transform_translation_to_tip_inverts_base_mapping() -> None:
    """
    Verify base->tip translation conversion inverts the existing tip->base helper.
    """
    tcp_quat = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_quat()
    delta_tip = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    delta_base, _ = transform_delta_to_base(
        delta_tip,
        np.zeros((3,), dtype=np.float32),
        tcp_quat,
    )

    recovered_tip = transform_translation_to_tip(delta_base, tcp_quat)
    assert np.allclose(recovered_tip, delta_tip, atol=1e-6)


def test_transform_translation_to_tip_rotates_base_into_tool_axes() -> None:
    """
    Verify a base-frame vector is expressed correctly in the TCP/tool frame.
    """
    tcp_quat = Rotation.from_euler("xyz", [0.0, 0.0, np.pi / 2]).as_quat()
    delta_base = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    delta_tip = transform_translation_to_tip(delta_base, tcp_quat)
    assert np.allclose(delta_tip, np.array([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)
