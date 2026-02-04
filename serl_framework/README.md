# serl_framework

Robot-agnostic, ROS2-free environment components for HIL-SERL. This package hosts
generic Gymnasium environments and utilities that work with any robot integration
via the `serl_framework.robot_adapter` interface.

## Overview

- `serl_framework.envs.robot_env.RobotEnv` is the core env. It consumes synchronized
  state + image snapshots from a `RobotAdapter`, and emits the observation layout
  expected by existing HIL-SERL configs and wrappers.
- Config is Python-first (via `DefaultEnvConfig`), with optional YAML overrides.
- Camera configuration lives in YAML (or your config class), while image cropping
  still uses Python callables (`IMAGE_CROP`) for flexibility.
- `REALSENSE_CAMERAS` is not supported in `serl_framework`. Use `CAMERAS` instead.

## Training entry points (`serl_framework.train`)

The executable training and data collection scripts live in the installed
`serl_framework.train` subpackage and are run as modules:

```bash
python -m serl_framework.train.train_rlpd --help
python -m serl_framework.train.record_demos --help
python -m serl_framework.train.record_success_fail --help
python -m serl_framework.train.train_reward_classifier --help
python -m serl_framework.train.stream_classifier_prob --help
```

They additionally require the RL stack, which is not part of this package's base dependencies.
From the repo checkout:

```bash
pip install -r requirements.txt
pip install -e serl_launcher
```

Experiment configs are resolved through a mapping module passed via `--config_mapping`
(default `experiments.mappings`): a module exporting a `CONFIG_MAPPING` dict from experiment name
to `TrainConfig` class (or a lazy dotted path string, see `serl_framework/train/mappings.py`).
The bundled examples keep their mapping in `examples/experiments/mappings.py`, so runs from the
repo root use `PYTHONPATH=examples`. For your own experiments, create your own package with a
mapping module and point `--config_mapping` at it; the experiment classes derive from
`serl_framework.train.config.DefaultTrainingConfig`.

The subpackage also hosts the shared training helpers: `csv_logger` (CSV metrics logging),
`csv_log_cleanup` (pruning stale rows on resume), and `pause_prompt` (interactive learner pause).

## Installation

Install `serl_framework`:

```bash
pip install -e <path-to>/serl_framework
```


### Spacemouse

If you plan to use `SpaceMouseTeleop` (pyspacemouse), install the HID dependency:

```bash
sudo apt install libhidapi-dev
```

> [!NOTE]
> You do **not** need the common package `spacenavd`, though it is also fine if you have it installed.

Test your spacemouse with

```bash
python3 -c "
import time, pyspacemouse
pyspacemouse.open()
print('Move the mouse:')
for _ in range(20):
    print(pyspacemouse.read())
    time.sleep(0.2)
"
```

Please also see the Troubleshoot section below for more details (e.g. permission denied issue)

## Running unit tests

From the `serl_framework` directory:

```bash
pytest -q ./test
```

Run a single test by name:

```bash
pytest -q ./test -k test_get_observation_returns_cached_state
```


## Concept: RobotAdapter

The RobotAdapter is introduced as a communication interface between a robot and
Hil-serl. It is ROS-free; user code imports the robot adapter, starts it, and
calls services from a plain Python loop, as in this example (using the ROS2
implementation from `serl_ros2`):

```python
import time

# only ROS-specific dependency
from serl_ros2.robot_adapter import RobotAdapter


def main() -> int:
    adapter = RobotAdapter.create(config_path="config/adapter_example.yaml")
    adapter.start()
    try:
        while True:
            result = adapter.call_service("clear_error", timeout_sec=2.0)
            print(result)
            time.sleep(0.5)
    finally:
        adapter.stop()

    return 0
```

This example assumes you run it from the `serl_ros2` package directory so the
config path resolves correctly.

### Running in the same thread

The RobotAdapter method `start()` will kick off a thread to run the communication with the robot
(e.g. in the ROS2 implementation).
Some use cases (tests, simple scripts) prefer to keep everything on one thread.
Adapters can expose a `run()` method (instead of `start()/stop()`) that spins
in the current thread and invokes best-effort timed callbacks.

## Concept: TeleopAdapter

TeleopAdapter is the device-agnostic interface for human teleoperation input
used during interventions. It produces robot-agnostic deltas in the form
`(xyz, rpy, buttons, frame_id)`.

- RobotAdapter accepts an optional TeleopAdapter instance, calls
  `teleop_adapter.start()` during `RobotAdapter.start()`, and surfaces teleop data
  via `RobotAdapter.get_teleop()` so env wrappers can build `info["intervene_action"]`
  without knowing which device produced the input.
- TeleopAdapter axes are interpreted by HIL-SERL as: translation along x/y/z
  followed by rotation-vector deltas around x/y/z (also understood as RPY).
  Device implementations may remap axes to fit with intuitive control.
- `frame_id` indicates whether the command is base-frame or TCP-frame (or any
  configured alias). It may be empty if the source has no frame metadata; in
  that case `TeleopIntervention` defaults to TCP-frame behavior.
- TeleopAdapter supports multiple devices by concatenating per-device xyz/rpy
  and buttons in device order.

### Available TeleopAdapter implementations

| Adapter | Package | Description |
|---------|---------|-------------|
| `SpaceMouseTeleop` | `serl_framework` | 3Dconnexion SpaceMouse (direct HID) |
| `JoyTeleopAdapter` | `serl_ros2` | ROS2 Joy messages (keyboard, gamepad, or pose-to-joy source) |

### SpaceMouse example

```python
from serl_framework.spacemouse_teleop import SpaceMouseTeleop
from serl_ros2.robot_adapter import RobotAdapter

teleop = SpaceMouseTeleop(frame_id="tcp")
adapter = RobotAdapter.create(config_path="config/adapter_example.yaml", teleop_adapter=teleop)
adapter.start()
```

Set `frame_id="base"` if your SpaceMouse deltas should be interpreted in base frame.

When you are done with teleop input, call `teleop.close()` to stop the background
reader process.

### Joy (keyboard/gamepad/pose-to-joy) example

If you don't have a SpaceMouse, use `JoyTeleopAdapter` from `serl_ros2` which
subscribes to `sensor_msgs/Joy` messages:

```python
from serl_ros2.joy_teleop_adapter import JoyTeleopAdapter
from serl_ros2.robot_adapter import RobotAdapter

teleop = JoyTeleopAdapter(topic="teleop_joy")
adapter = RobotAdapter.create(config_path="config/adapter_example.yaml", teleop_adapter=teleop)
adapter.start()
```

See [serl_ros2/README.md - Teleop](../serl_ros2/README.md#teleop) for keyboard,
gamepad, and pose-to-joy setup.

## Interface return types

Observation, state, and service responses use plain dictionaries and arrays
instead of custom dataclasses. This keeps the interface light-weight and avoids
forcing downstream packages (e.g., `serl_launcher`) to import custom types. The
expected dictionary keys are documented in
`serl_framework/serl_framework/robot_adapter.py`.


## Troubleshooting

### Spacemouse

If you encounter an error like `Failed to open device` or `Permission denied` when
trying to use your SpaceMouse on Linux, this is typically a permissions issue.
Normal users don't have permission to access HID devices by default.

Error example:

```
Traceback (most recent call last):
  File "/home/user/.local/lib/python3.8/site-packages/pyspacemouse/pyspacemouse.py", line 183, in open
    self.device.open()
  File "/home/user/.local/lib/python3.8/site-packages/easyhid/easyhid.py", line 134, in open
    raise HIDException("Failed to open device")
easyhid.easyhid.HIDException: Failed to open device
```

Solution:
Source: https://spacemouse.kubaandrysek.cz/troubleshooting/#failed-to-open-device-permission-denied

1) Find your device's Vendor ID and Product ID:
```
lsusb
```

Look for your SpaceMouse device. Example output:
```
Bus 001 Device 013: ID 256f:c652 3Dconnexion Universal Receiver
```
Here, `256f` is the Vendor ID and `c652` is the Product ID.

2) Create udev rules to grant permissions:
```
cd /etc/udev/rules.d
sudo touch 99-spacemouse.rules
sudo nano 99-spacemouse.rules
```

Add the following rules (replace `256f` and `c652` with your Vendor ID and Product ID):
```
SUBSYSTEM=="input", GROUP="input", MODE="0660"
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c652", MODE="0666"
```

3) Reload udev rules:
```
sudo udevadm control --reload-rules
sudo udevadm trigger
```

4) Add your user to the input group:
```
sudo usermod -a -G input $USER
```

Disconnect and reconnect your SpaceMouse, then log out and log back in to Ubuntu
(or restart your computer).

After these steps, your SpaceMouse should work correctly without permission errors.

## The robot environment

This package also provides a RobotEnv gymnasium environment.

### YAML example

```yaml
# robot_env.yaml
cameras:
  wrist_1:
    topic: "/camera/wrist_1/image"
  wrist_2:
    topic: "/camera/wrist_2/image"
target_pose: [0.58, -0.03, 0.27, 3.14, 0.0, 0.0]
abs_pose_limit_low: [0.55, -0.06, 0.25, 3.12, -0.1, -0.4]
abs_pose_limit_high: [0.61, 0.00, 0.32, 3.16, 0.1, 0.4]
action_scale: [0.01, 0.06, 1.0]
```

### Python example

```python
from serl_framework.envs.robot_env import RobotEnv

env = RobotEnv(config="robot_env.yaml", adapter=adapter)
```

### Smoke test executable

The smoke test runs the env with a fake adapter to validate basic wiring.
Run from the `hil-serl/` directory:

```bash
python serl_framework/scripts/env_smoke.py --steps 3
```
