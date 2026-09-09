# Installation

Tested with Python 3.12 on Ubuntu 24.04. Expected to also work on Python 3.10+.

## 1. Python environment

Create and activate a Python environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

## 2. JAX (pinned to 0.4.36)

**This fork requires JAX 0.4.36.** The pin is deliberate, not just a frozen requirements file:

- The original repository uses `0.4.35`, which breaks on Python 3.12 / Ubuntu 24.04
  (see [jax#24826](https://github.com/jax-ml/jax/issues/24826#issuecomment-2528091995)).
- Newer JAX releases do **not** work yet. The code was tried with JAX 0.9 and fails.
  Upgrading requires code changes (and matching flax/optax/orbax version bumps).

Install for CPU:

```bash
pip install --upgrade "jax[cpu]==0.4.36"
```

or for GPU (CUDA 12):

```bash
pip install --upgrade "jax[cuda12_pip]==0.4.36" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 3. Core packages

From the repo root:

```bash
pip install -r requirements.txt
pip install -e serl_launcher
pip install -e serl_framework
```

`requirements.txt` pins the JAX ecosystem packages (flax, optax, orbax-checkpoint, ...) to
versions compatible with JAX 0.4.36.

> [!NOTE]
> Installing `serl_launcher` automatically pulls [agentlace](https://github.com/youliangtan/agentlace)
> (the actor/learner communication layer) from GitHub, pinned to a commit of a fork. The fork is
> identical to the commit the original hil-serl pins, plus a one-line fix that removes an obsolete
> dependency breaking installation on modern Python. No separate install step is needed, but this
> pip step does clone from GitHub.

## 4. Verify the install

Check JAX/Flax and the available devices (should print `0.4.36`):

```bash
python -c "import jax, flax; print(jax.__version__, flax.__version__, jax.devices())"
```

Run the ROS-free environment smoke test (steps a mock robot through `RobotEnv`):

```bash
python serl_framework/scripts/env_smoke.py --steps 3
```

Optionally run the unit tests:

```bash
cd serl_framework && pytest -q ./test && cd ..
```

## 5. ROS2 workspace

Add the ROS2 adapter packages to your ROS workspace and build (with the venv active):

```bash
cd <your_ros_ws>/src
ln -s <path-to>/hil-serl/serl_ros2
ln -s <path-to>/hil-serl/serl_msgs
cd <your_ros_ws>
colcon build --symlink-install
```

## 6. Optional: SpaceMouse

For SpaceMouse teleop, install the HID dependency and see the setup and troubleshooting notes in
[serl_framework/README.md](serl_framework/README.md#spacemouse):

```bash
sudo apt install libhidapi-dev
```

## Next steps

Run the hardware-free cube demo:
[examples/experiments/cube_demo_ros2/README.md](examples/experiments/cube_demo_ros2/README.md).
