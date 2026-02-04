export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.1} && \
export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}" && \
python -m serl_framework.train.train_rlpd "$@" \
    --exp_name=cube_demo_ros2 \
    --checkpoint_path=first_run \
    --actor \
