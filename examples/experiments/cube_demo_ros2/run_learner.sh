if [ -z "$1" ]; then
    echo "Usage: $0 <demo_path> [extra args...]"
    exit 1
fi

DEMO_PATH="$1"
shift

export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.3} && \
export PYTHONPATH="$(cd ../.. && pwd):${PYTHONPATH:-}" && \
python -m serl_framework.train.train_rlpd "$@" \
    --exp_name=cube_demo_ros2 \
    --checkpoint_path=first_run \
    --demo_path="$DEMO_PATH" \
    --learner \
