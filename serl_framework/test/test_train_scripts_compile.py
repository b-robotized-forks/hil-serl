"""Compile check for all modules in serl_framework.train.

The training scripts require jax and serl_launcher at runtime, which are not
test dependencies here. Byte-compiling every module still catches syntax
errors without importing anything heavy.
"""

import py_compile
from pathlib import Path

import pytest

TRAIN_DIR = Path(__file__).resolve().parent.parent / "serl_framework" / "train"


@pytest.mark.parametrize("path", sorted(TRAIN_DIR.glob("*.py")), ids=lambda p: p.name)
def test_module_compiles(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)
