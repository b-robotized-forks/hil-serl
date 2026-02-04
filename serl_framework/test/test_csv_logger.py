"""Tests for the CSV metrics logger."""

import csv
from pathlib import Path

from serl_framework.train.csv_logger import CSVLogger


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_log_writes_header_and_row(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(str(path))
    logger.log({"loss": 1.5}, step=10)

    rows = _read_rows(path)
    assert rows == [{"step": "10", "loss": "1.5"}]


def test_log_flattens_nested_dicts(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(str(path))
    logger.log({"actor": {"loss": 1.0, "entropy": 0.5}}, step=1)

    rows = _read_rows(path)
    assert rows == [{"step": "1", "actor/loss": "1.0", "actor/entropy": "0.5"}]


def test_schema_growth_preserves_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    logger = CSVLogger(str(path))
    logger.log({"loss": 1.0}, step=1)
    logger.log({"loss": 0.5, "q": 2.0}, step=2)

    rows = _read_rows(path)
    assert [row["loss"] for row in rows] == ["1.0", "0.5"]
    assert rows[0]["q"] == ""
    assert rows[1]["q"] == "2.0"


def test_append_to_existing_file_keeps_schema(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    CSVLogger(str(path)).log({"loss": 1.0}, step=1)
    # A fresh logger instance (e.g. after a resume) must reuse the existing header.
    CSVLogger(str(path)).log({"loss": 0.9}, step=2)

    content = path.read_text(encoding="utf-8")
    assert content.count("step,loss") == 1
    assert [row["step"] for row in _read_rows(path)] == ["1", "2"]


def test_non_scalar_values_are_stringified(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    CSVLogger(str(path)).log({"shape": [128, 128]}, step=1)

    rows = _read_rows(path)
    assert rows[0]["shape"] == "[128, 128]"
