"""Tests for pruning resumed training CSV logs."""

import csv
from pathlib import Path

from serl_framework.train.csv_log_cleanup import (
    prune_csv_rows_after_step,
    prune_csv_rows_on_backward_jumps,
    prune_training_csv_logs,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_prune_csv_rows_after_step_removes_newer_rows(tmp_path: Path) -> None:
    path = tmp_path / "learner_update_metrics.csv"
    _write_csv(
        path,
        ["step", "critic/critic_loss"],
        [
            {"step": 4999, "critic/critic_loss": 0.4},
            {"step": 5000, "critic/critic_loss": 0.3},
            {"step": 5001, "critic/critic_loss": 0.2},
        ],
    )

    result = prune_csv_rows_after_step(
        path,
        max_step=5000,
        step_columns=("step",),
    )

    assert result.step_column == "step"
    assert result.removed_rows == 1
    assert result.kept_rows == 2
    assert [row["step"] for row in _read_rows(path)] == ["4999", "5000"]


def test_prune_training_csv_logs_uses_callback_learner_step_for_actor_stats(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "actor_stats.csv",
        ["callback_learner_step", "step", "environment/succeed"],
        [
            {"callback_learner_step": 4998, "step": 10, "environment/succeed": 0},
            {"callback_learner_step": 5000, "step": 20, "environment/succeed": 1},
            {"callback_learner_step": 5002, "step": 0, "environment/succeed": 1},
        ],
    )

    results = prune_training_csv_logs(tmp_path, max_step=5000)

    actor_result = next(r for r in results if r.path.name == "actor_stats.csv")
    assert actor_result.step_column == "callback_learner_step"
    assert actor_result.removed_rows == 1
    remaining = _read_rows(tmp_path / "actor_stats.csv")
    assert [row["callback_learner_step"] for row in remaining] == ["4998", "5000"]


def test_prune_csv_rows_after_step_skips_missing_step_column(tmp_path: Path) -> None:
    path = tmp_path / "actor_stats.csv"
    _write_csv(
        path,
        ["environment/succeed"],
        [{"environment/succeed": 1}],
    )

    result = prune_csv_rows_after_step(
        path,
        max_step=5000,
        step_columns=("callback_learner_step", "step"),
    )

    assert result.reason == "step_column_missing"
    assert result.removed_rows == 0
    assert _read_rows(path) == [{"environment/succeed": "1"}]


def test_prune_csv_rows_on_backward_jumps_removes_stale_overlap(tmp_path: Path) -> None:
    path = tmp_path / "learner_update_metrics.csv"
    _write_csv(
        path,
        ["step", "critic/critic_loss"],
        [
            {"step": 4900, "critic/critic_loss": 0.5},
            {"step": 5000, "critic/critic_loss": 0.4},
            {"step": 5100, "critic/critic_loss": 0.3},
            {"step": 5200, "critic/critic_loss": 0.2},
            {"step": 5000, "critic/critic_loss": 0.35},
            {"step": 5100, "critic/critic_loss": 0.25},
        ],
    )

    result = prune_csv_rows_on_backward_jumps(
        path,
        step_columns=("step",),
    )

    assert result.step_column == "step"
    assert result.detected_jumps == 1
    assert result.removed_rows == 2
    assert [row["step"] for row in _read_rows(path)] == ["4900", "5000", "5000", "5100"]


def test_prune_training_csv_logs_auto_detect_handles_multiple_restarts(
    tmp_path: Path,
) -> None:
    _write_csv(
        tmp_path / "actor_stats.csv",
        ["callback_learner_step", "environment/succeed"],
        [
            {"callback_learner_step": 100, "environment/succeed": 1},
            {"callback_learner_step": 120, "environment/succeed": 1},
            {"callback_learner_step": 140, "environment/succeed": 0},
            {"callback_learner_step": 110, "environment/succeed": 1},
            {"callback_learner_step": 130, "environment/succeed": 1},
            {"callback_learner_step": 90, "environment/succeed": 0},
            {"callback_learner_step": 100, "environment/succeed": 1},
        ],
    )

    results = prune_training_csv_logs(tmp_path)

    actor_result = next(r for r in results if r.path.name == "actor_stats.csv")
    assert actor_result.step_column == "callback_learner_step"
    assert actor_result.detected_jumps == 2
    assert actor_result.removed_rows == 5
    assert [row["callback_learner_step"] for row in _read_rows(tmp_path / "actor_stats.csv")] == [
        "90",
        "100",
    ]
