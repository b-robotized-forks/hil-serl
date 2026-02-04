"""Helpers for pruning resumed training CSV logs to a checkpoint step."""

from dataclasses import dataclass
import csv
from pathlib import Path


@dataclass(frozen=True)
class CSVPruneSpec:
    filename: str
    step_columns: tuple[str, ...]


@dataclass(frozen=True)
class CSVPruneResult:
    path: Path
    step_column: str | None
    total_rows: int
    kept_rows: int
    removed_rows: int
    applied: bool
    detected_jumps: int = 0
    reason: str | None = None


DEFAULT_PRUNE_SPECS: tuple[CSVPruneSpec, ...] = (
    CSVPruneSpec("learner_update_metrics.csv", ("step",)),
    CSVPruneSpec("learner_timer_metrics.csv", ("step",)),
    CSVPruneSpec("actor_stats.csv", ("callback_learner_step", "step")),
)


def _parse_step_value(value: object) -> float | None:
    """Parse a CSV step value as float, accepting ints and int-like strings."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def prune_csv_rows_after_step(
    path: str | Path,
    *,
    max_step: int,
    step_columns: tuple[str, ...],
    dry_run: bool = False,
) -> CSVPruneResult:
    """Delete rows whose selected step column is newer than ``max_step``."""
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return CSVPruneResult(
            path=csv_path,
            step_column=None,
            total_rows=0,
            kept_rows=0,
            removed_rows=0,
            applied=False,
            detected_jumps=0,
            reason="missing_or_empty",
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    step_column = next((col for col in step_columns if col in fieldnames), None)
    if step_column is None:
        return CSVPruneResult(
            path=csv_path,
            step_column=None,
            total_rows=len(rows),
            kept_rows=len(rows),
            removed_rows=0,
            applied=False,
            detected_jumps=0,
            reason="step_column_missing",
        )

    kept_rows: list[dict[str, str]] = []
    removed_rows = 0
    for row in rows:
        step_value = _parse_step_value(row.get(step_column))
        if step_value is not None and step_value > float(max_step):
            removed_rows += 1
            continue
        kept_rows.append(row)

    if removed_rows > 0 and not dry_run:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(kept_rows)

    return CSVPruneResult(
        path=csv_path,
        step_column=step_column,
        total_rows=len(rows),
        kept_rows=len(kept_rows),
        removed_rows=removed_rows,
        applied=removed_rows > 0 and not dry_run,
        detected_jumps=0,
        reason=None,
    )


def prune_csv_rows_on_backward_jumps(
    path: str | Path,
    *,
    step_columns: tuple[str, ...],
    dry_run: bool = False,
) -> CSVPruneResult:
    """Delete stale rows by detecting strict backward jumps in learner step."""
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return CSVPruneResult(
            path=csv_path,
            step_column=None,
            total_rows=0,
            kept_rows=0,
            removed_rows=0,
            applied=False,
            detected_jumps=0,
            reason="missing_or_empty",
        )

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    step_column = next((col for col in step_columns if col in fieldnames), None)
    if step_column is None:
        return CSVPruneResult(
            path=csv_path,
            step_column=None,
            total_rows=len(rows),
            kept_rows=len(rows),
            removed_rows=0,
            applied=False,
            detected_jumps=0,
            reason="step_column_missing",
        )

    kept_rows: list[dict[str, str]] = []
    kept_step_values: list[float | None] = []
    removed_rows = 0
    detected_jumps = 0
    last_seen_step: float | None = None

    for row in rows:
        step_value = _parse_step_value(row.get(step_column))
        if (
            step_value is not None
            and last_seen_step is not None
            and step_value < last_seen_step
        ):
            detected_jumps += 1
            while kept_rows and kept_step_values[-1] is not None and kept_step_values[-1] > step_value:
                kept_rows.pop()
                kept_step_values.pop()
                removed_rows += 1
        kept_rows.append(row)
        kept_step_values.append(step_value)
        if step_value is not None:
            last_seen_step = step_value

    if removed_rows > 0 and not dry_run:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(kept_rows)

    return CSVPruneResult(
        path=csv_path,
        step_column=step_column,
        total_rows=len(rows),
        kept_rows=len(kept_rows),
        removed_rows=removed_rows,
        applied=removed_rows > 0 and not dry_run,
        detected_jumps=detected_jumps,
        reason=None,
    )


def prune_training_csv_logs(
    csv_dir: str | Path,
    *,
    max_step: int | None = None,
    dry_run: bool = False,
    specs: tuple[CSVPruneSpec, ...] = DEFAULT_PRUNE_SPECS,
) -> list[CSVPruneResult]:
    """Prune the standard training CSVs under ``csv_dir``."""
    base = Path(csv_dir)
    results: list[CSVPruneResult] = []
    for spec in specs:
        path = base / spec.filename
        if max_step is None:
            result = prune_csv_rows_on_backward_jumps(
                path,
                step_columns=spec.step_columns,
                dry_run=dry_run,
            )
        else:
            result = prune_csv_rows_after_step(
                path,
                max_step=max_step,
                step_columns=spec.step_columns,
                dry_run=dry_run,
            )
        results.append(result)
    return results
