"""CSV metrics logger for training loops."""

import csv
import os
import threading


class CSVLogger:
    """Lightweight logger that appends flattened metrics to a CSV file.

    New columns are added automatically as they appear. The header row is
    written once on the first ``log()`` call and rewritten (with the file
    contents preserved) whenever the schema grows. Appending to an existing
    file continues its schema instead of duplicating the header.
    """

    def __init__(self, path: str):
        self._path = path
        self._fieldnames: list[str] = []
        self._header_written = False
        self._lock = threading.Lock()
        # If the file already exists, read its header so we don't duplicate it.
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                self._fieldnames = list(reader.fieldnames or [])
                self._header_written = True
        print(f"CSVLogger: writing to {path}")

    def log(self, data: dict, step: int | None = None):
        """Append one row of (possibly nested) metrics, optionally keyed by step."""
        flat = self._flatten(data)
        if step is not None:
            flat = {"step": step, **flat}
        for k, v in flat.items():
            if isinstance(v, (dict, list, tuple)):
                flat[k] = str(v)

        with self._lock:
            new_keys = [k for k in flat if k not in self._fieldnames]
            if new_keys:
                self._fieldnames.extend(new_keys)
                if self._header_written:
                    # Rewrite the whole file with the expanded header.
                    self._rewrite_with_new_header()
                else:
                    self._write_header()
                    self._header_written = True

            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
                writer.writerow(flat)

    def _write_header(self):
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()

    def _rewrite_with_new_header(self):
        """Rewrite the file with an expanded header, preserving all existing rows."""
        rows = []
        if os.path.exists(self._path):
            with open(self._path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        with open(self._path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _flatten(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
            if isinstance(v, dict):
                out.update(CSVLogger._flatten(v, key))
            else:
                out[key] = v
        return out
