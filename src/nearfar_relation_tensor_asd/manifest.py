from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = ["basename", "near_path", "far_path", "machine_type", "domain", "split"]


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [c for c in REQUIRED_COLUMNS if rows and c not in rows[0]]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return rows


def write_manifest(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fieldnames or sorted({k for r in rows for k in r})
    with p.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
