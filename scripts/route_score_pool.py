#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from nearfar_relation_tensor_asd.metrics import group_auc_hmean
from nearfar_relation_tensor_asd.routing import restricted_group_routing


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Development-protocol score-pool routing from a score manifest.")
    p.add_argument("--score-manifest", required=True)
    p.add_argument("--base-column", required=True)
    p.add_argument("--score-columns", required=True, help="Comma-separated tensor score columns.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alpha-grid", default="0,0.075,0.15,0.225")
    return p.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r}) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.score_manifest)
    base = np.asarray([float(r[args.base_column]) for r in rows], dtype=np.float64)
    score_pool = {c: np.asarray([float(r[c]) for r in rows], dtype=np.float64) for c in args.score_columns.split(",")}
    alpha_grid = tuple(float(x) for x in args.alpha_grid.split(","))
    routed, choices = restricted_group_routing(rows, base, score_pool, alpha_grid=alpha_grid)
    aucs, hm = group_auc_hmean(rows, routed)
    out_dir = Path(args.out_dir)
    write_csv(
        out_dir / "routing_choices.csv",
        [{"group": c.group, "score_id": c.score_id, "alpha": c.alpha, "group_hmean": c.hmean} for c in choices],
    )
    write_csv(out_dir / "group_auc.csv", [{"group": g, "auc": v} for g, v in sorted(aucs.items())])
    write_csv(out_dir / "summary.csv", [{"hmean": hm, "groups": len(aucs), "routing": "development_protocol"}])


if __name__ == "__main__":
    main()
