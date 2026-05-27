#!/usr/bin/env python
"""Stage23B: fan density-failure audit across Stage23 ablations.

This is a read-only diagnostic. It compares fan baseline-vs-LDN score behavior
across adapted routes and checks whether fan degradation is caused by the frozen
LDN subtraction under adapted geometry rather than by one isolated loss term.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from audit_stage21e_adapted_eval import (
    baseline_knn_scores,
    chord_distance_matrix,
    local_density_knn,
)


OUTPUT_ROOT_DEFAULT = "results/stage23b_fan_density_failure"
DOC_PATH_DEFAULT = "docs/CTAF_STAGE23B_FAN_DENSITY_FAILURE.md"
ROUTES_DEFAULT = (
    "stage22b=results/stage22d_adapted_full7_eval,"
    "no_ldn_proxy=results/stage23a_no_ldn_proxy_pair_eval,"
    "no_aug=results/stage23b_no_aug_pair_eval,"
    "no_dispersion=results/stage23c_no_dispersion_pair_eval,"
    "stronger_neighbor=results/stage23d_stronger_neighbor_pair_eval"
)
MACHINE_DEFAULT = "fan"
BASELINE_CANDIDATE = "adapted_baseline_knn_k5"
LDN_CANDIDATE = "adapted_ldn_diff_K1_scorek5"


def parse_args():
    p = argparse.ArgumentParser(description="Stage23B fan density failure audit")
    p.add_argument("--routes", default=ROUTES_DEFAULT)
    p.add_argument("--machine", default=MACHINE_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=DOC_PATH_DEFAULT)
    p.add_argument("--score_k", type=int, default=5)
    return p.parse_args()


def parse_routes(text: str) -> list[tuple[str, Path]]:
    routes = []
    for item in [x.strip() for x in text.split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Route must be name=path, got: {item}")
        name, path = item.split("=", 1)
        routes.append((name.strip(), Path(path.strip())))
    return routes


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def find_row(rows: list[dict], candidate: str, machine: str) -> dict | None:
    for row in rows:
        if row.get("candidate") == candidate and row.get("machine") == machine:
            return row
    return None


def f(row: dict | None, key: str) -> float | None:
    if row is None:
        return None
    try:
        return float(row[key])
    except Exception:
        return None


def quantiles(x: np.ndarray, prefix: str) -> dict:
    if x.size == 0:
        return {f"{prefix}_{k}": None for k in ["mean", "std", "q50", "q95", "q99"]}
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_q50": float(np.quantile(x, 0.50)),
        f"{prefix}_q95": float(np.quantile(x, 0.95)),
        f"{prefix}_q99": float(np.quantile(x, 0.99)),
    }


def embedding_path(root: Path, machine: str) -> Path:
    return root / "embeddings" / f"{machine}_adapted_minclamp_mean_embeddings.npz"


def load_scores(root: Path, candidate: str, machine: str):
    path = root / "scores" / candidate / f"{machine}_scores.npz"
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    return {
        "scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "d_test": arr["d_test"].astype(str),
    }


def route_summary(name: str, root: Path, machine: str, score_k: int) -> dict:
    per_machine = read_csv(root / "per_machine.csv")
    base_row = find_row(per_machine, BASELINE_CANDIDATE, machine)
    ldn_row = find_row(per_machine, LDN_CANDIDATE, machine)
    emb = embedding_path(root, machine)

    row = {
        "route": name,
        "eval_root": str(root),
        "eval_available": bool(per_machine),
        "embedding_available": emb.exists(),
        "baseline_hmean": f(base_row, "hmean"),
        "ldn_hmean": f(ldn_row, "hmean"),
        "ldn_minus_baseline_hmean": None,
        "baseline_pAUC": f(base_row, "pAUC"),
        "ldn_pAUC": f(ldn_row, "pAUC"),
        "ldn_minus_baseline_pAUC": None,
        "baseline_TPR99": f(base_row, "TPR@99n"),
        "ldn_TPR99": f(ldn_row, "TPR@99n"),
        "ldn_minus_baseline_TPR99": None,
    }
    if row["baseline_hmean"] is not None and row["ldn_hmean"] is not None:
        row["ldn_minus_baseline_hmean"] = row["ldn_hmean"] - row["baseline_hmean"]
    if row["baseline_pAUC"] is not None and row["ldn_pAUC"] is not None:
        row["ldn_minus_baseline_pAUC"] = row["ldn_pAUC"] - row["baseline_pAUC"]
    if row["baseline_TPR99"] is not None and row["ldn_TPR99"] is not None:
        row["ldn_minus_baseline_TPR99"] = row["ldn_TPR99"] - row["baseline_TPR99"]

    base_scores = load_scores(root, BASELINE_CANDIDATE, machine)
    ldn_scores = load_scores(root, LDN_CANDIDATE, machine)
    if base_scores and ldn_scores:
        y = base_scores["y_test"]
        base = base_scores["scores"]
        ldn = ldn_scores["scores"]
        delta = ldn - base
        normal_delta = delta[y == 0]
        anomaly_delta = delta[y == 1]
        row.update(quantiles(normal_delta, "score_delta_ldn_minus_base_normal"))
        row.update(quantiles(anomaly_delta, "score_delta_ldn_minus_base_anomaly"))
        row["delta_gap_anomaly_minus_normal_mean"] = (
            float(np.mean(anomaly_delta) - np.mean(normal_delta))
            if normal_delta.size and anomaly_delta.size else None
        )

    if emb.exists():
        arr = np.load(emb, allow_pickle=True)
        train = arr["train_embeddings"].astype(np.float64)
        test = arr["test_embeddings"].astype(np.float64)
        y = arr["y_test"].astype(np.int64)
        test_ref = chord_distance_matrix(test, train)
        ref_ref = chord_distance_matrix(train, train)
        rho = local_density_knn(ref_ref, K=1)
        base = baseline_knn_scores(test_ref, score_k=score_k)
        ldn = baseline_knn_scores(test_ref - rho[None, :], score_k=score_k)
        row.update(quantiles(rho, "rho_K1_train"))
        row.update(quantiles(base[y == 0], "baseline_normal"))
        row.update(quantiles(ldn[y == 0], "ldn_normal"))
        row.update(quantiles(base[y == 1], "baseline_anomaly"))
        row.update(quantiles(ldn[y == 1], "ldn_anomaly"))
        row["normal_q99_margin_baseline"] = (
            float(np.quantile(base[y == 1], 0.50) - np.quantile(base[y == 0], 0.99))
            if np.any(y == 0) and np.any(y == 1) else None
        )
        row["normal_q99_margin_ldn"] = (
            float(np.quantile(ldn[y == 1], 0.50) - np.quantile(ldn[y == 0], 0.99))
            if np.any(y == 0) and np.any(y == 1) else None
        )
    return row


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = [route_summary(name, root, args.machine, args.score_k) for name, root in parse_routes(args.routes)]
    available = [r for r in rows if r["eval_available"]]
    worst_ldn_gap = min(
        available,
        key=lambda r: r["ldn_minus_baseline_hmean"] if r["ldn_minus_baseline_hmean"] is not None else 999,
    ) if available else None
    best_baseline = max(
        available,
        key=lambda r: r["baseline_hmean"] if r["baseline_hmean"] is not None else -1,
    ) if available else None

    all_ldn_worse = all((r["ldn_minus_baseline_hmean"] is not None and r["ldn_minus_baseline_hmean"] < 0) for r in available)
    verdict = {
        "machine": args.machine,
        "available_route_count": len(available),
        "all_routes_ldn_worse_than_adapted_baseline": all_ldn_worse,
        "worst_ldn_gap_route": worst_ldn_gap["route"] if worst_ldn_gap else None,
        "worst_ldn_gap_hmean": worst_ldn_gap["ldn_minus_baseline_hmean"] if worst_ldn_gap else None,
        "best_adapted_baseline_route": best_baseline["route"] if best_baseline else None,
        "best_adapted_baseline_hmean": best_baseline["baseline_hmean"] if best_baseline else None,
        "interpretation": (
            "If all routes show LDN hmean below adapted baseline on fan, fan harm is a frozen-density "
            "recalibration mismatch under adapted geometry rather than a single loss-component failure."
        ),
        "policy": {
            "no_ldn_parameter_tuning": True,
            "no_selector": True,
            "no_test_label_checkpoint_selection": True,
        },
    }

    write_csv(out_root / "stage23b_fan_density_failure_table.csv", rows)
    write_json(out_root / "stage23b_fan_density_failure_verdict.json", verdict)
    lines = [
        "# CTAF Stage23B Fan Density Failure Audit",
        "",
        f"- machine: `{args.machine}`",
        f"- all routes LDN worse than adapted baseline: `{all_ldn_worse}`",
        f"- worst LDN gap route: `{verdict['worst_ldn_gap_route']}`",
        f"- worst LDN gap hmean: `{verdict['worst_ldn_gap_hmean']}`",
        f"- best adapted baseline route: `{verdict['best_adapted_baseline_route']}`",
        f"- best adapted baseline hmean: `{verdict['best_adapted_baseline_hmean']}`",
        "",
        "## Route Table",
        "",
        "| route | baseline hmean | LDN hmean | LDN-baseline | normal q99 margin baseline | normal q99 margin LDN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(x):
            return "NA" if x is None else f"{x:.12f}"
        lines.append(
            f"| `{r['route']}` | {fmt(r['baseline_hmean'])} | {fmt(r['ldn_hmean'])} | "
            f"{fmt(r['ldn_minus_baseline_hmean'])} | {fmt(r.get('normal_q99_margin_baseline'))} | "
            f"{fmt(r.get('normal_q99_margin_ldn'))} |"
        )
    lines.extend([
        "",
        "No selector, LDN parameter tuning, or test-label checkpoint selection is introduced.",
    ])
    write_markdown(Path(args.doc_path), lines)


if __name__ == "__main__":
    main()
