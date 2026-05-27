#!/usr/bin/env python
"""Stage57A: current champion vs V61 cross-year score complementarity audit.

This is a read-only audit. It aligns the current Stage53 champion score files
with V61 cross-year original-encoder score files and evaluates fixed candidates:

- current champion score
- V61 raw score
- 0.5/0.5 per-machine rank average

No training, score extraction, fusion weight tuning, per-machine selection, or
checkpoint selection is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry

from audit_stage21e_adapted_eval import tail_metrics


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
CHAMPION_ROOT_DEFAULT = "results/stage53a_stage52_w006_e005_domain_calibrated_score"
CHAMPION_CANDIDATE_DEFAULT = "adapted_ldn_diff_K1_scorek5_domain_q95_ratio"
V61_ROOT_DEFAULT = "results/v61_cross_year_original_encoder_seedpanel/seed_13711"
OUTPUT_ROOT_DEFAULT = "results/stage57a_champion_v61_complementarity"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE57A_CHAMPION_V61_COMPLEMENTARITY.md"


def parse_args():
    p = argparse.ArgumentParser(description="Stage57A champion/V61 complementarity audit")
    p.add_argument("--mode", required=True, choices=["audit_complementarity"])
    p.add_argument("--champion_root", default=CHAMPION_ROOT_DEFAULT)
    p.add_argument("--champion_candidate", default=CHAMPION_CANDIDATE_DEFAULT)
    p.add_argument("--v61_root", default=V61_ROOT_DEFAULT)
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    return p.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
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


def rank_norm(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    if len(scores) <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    return ranks / float(len(scores) - 1)


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    return {
        "hmean": float(met["hmean"]),
        "auc_source": float(met["auc_source"]),
        "auc_target": float(met["auc_target"]),
        "pAUC": float(met["pauc"]),
        "KS": float(geom["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
    }


def champion_score_path(root: Path, candidate: str, machine: str) -> Path:
    return root / "scores" / candidate / f"{machine}_scores.npz"


def v61_score_path(root: Path, machine: str) -> Path:
    return root / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"


def load_pair(champion_root: Path, champion_candidate: str, v61_root: Path, machine: str):
    c = np.load(champion_score_path(champion_root, champion_candidate, machine), allow_pickle=True)
    v = np.load(v61_score_path(v61_root, machine), allow_pickle=True)
    c_scores = c["test_scores"].astype(np.float64)
    v_scores = v["test_scores"].astype(np.float64)
    c_y = c["y_test"].astype(np.int64)
    v_y = v["y_test"].astype(np.int64)
    c_d = c["d_test"].astype(str)
    v_d = v["domains"].astype(str)
    c_paths = c["test_paths"].astype(str)
    v_paths = v["test_paths"].astype(str)
    aligned = bool(
        np.array_equal(c_y, v_y)
        and np.array_equal(c_d, v_d)
        and np.array_equal(c_paths, v_paths)
    )
    return c_scores, v_scores, c_y, c_d, c_paths, aligned


def complementarity_row(machine: str, c_scores: np.ndarray, v_scores: np.ndarray, y_test: np.ndarray) -> dict:
    normals = y_test == 0
    anomalies = y_test == 1
    c_q99 = float(np.quantile(c_scores[normals], 0.99))
    v_q99 = float(np.quantile(v_scores[normals], 0.99))
    c_top_norm = set(np.where(normals & (c_scores >= c_q99))[0].tolist())
    v_top_norm = set(np.where(normals & (v_scores >= v_q99))[0].tolist())
    c_missed = set(np.where(anomalies & (c_scores <= c_q99))[0].tolist())
    v_missed = set(np.where(anomalies & (v_scores <= v_q99))[0].tolist())
    detected_by_c_only = set(np.where(anomalies & (c_scores > c_q99) & (v_scores <= v_q99))[0].tolist())
    detected_by_v_only = set(np.where(anomalies & (v_scores > v_q99) & (c_scores <= c_q99))[0].tolist())

    def jaccard(a: set[int], b: set[int]) -> float:
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    rho = spearmanr(c_scores, v_scores).correlation
    return {
        "machine": machine,
        "spearman_r": float(rho) if rho == rho else 0.0,
        "top_q99_normal_jaccard": float(jaccard(c_top_norm, v_top_norm)),
        "missed_anomaly_jaccard": float(jaccard(c_missed, v_missed)),
        "champion_detect_v61_miss_count": int(len(detected_by_c_only)),
        "v61_detect_champion_miss_count": int(len(detected_by_v_only)),
        "champion_missed_count": int(len(c_missed)),
        "v61_missed_count": int(len(v_missed)),
    }


def run_audit_complementarity(args):
    machines = parse_csv_list(args.machines)
    champion_root = Path(args.champion_root)
    v61_root = Path(args.v61_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = [
        "champion_q95",
        "v61_raw",
        "fixed_rank_avg_50_50",
    ]
    per_machine_rows = []
    comp_rows = []
    alignment_rows = []

    for machine in machines:
        c_scores, v_scores, y_test, domains, paths, aligned = load_pair(
            champion_root,
            args.champion_candidate,
            v61_root,
            machine,
        )
        alignment_rows.append({
            "machine": machine,
            "aligned": aligned,
            "count": int(len(y_test)),
        })
        if not aligned:
            raise RuntimeError(f"Score alignment failed for machine={machine}")
        rank_avg = 0.5 * rank_norm(c_scores) + 0.5 * rank_norm(v_scores)
        score_map = {
            "champion_q95": c_scores,
            "v61_raw": v_scores,
            "fixed_rank_avg_50_50": rank_avg,
        }
        for candidate in candidates:
            per_machine_rows.append({
                "candidate": candidate,
                "machine": machine,
                **evaluate(score_map[candidate], y_test, domains),
            })
        comp_rows.append(complementarity_row(machine, c_scores, v_scores, y_test))

    summary_rows = []
    for candidate in candidates:
        rows = [r for r in per_machine_rows if r["candidate"] == candidate]
        summary_rows.append({
            "candidate": candidate,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_auc_source": float(np.mean([r["auc_source"] for r in rows])),
            "mean_auc_target": float(np.mean([r["auc_target"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })

    champion = next(r for r in summary_rows if r["candidate"] == "champion_q95")
    rank_avg = next(r for r in summary_rows if r["candidate"] == "fixed_rank_avg_50_50")
    verdict = {
        "mode": "audit_complementarity",
        "champion_root": str(champion_root),
        "champion_candidate": args.champion_candidate,
        "v61_root": str(v61_root),
        "alignment_pass": all(r["aligned"] for r in alignment_rows),
        "champion_mean_hmean": champion["mean_hmean"],
        "fixed_rank_avg_mean_hmean": rank_avg["mean_hmean"],
        "delta_hmean_rank_avg_vs_champion": rank_avg["mean_hmean"] - champion["mean_hmean"],
        "promotable": bool(rank_avg["mean_hmean"] > champion["mean_hmean"]),
        "negative": bool(rank_avg["mean_hmean"] <= champion["mean_hmean"]),
        "notes": [
            "Read-only score complementarity audit.",
            "No training, score extraction, fusion weight tuning, per-machine selection, or checkpoint selection.",
            "The fixed 50/50 rank average is an audit candidate, not a promoted route unless it beats the frozen champion.",
        ],
    }

    write_csv(out_root / "alignment_summary.csv", alignment_rows)
    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "complementarity.csv", comp_rows)
    write_json(out_root / "stage57a_complementarity_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage57A Champion/V61 Complementarity",
        "",
        f"- champion candidate: `{args.champion_candidate}`",
        f"- champion mean hmean: `{verdict['champion_mean_hmean']:.12f}`",
        f"- fixed rank avg mean hmean: `{verdict['fixed_rank_avg_mean_hmean']:.12f}`",
        f"- delta vs champion: `{verdict['delta_hmean_rank_avg_vs_champion']:.12f}`",
        f"- alignment pass: `{verdict['alignment_pass']}`",
        f"- promotable: `{verdict['promotable']}`",
        "",
        "No training, score extraction, fusion weight tuning, or per-machine selection was performed.",
    ])


def main():
    args = parse_args()
    if args.mode == "audit_complementarity":
        run_audit_complementarity(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
