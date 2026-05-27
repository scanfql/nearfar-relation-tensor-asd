#!/usr/bin/env python
"""Stage53A: unsupervised source/target domain calibration for target-memory scores.

This script reads Stage48A embedding NPZ files and scores fixed candidates that
do not change the frozen LDN K/r/score_k backend. The new candidates only
calibrate score scale by source/target domain using train-normal memory scores:

- raw LDN diff score (Stage48A reference)
- domain robust z-score over train-normal reference scores
- domain empirical CDF over train-normal reference scores

No model training, embedding extraction, selector, fusion, LDN tuning, anomaly
labels for calibration, or checkpoint selection is performed here. Test labels
are used only for final audit metrics.
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

from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry

from audit_stage21e_adapted_eval import (
    baseline_knn_scores,
    chord_distance_matrix,
    local_density_knn,
    tail_metrics,
)
from audit_stage48a_adapted_full7_eval_target_memory import (
    BASELINE_ROUTE,
    CONFIRMED_CLEAN_BEST_MEAN_HMEAN,
    CONFIRMED_CLEAN_BEST_MEAN_PAUC,
    CONFIRMED_CLEAN_BEST_MEAN_TPR99,
    FROZEN_ROUTE,
    MACHINES_ALL,
    emb_path,
    parse_csv_list,
    write_csv,
    write_json,
    write_markdown,
)


OUTPUT_ROOT_DEFAULT = "results/stage53a_target_memory_domain_calibrated_score"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE53A_TARGET_MEMORY_DOMAIN_CALIBRATED_SCORE.md"
EPS = 1e-12

RAW_LDN_ROUTE = FROZEN_ROUTE
Z_LDN_ROUTE = "adapted_ldn_diff_K1_scorek5_domain_robust_z"
ECDF_LDN_ROUTE = "adapted_ldn_diff_K1_scorek5_domain_ecdf"
Q95_LDN_ROUTE = "adapted_ldn_diff_K1_scorek5_domain_q95_ratio"


def parse_args():
    p = argparse.ArgumentParser(description="Stage53A target-memory domain calibrated scoring")
    p.add_argument("--mode", required=True, choices=["score_domain_calibrated"])
    p.add_argument("--embedding_root", required=True, help="Stage48A output root containing embeddings/*.npz.")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def score_from_dist(dist_matrix: np.ndarray, score_k: int) -> np.ndarray:
    return baseline_knn_scores(dist_matrix, score_k=score_k)


def train_reference_scores(raw_ref_dist: np.ndarray, score_k: int) -> tuple[np.ndarray, np.ndarray]:
    ref_ref_dist = raw_ref_dist.copy()
    rho_k1 = local_density_knn(ref_ref_dist, K=1)
    baseline_dist = ref_ref_dist.copy()
    ldn_dist = ref_ref_dist - rho_k1[None, :]
    np.fill_diagonal(baseline_dist, np.inf)
    np.fill_diagonal(ldn_dist, np.inf)
    return score_from_dist(baseline_dist, score_k), score_from_dist(ldn_dist, score_k)


def robust_z_by_domain(test_scores: np.ndarray, ref_scores: np.ndarray, test_domains: np.ndarray, ref_domains: np.ndarray) -> np.ndarray:
    out = np.zeros_like(test_scores, dtype=np.float64)
    global_med = float(np.median(ref_scores))
    global_iqr = float(np.percentile(ref_scores, 75) - np.percentile(ref_scores, 25))
    for domain in ["source", "target"]:
        ref = ref_scores[ref_domains == domain]
        med = float(np.median(ref)) if len(ref) else global_med
        iqr = float(np.percentile(ref, 75) - np.percentile(ref, 25)) if len(ref) else global_iqr
        mask = test_domains == domain
        out[mask] = (test_scores[mask] - med) / max(iqr, global_iqr * 0.25, EPS)
    return out


def ecdf_by_domain(test_scores: np.ndarray, ref_scores: np.ndarray, test_domains: np.ndarray, ref_domains: np.ndarray) -> np.ndarray:
    out = np.zeros_like(test_scores, dtype=np.float64)
    global_ref = np.sort(ref_scores.astype(np.float64))
    for domain in ["source", "target"]:
        ref = np.sort(ref_scores[ref_domains == domain].astype(np.float64))
        if len(ref) == 0:
            ref = global_ref
        mask = test_domains == domain
        ranks = np.searchsorted(ref, test_scores[mask], side="right")
        out[mask] = ranks.astype(np.float64) / max(len(ref), 1)
    return out


def q95_ratio_by_domain(test_scores: np.ndarray, ref_scores: np.ndarray, test_domains: np.ndarray, ref_domains: np.ndarray) -> np.ndarray:
    out = np.zeros_like(test_scores, dtype=np.float64)
    global_q95 = float(np.percentile(ref_scores, 95))
    for domain in ["source", "target"]:
        ref = ref_scores[ref_domains == domain]
        q95 = float(np.percentile(ref, 95)) if len(ref) else global_q95
        mask = test_domains == domain
        out[mask] = test_scores[mask] / max(q95, global_q95 * 0.25, EPS)
    return out


def evaluate_scores(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    summary = {
        "hmean": float(met["hmean"]),
        "auc_source": float(met["auc_source"]),
        "auc_target": float(met["auc_target"]),
        "pAUC": float(met["pauc"]),
        "KS": float(geom["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "anomaly_median": float(tail["anomaly_median"]),
        "anomaly_q75": float(tail["anomaly_q75"]),
    }
    return summary, tail, geom


def score_machine(machine: str, embedding_root: Path, score_k: int):
    arr = np.load(emb_path(embedding_root, machine), allow_pickle=True)
    train_emb = arr["train_embeddings"].astype(np.float64)
    test_emb = arr["test_embeddings"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)
    d_test = arr["d_test"].astype(str)
    train_domains = arr["train_domains"].astype(str)
    test_paths = arr["test_paths"].astype(str)

    test_ref_dist = chord_distance_matrix(test_emb, train_emb)
    ref_ref_dist = chord_distance_matrix(train_emb, train_emb)
    rho_k1 = local_density_knn(ref_ref_dist, K=1)

    train_baseline, train_ldn = train_reference_scores(ref_ref_dist, score_k)
    test_baseline = score_from_dist(test_ref_dist, score_k)
    test_ldn = score_from_dist(test_ref_dist - rho_k1[None, :], score_k)

    scores = {
        BASELINE_ROUTE: test_baseline,
        RAW_LDN_ROUTE: test_ldn,
        Z_LDN_ROUTE: robust_z_by_domain(test_ldn, train_ldn, d_test, train_domains),
        ECDF_LDN_ROUTE: ecdf_by_domain(test_ldn, train_ldn, d_test, train_domains),
        Q95_LDN_ROUTE: q95_ratio_by_domain(test_ldn, train_ldn, d_test, train_domains),
    }
    calibration_rows = []
    for domain in ["source", "target"]:
        ref = train_ldn[train_domains == domain]
        calibration_rows.append({
            "machine": machine,
            "domain": domain,
            "train_normal_count": int(len(ref)),
            "train_ldn_median": float(np.median(ref)) if len(ref) else 0.0,
            "train_ldn_iqr": float(np.percentile(ref, 75) - np.percentile(ref, 25)) if len(ref) else 0.0,
            "train_ldn_q95": float(np.percentile(ref, 95)) if len(ref) else 0.0,
        })
    return scores, y_test, d_test, test_paths, calibration_rows


def run_score_domain_calibrated(args):
    embedding_root = Path(args.embedding_root)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    missing = [str(emb_path(embedding_root, m)) for m in machines if not emb_path(embedding_root, m).exists()]
    if missing:
        raise FileNotFoundError(f"Missing embeddings: {missing}")

    candidates = [BASELINE_ROUTE, RAW_LDN_ROUTE, Z_LDN_ROUTE, ECDF_LDN_ROUTE, Q95_LDN_ROUTE]
    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    calibration_rows = []
    scores_root = out_root / "scores"

    for machine in machines:
        scores_dict, y_test, d_test, test_paths, cal_rows = score_machine(machine, embedding_root, args.latent_knn_k)
        calibration_rows.extend(cal_rows)
        for candidate in candidates:
            scores = scores_dict[candidate]
            summary, tail, geom = evaluate_scores(scores, y_test, d_test)
            per_machine_rows.append({"candidate": candidate, "machine": machine, **summary})
            tail_rows.append({"candidate": candidate, "machine": machine, **tail})
            geom_rows.append({"candidate": candidate, "machine": machine, **geom})
            cand_dir = scores_root / candidate
            cand_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cand_dir / f"{machine}_scores.npz",
                test_scores=scores.astype(np.float64),
                y_test=y_test,
                d_test=d_test.astype("U16"),
                test_paths=np.asarray(test_paths, dtype=object),
                candidate=candidate,
                machine=machine,
            )

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

    raw = next(r for r in summary_rows if r["candidate"] == RAW_LDN_ROUTE)
    best = max(summary_rows, key=lambda r: r["mean_hmean"])
    verdict = {
        "route": best["candidate"],
        "best_candidate": best["candidate"],
        "mean_hmean": best["mean_hmean"],
        "mean_pAUC": best["mean_pAUC"],
        "mean_TPR99": best["mean_TPR@99n"],
        "raw_ldn_mean_hmean": raw["mean_hmean"],
        "raw_ldn_mean_pAUC": raw["mean_pAUC"],
        "delta_hmean_vs_raw_ldn": best["mean_hmean"] - raw["mean_hmean"],
        "confirmed_clean_best_mean_hmean": CONFIRMED_CLEAN_BEST_MEAN_HMEAN,
        "confirmed_clean_best_mean_pAUC": CONFIRMED_CLEAN_BEST_MEAN_PAUC,
        "confirmed_clean_best_mean_TPR99": CONFIRMED_CLEAN_BEST_MEAN_TPR99,
        "uses_anomaly_labels_for_calibration": False,
        "uses_test_labels_for_selection": False,
        "ldn_parameters_tuned": False,
        "notes": [
            "Reads fixed Stage48A embeddings only; no extraction or training.",
            "Domain calibration uses train-normal memory scores and test domain metadata only.",
            "LDN K/r/score_k are unchanged.",
            "The best candidate field is an audit summary; do not treat it as a train-time selector.",
        ],
    }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "domain_calibration_stats.csv", calibration_rows)
    write_json(out_root / "stage53a_domain_calibrated_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage53A Target Memory Domain Calibrated Score",
        "",
        f"- embedding root: `{embedding_root}`",
        f"- best candidate: `{verdict['best_candidate']}`",
        f"- mean hmean: `{verdict['mean_hmean']:.12f}`",
        f"- mean pAUC: `{verdict['mean_pAUC']:.12f}`",
        f"- mean TPR@99n: `{verdict['mean_TPR99']:.12f}`",
        f"- raw LDN mean hmean: `{verdict['raw_ldn_mean_hmean']:.12f}`",
        f"- delta vs raw LDN: `{verdict['delta_hmean_vs_raw_ldn']:.12f}`",
        "",
        "No training, selector, fusion, LDN tuning, anomaly-label calibration, or checkpoint selection is performed.",
    ])


def main():
    args = parse_args()
    if args.mode == "score_domain_calibrated":
        run_score_domain_calibrated(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
