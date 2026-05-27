#!/usr/bin/env python
"""Stage 18E: local-density frozen protocol full7 audit.

This script is strictly score-level and reads existing EAT min_clamp__mean
embeddings only.

It does NOT:
- load EAT
- load any pretrained model
- run model forward
- extract embeddings
- train models
- fine-tune
- run LoRA / adapter
- run Conformer
- search alpha
- do per-machine selection
- use oracle routing

It only:
- loads existing embedding npz files
- computes fixed local-density score candidates
- reports full7 metrics and verdict
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]

EMBED_ROOT_DEFAULT = "results/stage16d_minclamp_mean_full7/embeddings"
OUTPUT_ROOT_DEFAULT = "results/stage18e_local_density_protocol_full7"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18E_LOCAL_DENSITY_PROTOCOL_FULL7_AUDIT.md"

REGISTERED_BASELINE_MEAN_HMEAN = 0.604713
REGISTERED_CURRENT_BEST_MEAN_HMEAN = 0.607175
EPS = 1e-12

CANDIDATES = [
    "baseline_knn_k5",
    "ldn_ratio_K1_scorek5",
    "ldn_diff_K1_scorek5",
    "ldn_ratio_K5_scorek5",
    "ldn_diff_K5_scorek5",
    "gwrp_ratio_r09_scorek5",
    "gwrp_diff_r09_scorek5",
    "rasdsr_ratio_avg_K5_r09_scorek5",
]


def parse_args():
    p = argparse.ArgumentParser(description="Stage18E local-density frozen protocol full7 audit")
    p.add_argument("--mode", required=True, choices=["check", "score_ldn"])
    p.add_argument("--embedding_root", default=EMBED_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
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
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def import_status():
    status = {}
    for name in ["numpy", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def embedding_npz_path(embedding_root: Path, machine: str) -> Path:
    return embedding_root / f"{machine}_minclamp_mean_embeddings.npz"


def load_embedding_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    required = ["train_embeddings", "test_embeddings", "y_test", "d_test", "test_paths"]
    missing = [k for k in required if k not in arr.files]
    if missing:
        raise KeyError(f"{npz_path} missing keys: {missing}; keys={arr.files}")
    return {
        "path": str(npz_path),
        "train_embeddings": arr["train_embeddings"].astype(np.float64),
        "test_embeddings": arr["test_embeddings"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "d_test": arr["d_test"].astype(str),
        "test_paths": arr["test_paths"].astype(str),
    }


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), EPS)


def chord_distance_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = l2_normalize(query.astype(np.float64), axis=1)
    r = l2_normalize(ref.astype(np.float64), axis=1)
    qn2 = np.sum(q * q, axis=1, keepdims=True)
    rn2 = np.sum(r * r, axis=1)[None, :]
    d2 = qn2 + rn2 - 2.0 * (q @ r.T)
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2).astype(np.float64)


def baseline_knn_scores(dist_matrix: np.ndarray, score_k: int = 5) -> np.ndarray:
    score_k = min(score_k, dist_matrix.shape[1])
    nearest = np.partition(dist_matrix, kth=score_k - 1, axis=1)[:, :score_k]
    return nearest.mean(axis=1).astype(np.float64)


def local_density_knn(ref_ref_dist: np.ndarray, K: int) -> np.ndarray:
    if ref_ref_dist.shape[0] <= 1:
        return np.zeros((ref_ref_dist.shape[0],), dtype=np.float64)
    K = min(K, ref_ref_dist.shape[1] - 1)
    d = ref_ref_dist.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.partition(d, kth=K - 1, axis=1)[:, :K]
    return nearest.mean(axis=1).astype(np.float64)


def local_density_gwrp(ref_ref_dist: np.ndarray, r: float = 0.9) -> np.ndarray:
    if ref_ref_dist.shape[0] <= 1:
        return np.zeros((ref_ref_dist.shape[0],), dtype=np.float64)
    d = ref_ref_dist.copy()
    np.fill_diagonal(d, np.inf)
    sorted_d = np.sort(d, axis=1)
    sorted_d = sorted_d[:, :-1]
    n = sorted_d.shape[1]
    weights = np.power(r, np.arange(n, dtype=np.float64))
    denom = float(np.sum(weights))
    return ((sorted_d * weights[None, :]).sum(axis=1) / denom).astype(np.float64)


def rank_norm(scores: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray) -> dict:
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95)) if len(normals) else 0.0
    q99 = float(np.quantile(normals, 0.99)) if len(normals) else 0.0
    anomaly_median = float(np.quantile(anomalies, 0.50)) if len(anomalies) else 0.0
    anomaly_q75 = float(np.quantile(anomalies, 0.75)) if len(anomalies) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": anomaly_median,
        "anomaly_q75": anomaly_q75,
        "TPR@95n": float(np.mean(anomalies > q95)) if len(anomalies) else 0.0,
        "TPR@99n": float(np.mean(anomalies > q99)) if len(anomalies) else 0.0,
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.score_geometry import compute_score_geometry

    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    summary = {
        "hmean": float(met["hmean"]),
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


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage18E Local Density Protocol Full7 Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no scoring",
        "- no model load",
        "- no forward",
        "- no embedding extraction",
        "",
        "## Status",
        "",
        f"- embedding root exists: `{status['embedding_root_exists']}`",
        f"- all machine embeddings present: `{status['all_machine_embeddings_present']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_score_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage18E Local Density Protocol Full7 Audit",
        "",
        "## Best Candidate",
        "",
        f"- best candidate: `{verdict['best_candidate']}`",
        f"- mean hmean: `{verdict['best_mean_hmean']:.6f}`",
        f"- mean pAUC: `{verdict['best_mean_pAUC']:.6f}`",
        f"- mean TPR@99n: `{verdict['best_mean_TPR@99n']:.6f}`",
        f"- improved machine count vs baseline: `{verdict['improved_machine_count_vs_baseline']}`",
        f"- strong-machine severe collapse: `{verdict['strong_machine_severe_collapse']}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- very_strong: `{verdict['very_strong']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- fixed candidates only",
        "- no per-machine best selection",
        "- no selector",
        "- no oracle",
        "- no fusion with A1 / BEATs / ATST / PaSST",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    embedding_root = Path(args.embedding_root)
    machines = parse_csv_list(args.machines)

    rows = []
    all_present = True
    for machine in machines:
        npz_path = embedding_npz_path(embedding_root, machine)
        exists = npz_path.exists()
        keys_ok = False
        keys = []
        if exists:
            arr = np.load(npz_path, allow_pickle=True)
            keys = list(arr.files)
            keys_ok = all(k in arr.files for k in ["train_embeddings", "test_embeddings", "y_test", "d_test", "test_paths"])
        all_present = all_present and exists and keys_ok
        rows.append({
            "machine": machine,
            "path": str(npz_path),
            "exists": exists,
            "keys_ok": keys_ok,
            "keys": ",".join(keys),
        })

    status = {
        "mode": "check",
        "embedding_root": str(embedding_root),
        "embedding_root_exists": embedding_root.exists(),
        "machines": machines,
        "all_machine_embeddings_present": all_present,
        "imports": import_status(),
        "notes": "No scoring, no model load, no forward, no embedding extraction.",
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "embedding_check.csv", rows)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_score_ldn(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    embedding_root = Path(args.embedding_root)
    machines = parse_csv_list(args.machines)
    score_k = int(args.latent_knn_k)

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    collapse_rows = []
    baseline_per_machine = {}

    for machine in machines:
        bundle = load_embedding_bundle(embedding_npz_path(embedding_root, machine))
        train_emb = bundle["train_embeddings"]
        test_emb = bundle["test_embeddings"]
        y_test = bundle["y_test"]
        d_test = bundle["d_test"]
        test_paths = bundle["test_paths"]

        test_ref_dist = chord_distance_matrix(test_emb, train_emb)
        ref_ref_dist = chord_distance_matrix(train_emb, train_emb)

        rho_k1 = local_density_knn(ref_ref_dist, K=1)
        rho_k5 = local_density_knn(ref_ref_dist, K=5)
        rho_gwrp = local_density_gwrp(ref_ref_dist, r=0.9)

        scores_dict = {
            "baseline_knn_k5": baseline_knn_scores(test_ref_dist, score_k=score_k),
            "ldn_ratio_K1_scorek5": baseline_knn_scores(test_ref_dist / (rho_k1[None, :] + EPS), score_k=score_k),
            "ldn_diff_K1_scorek5": baseline_knn_scores(test_ref_dist - rho_k1[None, :], score_k=score_k),
            "ldn_ratio_K5_scorek5": baseline_knn_scores(test_ref_dist / (rho_k5[None, :] + EPS), score_k=score_k),
            "ldn_diff_K5_scorek5": baseline_knn_scores(test_ref_dist - rho_k5[None, :], score_k=score_k),
            "gwrp_ratio_r09_scorek5": baseline_knn_scores(test_ref_dist / (rho_gwrp[None, :] + EPS), score_k=score_k),
            "gwrp_diff_r09_scorek5": baseline_knn_scores(test_ref_dist - rho_gwrp[None, :], score_k=score_k),
        }
        scores_dict["rasdsr_ratio_avg_K5_r09_scorek5"] = (
            0.5 * rank_norm(scores_dict["ldn_ratio_K5_scorek5"]) +
            0.5 * rank_norm(scores_dict["gwrp_ratio_r09_scorek5"])
        )

        for candidate in CANDIDATES:
            scores = scores_dict[candidate].astype(np.float64)
            summary, tail, geom = evaluate(scores, y_test, d_test)
            per_machine_rows.append({"candidate": candidate, "machine": machine, **summary})
            tail_rows.append({"candidate": candidate, "machine": machine, **tail})
            geom_rows.append({"candidate": candidate, "machine": machine, **geom})

            cand_dir = scores_root / candidate
            cand_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                cand_dir / f"{machine}_scores.npz",
                test_scores=scores,
                y_test=y_test,
                d_test=d_test.astype("U16"),
                test_paths=np.asarray(test_paths, dtype=object),
                candidate=candidate,
                machine=machine,
            )

            if candidate == "baseline_knn_k5":
                baseline_per_machine[machine] = summary

    summary_rows = []
    comparison_rows = []
    rank_rows = []
    for candidate in CANDIDATES:
        rows = [r for r in per_machine_rows if r["candidate"] == candidate]
        mean_h = float(np.mean([r["hmean"] for r in rows]))
        mean_p = float(np.mean([r["pAUC"] for r in rows]))
        mean_ks = float(np.mean([r["KS"] for r in rows]))
        mean_t95 = float(np.mean([r["TPR@95n"] for r in rows]))
        mean_t99 = float(np.mean([r["TPR@99n"] for r in rows]))

        improved_count = 0
        strong_collapse = False
        for row in rows:
            base = baseline_per_machine[row["machine"]]
            delta_h = row["hmean"] - base["hmean"]
            if delta_h > 0:
                improved_count += 1
            if row["machine"] in {"bearing", "toytrain", "valve"} and delta_h < -0.02:
                strong_collapse = True
            collapse_rows.append({
                "candidate": candidate,
                "machine": row["machine"],
                "strong_machine": row["machine"] in {"bearing", "toytrain", "valve"},
                "baseline_hmean": base["hmean"],
                "candidate_hmean": row["hmean"],
                "delta_hmean_vs_baseline": delta_h,
                "severe_collapse": bool(row["machine"] in {"bearing", "toytrain", "valve"} and delta_h < -0.02),
            })

        summary_rows.append({
            "candidate": candidate,
            "mean_hmean": mean_h,
            "mean_pAUC": mean_p,
            "mean_KS": mean_ks,
            "mean_TPR@95n": mean_t95,
            "mean_TPR@99n": mean_t99,
        })
        comparison_rows.append({
            "candidate": candidate,
            "mean_hmean": mean_h,
            "mean_pAUC": mean_p,
            "mean_TPR@99n": mean_t99,
            "delta_hmean_vs_baseline": mean_h - REGISTERED_BASELINE_MEAN_HMEAN,
            "delta_pAUC_vs_baseline": mean_p - next(r["mean_pAUC"] for r in summary_rows if r["candidate"] == "baseline_knn_k5") if candidate != "baseline_knn_k5" else 0.0,
            "delta_TPR99_vs_baseline": mean_t99 - next(r["mean_TPR@99n"] for r in summary_rows if r["candidate"] == "baseline_knn_k5") if candidate != "baseline_knn_k5" else 0.0,
            "improved_machine_count_vs_baseline": improved_count,
            "strong_machine_severe_collapse": strong_collapse,
        })

    baseline_summary = next(r for r in summary_rows if r["candidate"] == "baseline_knn_k5")
    for row in comparison_rows:
        if row["candidate"] != "baseline_knn_k5":
            row["delta_pAUC_vs_baseline"] = row["mean_pAUC"] - baseline_summary["mean_pAUC"]
            row["delta_TPR99_vs_baseline"] = row["mean_TPR@99n"] - baseline_summary["mean_TPR@99n"]

    ranked = sorted(summary_rows, key=lambda r: r["mean_hmean"], reverse=True)
    for idx, row in enumerate(ranked, start=1):
        rank_rows.append({
            "rank": idx,
            "candidate": row["candidate"],
            "mean_hmean": row["mean_hmean"],
            "mean_pAUC": row["mean_pAUC"],
            "mean_TPR@99n": row["mean_TPR@99n"],
        })

    best = ranked[0]
    best_comp = next(r for r in comparison_rows if r["candidate"] == best["candidate"])
    no_candidate_exceeds_baseline = not any(r["mean_hmean"] > baseline_summary["mean_hmean"] for r in summary_rows if r["candidate"] != "baseline_knn_k5")
    gains_only_one_machine = best_comp["improved_machine_count_vs_baseline"] <= 1
    strong_collapse = bool(best_comp["strong_machine_severe_collapse"])
    promising = bool(best["mean_hmean"] > REGISTERED_BASELINE_MEAN_HMEAN and not strong_collapse)
    strong_positive = bool(
        best["mean_hmean"] > REGISTERED_CURRENT_BEST_MEAN_HMEAN
        and best_comp["improved_machine_count_vs_baseline"] >= 4
        and not strong_collapse
    )
    very_strong = bool(best["mean_hmean"] >= 0.610000 and not strong_collapse)
    negative = bool(no_candidate_exceeds_baseline or gains_only_one_machine or strong_collapse)

    verdict = {
        "best_candidate": best["candidate"],
        "best_mean_hmean": best["mean_hmean"],
        "best_mean_pAUC": best["mean_pAUC"],
        "best_mean_TPR@99n": best["mean_TPR@99n"],
        "baseline_mean_hmean": baseline_summary["mean_hmean"],
        "registered_baseline_mean_hmean": REGISTERED_BASELINE_MEAN_HMEAN,
        "registered_current_best_mean_hmean": REGISTERED_CURRENT_BEST_MEAN_HMEAN,
        "improved_machine_count_vs_baseline": best_comp["improved_machine_count_vs_baseline"],
        "strong_machine_severe_collapse": strong_collapse,
        "promising": promising,
        "strong_positive": strong_positive,
        "very_strong": very_strong,
        "negative": negative,
        "notes": [
            "Fixed local-density candidates only.",
            "No per-machine best selection.",
            "No fusion with external routes.",
            "Human review required before any promotion.",
        ],
    }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "comparison_vs_baseline.csv", comparison_rows)
    write_csv(out_root / "candidate_rank.csv", rank_rows)
    write_csv(out_root / "collapse_check.csv", collapse_rows)
    write_json(out_root / "stage18e_ldn_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_score_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "score_ldn":
        run_score_ldn(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
