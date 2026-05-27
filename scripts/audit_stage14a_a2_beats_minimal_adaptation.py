#!/usr/bin/env python
"""Stage 14A-A2: BEATs minimal normal-only adaptation audit (PCA/whitening only)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 14A-A2 minimal BEATs adaptation audit")
    p.add_argument("--mode", required=True, choices=["check", "a2_pca_beats", "a2_v63_pca_beats_fusion"])
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--a0_root", default="results/stage14a_frozen_beats_v63_fusion/a0_frozen_beats")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--v37_root", default="results/v37_teacher_student_pretrain_seedpanel")
    p.add_argument("--v61_root", default="results/v61_cross_year_original_encoder_seedpanel")
    p.add_argument("--seeds", default="13711,13721,13731")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--pca_dim", type=int, default=256)
    return p.parse_args()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def tail_stats(scores: np.ndarray, y_true: np.ndarray) -> dict:
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    q95 = float(np.percentile(normal_scores, 95)) if len(normal_scores) else 0.0
    q99 = float(np.percentile(normal_scores, 99)) if len(normal_scores) else 0.0
    tpr95 = float(np.mean(anomaly_scores > q95)) if len(anomaly_scores) else 0.0
    tpr99 = float(np.mean(anomaly_scores > q99)) if len(anomaly_scores) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "TPR@95n": tpr95,
        "TPR@99n": tpr99,
        "anomaly_median": float(np.median(anomaly_scores)) if len(anomaly_scores) else 0.0,
        "anomaly_q75": float(np.percentile(anomaly_scores, 75)) if len(anomaly_scores) else 0.0,
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_stats(scores, y_test)
    return {
        "hmean": met["hmean"],
        "pAUC": met["pauc"],
        "KS": geom["source_target_ks"],
        "TPR@95n": tail["TPR@95n"],
        "TPR@99n": tail["TPR@99n"],
        "q95": tail["q95"],
        "q99": tail["q99"],
        "anomaly_median": tail["anomaly_median"],
        "anomaly_q75": tail["anomaly_q75"],
    }


def expected_test_reference(data_root: Path, machine: str) -> dict:
    items = filter_items(scan_dataset(data_root), machine)
    _, test_items = split_train_test(items, use_target_train=False)
    return {
        "paths": np.asarray([str(x.path) for x in test_items], dtype=str),
        "y_test": np.asarray([x.label for x in test_items], dtype=np.int64),
        "domains": np.asarray([x.domain for x in test_items], dtype=str),
        "train_count": len([x for x in items if x.split == "train" and x.domain == "source" and x.label in (0, None)]),
        "test_count": len(test_items),
    }


def whiten_fit_transform(train_emb: np.ndarray, test_emb: np.ndarray, pca_dim: int):
    dim = min(pca_dim, train_emb.shape[0], train_emb.shape[1])
    pca = PCA(n_components=dim, whiten=True, svd_solver="auto", random_state=0)
    train_w = pca.fit_transform(train_emb)
    test_w = pca.transform(test_emb)
    return train_w.astype(np.float64), test_w.astype(np.float64), {
        "effective_dim": int(dim),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5):
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, len(train_n))
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    status = {
        "mode": "check",
        "data_root_exists": Path(args.data_root).exists(),
        "a0_root_exists": Path(args.a0_root).exists(),
        "v63_root_exists": Path(args.v63_root).exists(),
        "all_a0_embedding_files_exist": all((Path(args.a0_root) / m / f"{m}_beats_embeddings.npz").exists() for m in machines),
    }
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")
    Path(args.doc_path).write_text(
        "# CTAF Stage14A A2 Minimal Adaptation Audit\n\n"
        "Mode=check completed. No PCA fitting, no metric computation.\n"
    )


def run_a2_pca_beats(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    per_machine = []
    tail_rows = []
    pca_rows = []
    score_dir = out_root / "a2_pca_beats" / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    for machine in machines:
        emb_path = Path(args.a0_root) / machine / f"{machine}_beats_embeddings.npz"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing A0 embeddings for {machine}: {emb_path}")
        arr = np.load(emb_path, allow_pickle=True)
        train_emb = arr["train_embeddings"].astype(np.float64)
        test_emb = arr["test_embeddings"].astype(np.float64)
        y_test = arr["y_test"].astype(np.int64)
        domains = arr["domains"].astype(str)
        test_paths = arr["test_paths"].astype(str)
        train_w, test_w, pca_info = whiten_fit_transform(train_emb, test_emb, args.pca_dim)
        scores = knn_cosine_scores(train_w, test_w, k=args.latent_knn_k)
        ev = evaluate(scores, y_test, domains)
        np.savez(
            score_dir / f"{machine}_pca_beats_knn_scores.npz",
            test_scores=scores,
            y_test=y_test,
            domains=domains,
            test_paths=test_paths,
            score_mode="pca_beats_knn",
            pca_dim=args.pca_dim,
            effective_dim=pca_info["effective_dim"],
            explained_variance_ratio_sum=pca_info["explained_variance_ratio_sum"],
            source_embedding_path=str(emb_path),
        )
        per_machine.append({"machine": machine, **{k: ev[k] for k in ["hmean", "pAUC", "KS", "TPR@99n"]}})
        tail_rows.append({"machine": machine, **tail_stats(scores, y_test)})
        pca_rows.append({"machine": machine, **pca_info})

    write_csv(out_root / "a2_pca_beats" / "per_machine.csv", per_machine)
    write_csv(out_root / "a2_pca_beats" / "tail_table.csv", tail_rows)
    write_csv(out_root / "a2_pca_beats" / "pca_summary.csv", pca_rows)
    write_csv(out_root / "a2_pca_beats" / "summary.csv", [{
        "route": "pca_beats_knn",
        "mean_hmean": float(np.mean([r["hmean"] for r in per_machine])),
        "mean_pAUC": float(np.mean([r["pAUC"] for r in per_machine])),
        "mean_KS": float(np.mean([r["KS"] for r in per_machine])),
        "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in per_machine])),
        "pca_dim": args.pca_dim,
    }])
    Path(args.doc_path).write_text(
        "# CTAF Stage14A A2 Minimal Adaptation Audit\n\n"
        "Mode=a2_pca_beats completed.\n"
    )


def run_a2_fusion(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    alignment_rows = []
    seed_rows = []
    comp_rows = []
    alignment_issue = False

    for machine in machines:
        expected = expected_test_reference(Path(args.data_root), machine)
        pca_npz = Path(args.output_root) / "a2_pca_beats" / "scores" / f"{machine}_pca_beats_knn_scores.npz"
        if not pca_npz.exists():
            raise FileNotFoundError(f"Missing PCA-BEATs score for {machine}: {pca_npz}")
        b1 = load_bundle(pca_npz)
        for seed in seeds:
            v37_npz = Path(args.v37_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v61_npz = Path(args.v61_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
            v63_npz = Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"
            row = {
                "seed": seed,
                "machine": machine,
                "v63_exists": v63_npz.exists(),
                "b1_exists": pca_npz.exists(),
                "y_test_match": False,
                "domains_match": False,
                "sample_order_match": False,
                "alignment_ok": False,
                "issue": "",
            }
            if not v63_npz.exists():
                row["issue"] = "missing_score_file"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            b37 = load_bundle(v37_npz)
            b61 = load_bundle(v61_npz)
            v63 = load_bundle(v63_npz)
            y_match = np.array_equal(v63["y_test"], b1["y_test"]) and np.array_equal(v63["y_test"], expected["y_test"])
            d_match = np.array_equal(v63["domains"], b1["domains"]) and np.array_equal(v63["domains"], expected["domains"])
            paths_match = v63["test_paths"] is not None and b1["test_paths"] is not None and np.array_equal(v63["test_paths"], b1["test_paths"]) and np.array_equal(v63["test_paths"], expected["paths"])
            row["y_test_match"] = y_match
            row["domains_match"] = d_match
            row["sample_order_match"] = paths_match
            row["alignment_ok"] = y_match and d_match and paths_match
            if not row["alignment_ok"]:
                row["issue"] = "alignment_mismatch"
                alignment_rows.append(row)
                alignment_issue = True
                continue
            alignment_rows.append(row)

            c_v37 = b37["test_scores"]
            c_v61 = b61["test_scores"]
            c_v63 = v63["test_scores"]
            c_b1 = b1["test_scores"]
            c_f1 = 0.5 * rank_norm(c_v63) + 0.5 * rank_norm(c_b1)
            y_test = v63["y_test"]
            domains = v63["domains"]

            for cand, scores in [("V37", c_v37), ("V61", c_v61), ("V63", c_v63), ("B1_pca_beats", c_b1), ("F1_v63_pca_beats_rankfusion", c_f1)]:
                ev = evaluate(scores, y_test, domains)
                seed_rows.append({"seed": seed, "machine": machine, "candidate": cand, **ev})

            q99_v63 = tail_stats(c_v63, y_test)["q99"]
            q99_b1 = tail_stats(c_b1, y_test)["q99"]
            normal_mask = y_test == 0
            anomaly_mask = y_test == 1
            top_v63 = set(np.where(normal_mask & (c_v63 >= q99_v63))[0].tolist())
            top_b1 = set(np.where(normal_mask & (c_b1 >= q99_b1))[0].tolist())
            missed_v63 = set(np.where(anomaly_mask & (c_v63 <= q99_v63))[0].tolist())
            missed_b1 = set(np.where(anomaly_mask & (c_b1 <= q99_b1))[0].tolist())
            top_union = len(top_v63 | top_b1)
            miss_union = len(missed_v63 | missed_b1)
            sp = spearmanr(c_v63, c_b1)
            comp_rows.append({
                "seed": seed,
                "machine": machine,
                "spearman_r": float(sp.correlation) if sp.correlation == sp.correlation else 0.0,
                "spearman_p": float(sp.pvalue) if sp.pvalue == sp.pvalue else 1.0,
                "top_normal_tail_jaccard": float(len(top_v63 & top_b1) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_v63 & missed_b1) / miss_union) if miss_union else 1.0,
                "v63_detect_b1_miss": int(np.sum(anomaly_mask & (c_v63 > q99_v63) & (c_b1 <= q99_b1))),
                "b1_detect_v63_miss": int(np.sum(anomaly_mask & (c_b1 > q99_b1) & (c_v63 <= q99_v63))),
            })
            fused_dir = out_root / "a2_v63_pca_beats_fusion" / f"seed_{seed}" / "scores"
            fused_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                fused_dir / f"{machine}_v63_pca_beats_rankfusion_scores.npz",
                test_scores=c_f1,
                y_test=y_test,
                domains=domains,
                test_paths=expected["paths"],
                score_mode="v63_pca_beats_rankfusion",
                fusion_formula="0.5 * rank_norm(V63) + 0.5 * rank_norm(B1)",
                source_v63_score_path=str(v63_npz),
                source_b1_score_path=str(pca_npz),
            )

    write_csv(out_root / "a2_v63_pca_beats_fusion" / "alignment_summary.csv", alignment_rows)
    if alignment_issue:
        Path(args.doc_path).write_text(
            "# CTAF Stage14A A2 Minimal Adaptation Audit\n\n"
            "Mode=a2_v63_pca_beats_fusion failed due to alignment issue.\n"
        )
        return

    write_csv(out_root / "a2_v63_pca_beats_fusion" / "seed_metrics.csv", seed_rows)
    write_csv(out_root / "a2_v63_pca_beats_fusion" / "complementarity_diagnostics.csv", comp_rows)

    by_seed_cand = defaultdict(list)
    by_machine_cand = defaultdict(list)
    for row in seed_rows:
        by_seed_cand[(row["seed"], row["candidate"])].append(row)
        by_machine_cand[(row["machine"], row["candidate"])].append(row)

    summary_rows = []
    for (seed, cand), rows in sorted(by_seed_cand.items()):
        summary_rows.append({
            "seed": seed,
            "candidate": cand,
            "hmean": float(np.mean([r["hmean"] for r in rows])),
            "pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "KS": float(np.mean([r["KS"] for r in rows])),
            "TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "summary.csv", summary_rows)

    aggregate_rows = []
    for cand in ["V37", "V61", "V63", "B1_pca_beats", "F1_v63_pca_beats_rankfusion"]:
        rows = [r for r in summary_rows if r["candidate"] == cand]
        aggregate_rows.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "std_hmean": float(np.std([r["hmean"] for r in rows], ddof=0)),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "std_pAUC": float(np.std([r["pAUC"] for r in rows], ddof=0)),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "std_KS": float(np.std([r["KS"] for r in rows], ddof=0)),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })

    per_machine_rows = []
    comp_summary_rows = []
    by_machine_comp = defaultdict(list)
    for row in comp_rows:
        by_machine_comp[row["machine"]].append(row)
    for (machine, cand), rows in sorted(by_machine_cand.items()):
        per_machine_rows.append({
            "machine": machine,
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    for machine, rows in sorted(by_machine_comp.items()):
        comp_summary_rows.append({
            "machine": machine,
            "mean_spearman_r": float(np.mean([r["spearman_r"] for r in rows])),
            "mean_top_normal_tail_jaccard": float(np.mean([r["top_normal_tail_jaccard"] for r in rows])),
            "mean_missed_anomaly_jaccard": float(np.mean([r["missed_anomaly_jaccard"] for r in rows])),
            "sum_v63_detect_b1_miss": int(np.sum([r["v63_detect_b1_miss"] for r in rows])),
            "sum_b1_detect_v63_miss": int(np.sum([r["b1_detect_v63_miss"] for r in rows])),
        })
    write_csv(out_root / "complementarity_summary.csv", comp_summary_rows)

    agg = {r["candidate"]: r for r in aggregate_rows}
    pm = {(r["machine"], r["candidate"]): r for r in per_machine_rows}
    a1 = {"mean_hmean": 0.5813, "mean_pAUC": 0.5484, "mean_TPR@99n": 0.0976}
    f1 = agg["F1_v63_pca_beats_rankfusion"]
    bottleneck_improved = sum(
        pm[(m, "F1_v63_pca_beats_rankfusion")]["mean_TPR@99n"] >= a1["mean_TPR@99n"] - 0.005
        for m in BOTTLENECK
    )
    verdict = "Stage14A A2 fail"
    if f1["mean_hmean"] >= a1["mean_hmean"] - 0.002 and f1["mean_pAUC"] >= a1["mean_pAUC"] and f1["mean_TPR@99n"] >= a1["mean_TPR@99n"] - 0.005:
        verdict = "Stage14A A2 pass"
    elif f1["mean_hmean"] > a1["mean_hmean"] and f1["mean_pAUC"] < a1["mean_pAUC"]:
        verdict = "hmean-positive but pAUC-negative"
    elif f1["mean_pAUC"] > a1["mean_pAUC"] and f1["mean_hmean"] < a1["mean_hmean"] - 0.002:
        verdict = "pAUC-positive but hmean-negative"

    lines = [
        "# CTAF Stage14A A2 Minimal Adaptation Audit",
        "",
        "## 1. Mode Summary",
        "",
        "- A2 minimal adaptation = frozen BEATs embedding + PCA/whitening only",
        "- no training",
        "- no fine-tuning",
        "- no projection head",
        "",
        "## 2. 3-Seed Comparison",
        "",
        "| route | mean hmean | std hmean | mean pAUC | std pAUC | mean KS | std KS | mean TPR@95n | mean TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_rows:
        lines.append(
            f"| {r['candidate']} | {r['mean_hmean']:.6f} | {r['std_hmean']:.6f} | "
            f"{r['mean_pAUC']:.6f} | {r['std_pAUC']:.6f} | {r['mean_KS']:.6f} | {r['std_KS']:.6f} | "
            f"{r['mean_TPR@95n']:.6f} | {r['mean_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 3. Per-Machine Comparison",
        "",
        "| machine | route | mean hmean | mean pAUC | mean KS | mean TPR@99n |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for r in per_machine_rows:
        lines.append(
            f"| {r['machine']} | {r['candidate']} | {r['mean_hmean']:.6f} | {r['mean_pAUC']:.6f} | "
            f"{r['mean_KS']:.6f} | {r['mean_TPR@99n']:.6f} |"
        )
    lines.extend([
        "",
        "## 4. Complementarity Diagnostics",
        "",
        "| machine | mean spearman_r | mean top normal-tail jaccard | mean missed-anomaly jaccard | V63-detect / B1-miss | B1-detect / V63-miss |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in comp_summary_rows:
        lines.append(
            f"| {r['machine']} | {r['mean_spearman_r']:.6f} | {r['mean_top_normal_tail_jaccard']:.6f} | "
            f"{r['mean_missed_anomaly_jaccard']:.6f} | {r['sum_v63_detect_b1_miss']} | {r['sum_b1_detect_v63_miss']} |"
        )
    lines.extend([
        "",
        "## 5. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "## 6. Confirmation",
        "",
        "- no training occurred",
        "- no fine-tuning occurred",
        "- no projection-head training occurred",
        "- no LoRA/adapter was used",
        "- no fusion weight tuning was performed",
        "- no per-machine selection was used",
        "- no Stage 14B opened",
    ])
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "a2_pca_beats":
        run_a2_pca_beats(args)
        return
    if args.mode == "a2_v63_pca_beats_fusion":
        run_a2_fusion(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
