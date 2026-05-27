#!/usr/bin/env python
"""V56A: BEATs-to-CTAF Distillability Diagnostic (no-training audit).

Tests whether BEATs utterance embeddings can be linearly predicted from
CTA-F V37 latent_summary vectors, and whether the two representations
share sample-level neighbor structure and score correlation.

No training. No model modification.
"""

import argparse, csv, os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v37_dir", default="results/v37_teacher_student_pretrain_seedpanel/seed_13711")
    p.add_argument("--v55_dir", default="results/v55_frozen_beats_representation_audit")
    p.add_argument("--out_dir", default="results/v56a_beats_ctaf_distillability_audit")
    p.add_argument("--out_md", default="docs/CTAF_V56A_BEATS_CTAF_DISTILLABILITY_AUDIT.md")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--ridge_alpha", type=float, default=1.0)
    p.add_argument("--cv_folds", type=int, default=5)
    return p.parse_args()


def deterministic_cv(n, folds=5):
    """Return list of (train_idx, test_idx) for n samples."""
    idx = np.arange(n)
    np.random.RandomState(13711).shuffle(idx)
    fold_size = n // folds
    splits = []
    for i in range(folds):
        start, end = i * fold_size, (i + 1) * fold_size if i < folds - 1 else n
        test_idx = idx[start:end]
        train_idx = np.setdiff1d(idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


def l2_normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return X / norms


def neighbor_recall(X, Y, k=5):
    """For each row, compute recall@k of Y-neighbors in X-neighbor set."""
    Xn = l2_normalize(X)
    Yn = l2_normalize(Y)
    n = Xn.shape[0]
    recall = np.zeros(n)
    jaccard = np.zeros(n)
    for i in range(n):
        xd = 2.0 - 2.0 * (Xn[i:i+1] @ Xn.T)
        xd[0, i] = np.inf
        x_nn = set(np.argpartition(xd[0], k)[:k])
        yd = 2.0 - 2.0 * (Yn[i:i+1] @ Yn.T)
        yd[0, i] = np.inf
        y_nn = set(np.argpartition(yd[0], k)[:k])
        inter = x_nn & y_nn
        recall[i] = len(inter) / k
        jaccard[i] = len(inter) / len(x_nn | y_nn)
    return float(np.mean(recall)), float(np.mean(jaccard))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ridge_rows, resid_rows, neighbor_rows, score_rows, align_rows = [], [], [], [], []

    for machine in MACHINES:
        print(f"\n{machine}")

        # Load V37
        v37_path = Path(args.v37_dir) / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
        v37 = np.load(v37_path, allow_pickle=True)
        ctaf_train = v37["train_latent_summaries"].astype(np.float64)  # (990, 192)
        ctaf_test = v37["test_latent_summaries"].astype(np.float64)     # (200, 192)

        # Load V55
        v55_path = Path(args.v55_dir) / machine / f"{machine}_beats_embeddings.npz"
        v55 = np.load(v55_path, allow_pickle=True)
        beats_train = v55["train_embeddings"].astype(np.float64)  # (990, 768)
        beats_test = v55["test_embeddings"].astype(np.float64)    # (200, 768)
        v55_scores = v55["test_scores"].astype(np.float64)        # (200,)
        v37_scores = v37["test_scores"].astype(np.float64)

        n_train = ctaf_train.shape[0]
        n_test = ctaf_test.shape[0]

        # Alignment check
        aligned = n_train == beats_train.shape[0] and n_test == beats_test.shape[0]
        align_rows.append({
            "machine": machine,
            "ctaf_train": n_train,
            "beats_train": beats_train.shape[0],
            "ctaf_test": n_test,
            "beats_test": beats_test.shape[0],
            "aligned": aligned,
        })

        if not aligned:
            print(f"  ALIGNMENT FAILED: ctaf({n_train},{n_test}) vs beats({beats_train.shape[0]},{beats_test.shape[0]})")
            continue

        # 1. Ridge regression
        X, Y = ctaf_train, beats_train
        folds = deterministic_cv(n_train, args.cv_folds)
        r2s = []
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        for tr_idx, te_idx in folds:
            X_tr, X_te = X[tr_idx], X[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]
            x_scaler = StandardScaler().fit(X_tr)
            y_scaler = StandardScaler().fit(Y_tr)
            X_tr_s = x_scaler.transform(X_tr)
            X_te_s = x_scaler.transform(X_te)
            Y_tr_s = y_scaler.transform(Y_tr)
            Y_te_s = y_scaler.transform(Y_te)
            model = Ridge(alpha=args.ridge_alpha).fit(X_tr_s, Y_tr_s)
            Y_pred = model.predict(X_te_s)
            ss_res = np.sum((Y_te_s - Y_pred) ** 2)
            ss_tot = np.sum((Y_te_s - np.mean(Y_te_s, axis=0)) ** 2)
            r2s.append(1.0 - ss_res / max(ss_tot, 1e-12))
        ridge_rows.append({
            "machine": machine, "n_samples": int(n_train),
            "r2_mean": round(float(np.mean(r2s)), 6),
            "r2_std": round(float(np.std(r2s)), 6),
        })
        print(f"  Ridge R² = {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")

        # 2. Residual analysis (full fit)
        x_scaler = StandardScaler().fit(X)
        X_s = x_scaler.transform(X)
        y_scaler = StandardScaler().fit(Y)
        Y_s = y_scaler.transform(Y)
        model = Ridge(alpha=args.ridge_alpha).fit(X_s, Y_s)
        Y_pred = model.predict(X_s)
        residuals = np.linalg.norm(Y_s - Y_pred, axis=1)  # per-sample L2
        resid_rows.append({
            "machine": machine,
            "residual_mean": round(float(np.mean(residuals)), 6),
            "residual_std": round(float(np.std(residuals)), 6),
            "n_samples": int(n_train),
        })
        # z-score: compute after all machines processed

        # 3. Neighbor preservation
        rec, jac = neighbor_recall(X, Y, args.k)
        neighbor_rows.append({
            "machine": machine,
            "recall_at_5": round(rec, 6),
            "jaccard_at_5": round(jac, 6),
        })
        print(f"  Neighbor recall@5 = {rec:.4f}, Jaccard@5 = {jac:.4f}")

        # 4. Score relation
        from scipy.stats import spearmanr, pearsonr
        sr, _ = spearmanr(v37_scores, v55_scores)
        pr, _ = pearsonr(v37_scores, v55_scores)
        score_rows.append({
            "machine": machine,
            "spearman_rho": round(float(sr), 6),
            "pearson_r": round(float(pr), 6),
            "n_samples": int(n_test),
        })
        print(f"  Score ρ = {sr:.4f}, r = {pr:.4f}")

    # Compute residual z-scores
    if resid_rows:
        all_means = np.array([r["residual_mean"] for r in resid_rows])
        global_mean = np.mean(all_means)
        global_std = np.std(all_means)
        for r in resid_rows:
            r["residual_z_vs_all"] = round((r["residual_mean"] - global_mean) / max(global_std, 1e-8), 4)

    # Write CSVs
    for name, rows in [("ridge_r2.csv", ridge_rows), ("residual_analysis.csv", resid_rows),
                        ("neighbor_preservation.csv", neighbor_rows), ("score_relation.csv", score_rows),
                        ("alignment_report.csv", align_rows)]:
        if rows:
            with (out_dir / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    # Build report
    report = _build_report(ridge_rows, resid_rows, neighbor_rows, score_rows, align_rows, args)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(report)
    print(f"\n{args.out_md}")
    print(report)


def _build_report(ridge, resid, neighbor, score, align, args):
    L = [
        "# CTAF V56A BEATs-to-CTAF Distillability Diagnostic",
        "",
        "## 1. V55 Summary",
        "",
        "V55 tested frozen BEATs + kNN as an external representation:",
        "- hmean 0.5786 > V37 0.5631 (PASS)",
        "- pAUC 0.5388 < V37 0.5505 (FAIL)",
        "- 6/7 machines normal q99 inflation",
        "- bearing collapse (TPR@99n 0.27→0.07), valve breakout (0.10→0.18)",
        "- BEATs has machine-level complementary signal but cannot directly replace CTAF",
        "",
        "## 2. V56A Motivation",
        "",
        "Determine whether BEATs embeddings are linearly predictable from CTA-F",
        "V37 latent_summary at the sample level. If yes, BEATs may serve as a",
        "distillation teacher. If no, BEATs is only machine-level complementary,",
        "and cross-architecture distillation is not justified.",
        "",
        f"Fixed: Ridge α={args.ridge_alpha}, CV={args.cv_folds}-fold, k={args.k}.",
        "No training. No BEATs fine-tuning. No CTAF training.",
        "",
        "## 3. Alignment Validity",
        "",
        "Train/test sample counts must match exactly between V37 NPZ and V55 NPZ.",
        "",
        "| Machine | CTAF train | BEATs train | CTAF test | BEATs test | Aligned? |",
        "|---------|-----------|-------------|----------|------------|----------|",
    ]
    for r in align:
        L.append(f"| {r['machine']:10s} | {r['ctaf_train']} | {r['beats_train']} | "
                 f"{r['ctaf_test']} | {r['beats_test']} | {'YES' if r['aligned'] else 'NO'} |")
    L.append("")

    # 4. Ridge R²
    if ridge:
        L.extend(["## 4. Ridge Regression Distillability", "",
                  "CTA-F latent_summary (192-dim) → BEATs embedding (768-dim), Ridge α=1.0, 5-fold CV.",
                  "",
                  "| Machine | n_train | R² mean | R² std |",
                  "|---------|--------|---------|--------|"])
        for r in ridge:
            L.append(f"| {r['machine']:10s} | {r['n_samples']} | {r['r2_mean']:.4f} | {r['r2_std']:.4f} |")
        L.append("")

    # 5. Residual
    if resid:
        L.extend(["## 5. Residual Analysis", "",
                  "Per-sample L2 residual norm from Ridge fit on full train set.",
                  "",
                  "| Machine | Residual mean | Residual std | z vs all machines |",
                  "|---------|--------------|-------------|-------------------|"])
        for r in resid:
            L.append(f"| {r['machine']:10s} | {r['residual_mean']:.4f} | {r['residual_std']:.4f} | {r['residual_z_vs_all']} |")
        L.append("")

    # 6. Neighbor
    if neighbor:
        L.extend(["## 6. Neighbor Preservation", "",
                  "k=5, cosine distance on L2-normalized train-normal vectors.",
                  "",
                  "| Machine | Recall@5 | Jaccard@5 |",
                  "|---------|----------|-----------|"])
        for r in neighbor:
            L.append(f"| {r['machine']:10s} | {r['recall_at_5']:.4f} | {r['jaccard_at_5']:.4f} |")
        L.append("")

    # 7. Score relation
    if score:
        L.extend(["## 7. Score Relation", "",
                  "Spearman ρ and Pearson r between V37 CTA-F test scores and V55 BEATs test scores.",
                  "",
                  "| Machine | Spearman ρ | Pearson r | n_test |",
                  "|---------|-----------|----------|--------|"])
        for r in score:
            L.append(f"| {r['machine']:10s} | {r['spearman_rho']:.4f} | {r['pearson_r']:.4f} | {r['n_samples']} |")
        L.append("")

    # 8. Verdict
    rd = {r["machine"]: r["r2_mean"] for r in ridge}
    res_z = {r["machine"]: r["residual_z_vs_all"] for r in resid}
    rec = {r["machine"]: r["recall_at_5"] for r in neighbor}
    sp = {r["machine"]: r["spearman_rho"] for r in score}

    if not rd:
        L.extend(["## 8. Final Verdict", "", "**unclear**: no data available for verdict."])
        return "\n".join(L) + "\n"

    mean_r2 = np.mean(list(rd.values()))
    c1 = sum(1 for m in MACHINES if rd.get(m, 0) >= 0.30) >= 4
    c2 = all(rd.get(m, 0) >= mean_r2 - 0.10 for m in ["bearing", "fan"])
    c3 = np.mean(list(rec.values())) >= 0.10 if rec else False
    c4 = sum(1 for m in MACHINES if sp.get(m, 0) >= 0.20) >= 4
    c5 = all(res_z.get(m, 999) <= 1.0 for m in ["bearing", "fan", "toycar"])
    c6 = all(r["aligned"] for r in align)

    all_pass = c1 and c2 and c3 and c4 and c5 and c6

    L.extend([
        "## 8. Final Verdict", "",
        "| Gate | Condition | Result |",
        "|------|-----------|--------|",
        f"| 1 | ≥4 machines R²≥0.30 | {sum(1 for m in MACHINES if rd.get(m,0)>=0.30)}/7 {'PASS' if c1 else 'FAIL'} |",
        f"| 2 | bearing/fan R² ≥ mean-0.10 ({mean_r2-0.10:.3f}) | bearing={rd.get('bearing',0):.3f} fan={rd.get('fan',0):.3f} {'PASS' if c2 else 'FAIL'} |",
        f"| 3 | mean recall@5 ≥ 0.10 | {np.mean(list(rec.values())):.4f} {'PASS' if c3 else 'FAIL'} |",
        f"| 4 | ≥4 machines score ρ≥0.20 | {sum(1 for m in MACHINES if sp.get(m,0)>=0.20)}/7 {'PASS' if c4 else 'FAIL'} |",
        f"| 5 | bearing/fan/toycar residual z≤1.0 | bearing={res_z.get('bearing',0)} fan={res_z.get('fan',0)} toycar={res_z.get('toycar',0)} {'PASS' if c5 else 'FAIL'} |",
        f"| 6 | data aligned | {'PASS' if c6 else 'FAIL'} |",
        "",
    ])

    if all_pass:
        L.append("**V56A pass: BEATs-to-CTAF distillation is eligible for human review.**")
    elif not c6:
        L.append("**unclear: alignment or missing-data issue prevents conclusion.**")
    else:
        L.append("**V56A fail: do not train V56. BEATs has machine-level complementary signal, but not sample-level distillable structure.**")

    L.extend(["", "---", "",
              "No training performed. No BEATs fine-tuning. No CTAF encoder modification. ",
              "No score fusion. No per-machine selection."])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
