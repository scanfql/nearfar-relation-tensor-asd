#!/usr/bin/env python
"""V57 No-training Discriminative Proxy Feasibility Check.

Tests whether frozen V37 full_5view latent_summary features contain sufficient
machine-type discriminative structure, and whether AAM-softmax training has room
to improve intra-class compactness.

Loads frozen V37 NPZ latent_summary. Trains only classification heads (linear +
AAM-softmax). Does NOT train the CTAF encoder. No anomaly labels.
"""

import argparse, csv, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--v37_dir", default="results/v37_teacher_student_pretrain_seedpanel/seed_13711")
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--out_dir", default="results/v57_discriminative_proxy_feasibility_audit")
    p.add_argument("--out_md", default="docs/CTAF_V57_DISCRIMINATIVE_PROXY_FEASIBILITY_AUDIT.md")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--aam_epochs", type=int, default=30)
    p.add_argument("--aam_lr", type=float, default=0.01)
    p.add_argument("--aam_s", type=float, default=30)
    p.add_argument("--aam_m", type=float, default=0.2)
    return p.parse_args()


def deterministic_cv(n, folds=5):
    idx = np.arange(n)
    np.random.RandomState(13711).shuffle(idx)
    splits = []
    for i in range(folds):
        start = i * (n // folds)
        end = (i + 1) * (n // folds) if i < folds - 1 else n
        test_idx = idx[start:end]
        train_idx = np.setdiff1d(idx, test_idx)
        splits.append((train_idx, test_idx))
    return splits


# ---------------------------------------------------------------------------
# AAM-Softmax head in numpy
# ---------------------------------------------------------------------------

class AAMSoftmaxNumpy:
    """AAM-softmax linear classifier implemented in numpy (no torch).

    Fixed: s=30, m=0.2. Single linear layer: 192 -> 7.
    Trains with softmax + angular margin via iterative gradient descent.
    """

    def __init__(self, in_dim=192, n_classes=7, s=30.0, m=0.2):
        self.s = s
        self.m = m
        rng = np.random.RandomState(13711)
        self.W = rng.randn(in_dim, n_classes).astype(np.float64) * 0.01

    def _cos_theta(self, X_norm, W_norm):
        return X_norm @ W_norm

    def _forward(self, X, y_true):
        X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
        W_norm = self.W / np.maximum(np.linalg.norm(self.W, axis=0, keepdims=True), 1e-8)
        cos_t = self._cos_theta(X_norm, W_norm)  # [N, C]
        n = X_norm.shape[0]
        # Add angular margin only to the target class
        cos_t_m = cos_t.copy()
        for i in range(n):
            c = y_true[i]
            # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
            theta = np.arccos(np.clip(cos_t[i, c], -0.99999, 0.99999))
            cos_t_m[i, c] = np.cos(theta + self.m)
        logits = self.s * cos_t_m
        # Softmax + cross-entropy (numpy for small scale)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        loss = -np.mean(np.log(np.maximum(probs[np.arange(n), y_true], 1e-12)))
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == y_true)
        # Gradient
        grad = probs.copy()
        grad[np.arange(n), y_true] -= 1.0
        grad_W = X_norm.T @ grad  # [D, C]
        return loss, acc, grad_W / n

    def fit(self, X, y, X_val=None, y_val=None, epochs=30, lr=0.01):
        n = X.shape[0]
        for ep in range(epochs):
            loss, acc, grad_W = self._forward(X, y)
            self.W -= lr * grad_W
        if X_val is not None and y_val is not None:
            _, val_acc, _ = self._forward(X_val, y_val)
            return val_acc
        return acc


# ---------------------------------------------------------------------------
# D1: Linear classification
# ---------------------------------------------------------------------------

def run_linear_classification(X, y, folds):
    from sklearn.preprocessing import StandardScaler
    accs, per_class_accs = [], []
    n_classes = len(np.unique(y))
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        W = np.linalg.lstsq(np.hstack([X_tr_s, np.ones((len(X_tr_s), 1))]),
                            np.eye(n_classes)[y_tr], rcond=None)[0]
        preds = np.argmax(np.hstack([X_te_s, np.ones((len(X_te_s), 1))]) @ W, axis=1)
        acc = np.mean(preds == y_te)
        accs.append(acc)

        for c in range(n_classes):
            mask = y_te == c
            if mask.sum() > 0:
                pc = np.mean(preds[mask] == c)
                per_class_accs.append((c, float(pc)))

        for t, p in zip(y_te, preds):
            confusion[int(t), int(p)] += 1

    return float(np.mean(accs)), per_class_accs, confusion


# ---------------------------------------------------------------------------
# D2: AAM-softmax classification
# ---------------------------------------------------------------------------

def run_aam_classification(X, y, folds, args):
    from sklearn.preprocessing import StandardScaler
    accs, per_class_accs = [], []
    n_classes = len(np.unique(y))
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for tr_idx, te_idx in folds:
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        model = AAMSoftmaxNumpy(192, n_classes, s=args.aam_s, m=args.aam_m)
        model.fit(X_tr_s, y_tr, X_val=X_te_s, y_val=y_te,
                  epochs=args.aam_epochs, lr=args.aam_lr)
        X_te_n = X_te_s / np.maximum(np.linalg.norm(X_te_s, axis=1, keepdims=True), 1e-12)
        W_n = model.W / np.maximum(np.linalg.norm(model.W, axis=0, keepdims=True), 1e-12)
        preds = np.argmax(X_te_n @ W_n, axis=1)
        acc = np.mean(preds == y_te)
        accs.append(acc)

        for c in range(n_classes):
            mask = y_te == c
            if mask.sum() > 0:
                per_class_accs.append((c, float(np.mean(preds[mask] == c))))

        for t, p in zip(y_te, preds):
            confusion[int(t), int(p)] += 1

    return float(np.mean(accs)), per_class_accs, confusion


# ---------------------------------------------------------------------------
# D3-D6
# ---------------------------------------------------------------------------

def run_geometry_diagnostics(X, y, aam_model):
    n_classes = len(np.unique(y))
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    # D3: intra-class cosine spread
    d3_rows = []
    for c in range(n_classes):
        Xi = X_norm[y == c]
        centroid = Xi.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        cos_dists = 1.0 - Xi @ centroid
        d3_rows.append({
            "machine_id": int(c),
            "machine_name": MACHINES[c],
            "cosine_spread_mean": round(float(np.mean(cos_dists)), 6),
            "cosine_spread_std": round(float(np.std(cos_dists)), 6),
            "cosine_spread_q95": round(float(np.quantile(cos_dists, 0.95)), 6),
            "cosine_spread_q99": round(float(np.quantile(cos_dists, 0.99)), 6),
        })

    # D4: inter-class angular margin (from AAM weight vectors)
    W = aam_model.W  # [192, 7]
    W_norm = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
    d4_rows = []
    for c in range(n_classes):
        cos_sim = W_norm[:, c] @ W_norm  # [7]
        cos_sim[c] = -1.0  # exclude self
        nearest = int(np.argmax(cos_sim))
        angle_rad = float(np.arccos(np.clip(cos_sim[nearest], -0.99999, 0.99999)))
        d4_rows.append({
            "machine_id": int(c),
            "machine_name": MACHINES[c],
            "nearest_class": MACHINES[nearest],
            "min_angle_rad": round(angle_rad, 6),
            "min_angle_deg": round(float(np.degrees(angle_rad)), 4),
        })

    # D5: per-class nearest centroid margin
    d5_rows = []
    centroids = []
    for c in range(n_classes):
        Xi = X_norm[y == c]
        centroids.append(Xi.mean(axis=0) / max(np.linalg.norm(Xi.mean(axis=0)), 1e-12))
    for c in range(n_classes):
        cos_sim = centroids[c] @ np.column_stack(centroids)
        cos_sim[c] = -1.0
        nearest = int(np.argmax(cos_sim))
        d5_rows.append({
            "machine_id": int(c),
            "machine_name": MACHINES[c],
            "nearest_class": MACHINES[nearest],
            "cosine_distance": round(float(1.0 - cos_sim[nearest]), 6),
            "angle_rad": round(float(np.arccos(np.clip(cos_sim[nearest], -0.99999, 0.99999))), 6),
        })

    # D6: compactness potential (= current_spread)
    d6_rows = []
    for r in d3_rows:
        d6_rows.append({
            "machine_id": r["machine_id"],
            "machine_name": r["machine_name"],
            "compactness_potential": round(r["cosine_spread_mean"], 6),
        })

    return d3_rows, d4_rows, d5_rows, d6_rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load data from all machines
    all_X, all_y = [], []
    for c, machine in enumerate(MACHINES):
        path = Path(args.v37_dir) / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
        d = np.load(path, allow_pickle=True)
        # train_latent_summaries = source train-normal for this machine
        X_m = d["train_latent_summaries"].astype(np.float64)  # (990, 192)
        all_X.append(X_m)
        all_y.append(np.full(990, c, dtype=np.int64))
    X = np.concatenate(all_X, axis=0)  # (6930, 192)
    y = np.concatenate(all_y, axis=0)  # (6930,)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print("Class distribution:", {MACHINES[c]: int(np.sum(y == c)) for c in range(7)})

    folds = deterministic_cv(X.shape[0], args.cv_folds)

    # D1
    print("\n[D1] Linear classification...")
    d1_acc, d1_per_class, d1_conf = run_linear_classification(X, y, folds)
    print(f"  macro accuracy = {d1_acc:.4f}")

    # D2
    print("\n[D2] AAM-softmax classification...")
    d2_acc, d2_per_class, d2_conf = run_aam_classification(X, y, folds, args)
    print(f"  macro accuracy = {d2_acc:.4f}")

    # Fit full AAM model for geometry
    from sklearn.preprocessing import StandardScaler
    X_s = StandardScaler().fit_transform(X)
    aam_full = AAMSoftmaxNumpy(192, 7, s=args.aam_s, m=args.aam_m)
    aam_acc = aam_full.fit(X_s, y, epochs=args.aam_epochs, lr=args.aam_lr)
    print(f"  full AAM train acc = {aam_acc:.4f}")

    # D3-D6
    print("\n[D3-D6] Geometry diagnostics...")
    d3, d4, d5, d6 = run_geometry_diagnostics(X, y, aam_full)
    for r in d3:
        print(f"  {r['machine_name']:10s}  spread={r['cosine_spread_mean']:.4f}  q99={r['cosine_spread_q99']:.4f}")

    # Save CSVs
    def _save_csv(name, rows):
        if rows:
            with (out_dir / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

    # D1 per-class
    d1_pc_rows = [{"machine": MACHINES[c], "accuracy": round(float(acc), 6)}
                  for c, acc in d1_per_class]
    _save_csv("linear_classification_accuracy.csv", [
        {"machine": MACHINES[c], "accuracy": round(float(dict(d1_per_class).get(c, 0)), 6)}
        for c in range(7)
    ] + [{"machine": "macro", "accuracy": round(d1_acc, 6)}])

    d2_pc = dict(d2_per_class)
    aam_rows = [{"machine": MACHINES[c], "accuracy": round(float(d2_pc.get(c, 0)), 6)} for c in range(7)]
    aam_rows.append({"machine": "macro", "accuracy": round(d2_acc, 6)})
    _save_csv("aam_classification_accuracy.csv", aam_rows)

    _save_csv("intra_class_cosine_spread.csv", d3)
    _save_csv("inter_class_angular_margin.csv", d4)
    _save_csv("nearest_class_centroid_margin.csv", d5)
    _save_csv("compactness_potential.csv", d6)

    # Confusion matrices
    np.savetxt(str(out_dir / "confusion_matrix_linear.csv"), d1_conf, fmt="%d", delimiter=",")
    np.savetxt(str(out_dir / "confusion_matrix_aam.csv"), d2_conf, fmt="%d", delimiter=",")

    # Build report
    report = _build_report(args, d1_acc, d2_acc, d3, d4, d5, d6, d1_conf, d2_conf)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(report)
    print(f"\n{args.out_md}")
    print(report)


def _build_report(args, d1_acc, d2_acc, d3, d4, d5, d6, d1_conf, d2_conf):
    L = [
        "# CTAF V57 Discriminative Proxy Feasibility Audit",
        "",
        "## 1. Stage 11 Prereg Summary",
        "",
        "V57 tests whether frozen V37 full_5view latent_summary (192-dim) contains",
        "sufficient machine-type discriminative structure for AAM-softmax to have",
        "room to improve intra-class normal manifold compactness.",
        "",
        "No CTAF encoder training. No anomaly labels. No V58.",
        "",
        "## 2. Protocol",
        "",
        f"- Feature: V37 frozen latent_summary, 192-dim, 7 machines × 990 train-normal",
        f"- Label: machine type (7 classes)",
        f"- D1: Linear classification, 5-fold CV, StandardScaler",
        f"- D2: AAM-softmax, s={args.aam_s}, m={args.aam_m}, {args.aam_epochs} epochs, lr={args.aam_lr}, 5-fold CV",
        f"- D3-D6: Geometry diagnostics on full train set using AAM weight vectors",
        "",
        "## 3. D1 — Linear Classification Accuracy",
        "",
        f"5-fold CV macro accuracy: **{d1_acc:.4f}**",
        "",
        "| Machine | Per-class accuracy |",
        "|---------|-------------------|",
    ]
    for c in range(7):
        pc = dict([(int(i), v) for i, v in zip(range(7), [np.diag(d1_conf)[c]/max(d1_conf[c].sum(),1) for c in range(7)])]).get(c, 0)
        L.append(f"| {MACHINES[c]:10s} | {pc:.4f} |")
    L.append("")
    L.append("Confusion matrix (rows=true, cols=pred):")
    L.append("```")
    L.append("        " + " ".join(f"{MACHINES[c][:4]:>5s}" for c in range(7)))
    for c in range(7):
        L.append(f"{MACHINES[c][:4]:5s} " + " ".join(f"{int(d1_conf[c,i]):5d}" for i in range(7)))
    L.append("```")
    L.append("")

    L.extend([
        "## 4. D2 — AAM-Softmax Classification Accuracy",
        "",
        f"5-fold CV macro accuracy: **{d2_acc:.4f}**",
        "",
        "| Machine | Per-class accuracy |",
        "|---------|-------------------|",
    ])
    for c in range(7):
        pc = d2_conf[c,c] / max(d2_conf[c].sum(), 1)
        L.append(f"| {MACHINES[c]:10s} | {pc:.4f} |")
    L.append("")
    L.append("Confusion matrix:")
    L.append("```")
    L.append("        " + " ".join(f"{MACHINES[c][:4]:>5s}" for c in range(7)))
    for c in range(7):
        L.append(f"{MACHINES[c][:4]:5s} " + " ".join(f"{int(d2_conf[c,i]):5d}" for i in range(7)))
    L.append("```")
    L.append("")

    L.extend([
        "## 5. D3 — Intra-Class Cosine Spread",
        "",
        "| Machine | Mean spread | Std | Q95 | Q99 |",
        "|---------|-------------|-----|-----|-----|",
    ])
    for r in d3:
        L.append(f"| {r['machine_name']:10s} | {r['cosine_spread_mean']:.4f} | {r['cosine_spread_std']:.4f} | {r['cosine_spread_q95']:.4f} | {r['cosine_spread_q99']:.4f} |")
    L.append("")

    L.extend([
        "## 6. D4 — Inter-Class Angular Margin (AAM weights)",
        "",
        "| Machine | Nearest class | Min angle (rad) | Min angle (deg) |",
        "|---------|---------------|-----------------|-----------------|",
    ])
    for r in d4:
        L.append(f"| {r['machine_name']:10s} | {r['nearest_class']:10s} | {r['min_angle_rad']:.4f} | {r['min_angle_deg']:.1f} |")
    L.append("")

    L.extend([
        "## 7. D5 — Nearest-Class Centroid Margin",
        "",
        "| Machine | Nearest class | Cosine distance | Angle (rad) |",
        "|---------|---------------|-----------------|-------------|",
    ])
    for r in d5:
        L.append(f"| {r['machine_name']:10s} | {r['nearest_class']:10s} | {r['cosine_distance']:.4f} | {r['angle_rad']:.4f} |")
    L.append("")

    L.extend([
        "## 8. D6 — Compactness Potential",
        "",
        "| Machine | Potential (mean spread) |",
        "|---------|------------------------|",
    ])
    for r in d6:
        L.append(f"| {r['machine_name']:10s} | {r['compactness_potential']:.4f} |")
    L.append("")

    # --- Gates ---
    n_angle_ok = sum(1 for r in d4 if r["min_angle_rad"] >= 0.05)
    n_compact_ok = sum(1 for r in d6 if r["compactness_potential"] > 0.10)
    no_collapse = all(r["cosine_spread_mean"] > 0.0 for r in d3)
    d1_d2_consistent = abs(d1_acc - d2_acc) < 0.10

    c1 = d1_acc >= 0.90
    c2 = d2_acc >= 0.90
    c3 = no_collapse
    c4 = n_angle_ok >= 2
    c5 = n_compact_ok >= 3
    c6 = d1_d2_consistent

    L.extend([
        "## 9. Pass/Fail Gate Table",
        "",
        "| Gate | Condition | Result |",
        "|------|-----------|--------|",
        f"| 1 | D1 linear macro acc ≥ 0.90 | {d1_acc:.4f} {'PASS' if c1 else 'FAIL'} |",
        f"| 2 | D2 AAM macro acc ≥ 0.90 | {d2_acc:.4f} {'PASS' if c2 else 'FAIL'} |",
        f"| 3 | No collapsed classes (spread > 0) | {'PASS' if c3 else 'FAIL'} |",
        f"| 4 | ≥2 machines with angle margin ≥ 0.05 rad | {n_angle_ok}/7 {'PASS' if c4 else 'FAIL'} |",
        f"| 5 | ≥3 machines with compactness potential > 0.10 | {n_compact_ok}/7 {'PASS' if c5 else 'FAIL'} |",
        f"| 6 | D1/D2 accuracy consistent (Δ < 0.10) | Δ={abs(d1_acc-d2_acc):.4f} {'PASS' if c6 else 'FAIL'} |",
        "",
    ])

    all_pass = c1 and c2 and c3 and c4 and c5 and c6
    if all_pass:
        L.append("**V57 pass: V58 single-seed AAM-softmax training is eligible for human review.**")
    else:
        L.append("**V57 fail: do not train V58. Frozen V37 features have insufficient room for AAM-softmax to improve intra-class compactness.**")

    L.extend(["", "---", "",
              "No CTAF encoder training. No anomaly labels. No checkpoint modification. ",
              "AAM parameters (s=30, m=0.2) fixed. No sweep. No per-machine selection."])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
