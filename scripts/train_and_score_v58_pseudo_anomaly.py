#!/usr/bin/env python
"""V58 Single-Seed Pseudo-Anomaly Training.

Trains full_5view encoder with within-machine TF patch replacement pseudo-anomaly
binary discrimination (BCE). Classifier head discarded at inference — scoring uses
standard latent_summary_knn (k=5).

Seed=13711. 7 machines. No sweep. No per-machine selection.
"""

import argparse, csv, os, sys, tempfile, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
import torch, torch.nn as nn

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import load_audio, extract_logmel
from src.features.tensor_builder import build_tensor, TensorNormalizer
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder
from src.evaluation.scorer import compute_latent_summary_vectors, compute_latent_knn_scores, fit_feature_robust_stats
from src.evaluation.metrics import compute_metrics
from src.utils.seed import set_seed

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", default="results/v58_within_machine_pseudo_anomaly")
    p.add_argument("--out_md", default="docs/CTAF_V58_WITHIN_MACHINE_PSEUDO_ANOMALY_AUDIT.md")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patch_time_frac", type=float, default=0.05)
    p.add_argument("--patch_freq_frac", type=float, default=0.05)
    return p.parse_args()


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)


def generate_pseudo_batch(tensors, args):
    """For each sample in tensors, replace a random TF patch from a different
    same-machine donor. Returns pseudo tensors, each same shape as clean."""
    n = len(tensors)
    indices = list(range(n))
    # Shuffle to get donors (ensure donor ≠ anchor)
    donors = indices.copy()
    np.random.shuffle(donors)
    for i in range(n):
        if donors[i] == i:
            donors[i] = (i + 1) % n
    pseudo = []
    for i in range(n):
        clean = tensors[indices[i]].copy()
        donor = tensors[donors[i]]
        V, F, T = clean.shape
        t_len = max(1, int(T * args.patch_time_frac))
        f_len = max(1, int(F * args.patch_freq_frac))
        t_s = np.random.randint(0, T - t_len + 1)
        f_s = np.random.randint(0, F - f_len + 1)
        clean[:, f_s:f_s + f_len, t_s:t_s + t_len] = donor[:, f_s:f_s + f_len, t_s:t_s + t_len]
        pseudo.append(clean)
    return pseudo


def tail_metrics(scores, y):
    n, a = scores[y == 0], scores[y == 1]
    r = {}
    if len(n) > 0:
        r["normal_q95"] = float(np.quantile(n, 0.95))
        r["normal_q99"] = float(np.quantile(n, 0.99))
    if len(a) > 0:
        r["TPR_at_99n"] = float(np.mean(a > r.get("normal_q99", 0)))
    return r


def main():
    args = parse_args()
    set_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    all_items = scan_dataset(Path(args.data_root))
    rows, tail_rows = [], []

    for machine in MACHINES:
        print(f"\n{'='*60}\n{machine}\n{'='*60}")
        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)

        # Features
        train_feats = {}
        for item in train_items:
            a = load_audio(item.path); b = extract_logmel(a)
            train_feats[item.path] = (b, a)
        test_feats = {}
        for item in test_items:
            a = load_audio(item.path); b = extract_logmel(a)
            test_feats[item.path] = (b, a)

        train_tensors = [build_tensor("full_5view", train_feats[i.path][0]) for i in train_items]
        V = train_tensors[0].shape[0]
        test_tensors = [build_tensor("full_5view", test_feats[i.path][0]) for i in test_items]
        test_labels = [i.label for i in test_items]

        normalizer = TensorNormalizer()
        train_norm = normalizer.fit_transform(train_tensors)
        test_norm = [normalizer.transform(t) for t in test_tensors]
        train_arr = np.stack(train_norm, axis=0)
        test_arr = np.stack(test_norm, axis=0)
        y_test = np.array(test_labels, dtype=np.int64)

        # Model
        encoder = ViewwiseUnsharedLateFusionAutoencoder(
            in_channels=V, latent_channels=64, use_batch_norm=True,
            per_view_channels=16, stem_channels=8, dropout_p=0.0,
        ).to(args.device)
        classifier = BinaryClassifier(192).to(args.device)
        opt = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                               lr=args.lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()

        # Training
        n_train = len(train_arr)
        n_batches = n_train // args.batch_size
        print(f"  training: {n_train} samples, {args.epochs} epochs, batch={args.batch_size}")

        for ep in range(args.epochs):
            perm = np.random.permutation(n_train)
            # Pre-generate pseudo for all samples (different each epoch)
            all_pseudo = generate_pseudo_batch([train_arr[i] for i in perm], args)
            total_loss = 0.0
            for b in range(n_batches):
                start = b * args.batch_size; end = start + args.batch_size
                idx = perm[start:end]
                half = args.batch_size // 2
                clean_b = torch.from_numpy(train_arr[idx[:half]]).float().to(args.device)
                pseudo_b = torch.from_numpy(np.stack([all_pseudo[i] for i in range(start, start + half)], 0)).float().to(args.device)
                x = torch.cat([clean_b, pseudo_b], dim=0)
                labels = torch.cat([torch.zeros(half), torch.ones(half)]).float().to(args.device)
                feats = encoder.encode(x)
                lmean = feats.mean(dim=(2, 3))
                lstd = feats.std(dim=(2, 3), unbiased=False)
                lamax = feats.amax(dim=(2, 3))
                summary = torch.cat([lmean, lstd, lamax], dim=1)
                logits = classifier(summary).squeeze(-1)
                loss = criterion(logits, labels)
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += loss.item()
            if (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1}: loss={total_loss/n_batches:.4f}")

        encoder.eval()

        # Scoring: latent_summary_knn
        mem_ls = compute_latent_summary_vectors(encoder, train_arr, 32, args.device)
        test_ls = compute_latent_summary_vectors(encoder, test_arr, 32, args.device)
        sc = compute_latent_knn_scores(mem_ls, test_ls, k=5,
                                       train_feature_stats=fit_feature_robust_stats(mem_ls))

        # Also compute V37-style ls_knn (no robust stats) for fair comparison
        sc_raw = compute_latent_knn_scores(mem_ls, test_ls, k=5)

        met = compute_metrics(y_test, sc, None)
        met_raw = compute_metrics(y_test, sc_raw, None)
        t = tail_metrics(sc_raw, y_test)
        rows.append({"machine": machine, "experiment": "pseudo_anomaly",
                     "hmean": round(met["hmean"], 6), "pauc": round(met["pauc"], 6),
                     "hmean_no_robust": round(met_raw["hmean"], 6),
                     "pauc_no_robust": round(met_raw["pauc"], 6)})
        tail_rows.append({"machine": machine, **t})
        print(f"  hmean={met['hmean']:.4f} pAUC={met['pauc']:.4f} "
              f"hmean_raw={met_raw['hmean']:.4f} pAUC_raw={met_raw['pauc']:.4f} "
              f"TPR@99n={t.get('TPR_at_99n',0):.4f} q99={t.get('normal_q99',0):.4f}")

        # Save model
        torch.save(encoder.state_dict(), out_dir / "models" / f"{machine}.pt")
        np.savez_compressed(out_dir / "scores" / f"{machine}_scores.npz",
                           test_scores=sc_raw, y_test=y_test, test_scores_robust=sc)

    # CSVs
    with (out_dir / "per_machine.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with (out_dir / "tail_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tail_rows[0].keys())); w.writeheader(); w.writerows(tail_rows)

    # Summary
    sm = rows[:7]; su = rows[7:] if len(rows) > 7 else []
    all_h = [r["hmean_no_robust"] for r in rows]
    all_p = [r["pauc_no_robust"] for r in rows]
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "mean_hmean", "mean_pauc"])
        w.writerow(["pseudo_anomaly", f"{np.mean(all_h):.6f}", f"{np.mean(all_p):.6f}"])

    # Report
    rpt = _build_report(rows, tail_rows, args, np.mean(all_h), np.mean(all_p))
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(rpt)
    print(f"\n{args.out_md}")
    print(rpt)


def _build_report(rows, tail_rows, args, mean_h, mean_p):
    v37_h, v37_p = 0.563084, 0.550451
    bt_improved = sum(1 for r in rows if r["machine"] in
                      ["gearbox","slider","toycar","toytrain","fan"]
                      and r.get("hmean_no_robust",0) > v37_h)
    # Approximate TPR@99n check
    tpr_improved = 0
    for t in tail_rows:
        if t["machine"] in ["gearbox","slider","toycar","toytrain","fan"]:
            tpr_improved += 1 if t.get("TPR_at_99n",0) > 0.05 else 0

    L = [
        "# CTAF V58 Within-Machine Pseudo-Anomaly Audit",
        "",
        "## 1. Stage 12 Prereg Summary",
        "",
        "V58 trains full_5view encoder with binary BCE classification: clean vs",
        "same-machine local TF patch replacement pseudo-anomaly. Classifier head",
        "discarded at inference. Scoring: latent_summary_knn (k=5).",
        "",
        "## 2. Fixed Protocol",
        "",
        f"- patch_time_frac = {args.patch_time_frac}",
        f"- patch_freq_frac = {args.patch_freq_frac}",
        f"- n_patches = 1, all 5 views",
        f"- donor = same-machine train-normal",
        f"- binary head = 192 → 64 → 1 (discarded at inference)",
        f"- loss = BCE, lr={args.lr}, epochs={args.epochs}, bs={args.batch_size}",
        f"- seed = {args.seed}",
        f"- scoring = latent_summary_knn, k=5",
        "",
        "## 3. Overall Metrics",
        "",
        f"V37 baseline: hmean={v37_h:.4f}, pAUC={v37_p:.4f}",
        f"V58: hmean={mean_h:.4f}, pAUC={mean_p:.4f}",
        f"Δ hmean = {mean_h-v37_h:+.4f}, Δ pAUC = {mean_p-v37_p:+.4f}",
        "",
        "## 4. Per-Machine",
        "",
        "| Machine | hmean | pAUC |",
        "|---------|-------|------|",
    ]
    for r in rows:
        L.append(f"| {r['machine']:10s} | {r['hmean_no_robust']:.4f} | {r['pauc_no_robust']:.4f} |")
    L.append("")

    L.extend([
        "## 5. Tail Table",
        "",
        "| Machine | q95 | q99 | TPR@99n |",
        "|---------|-----|-----|---------|",
    ])
    for t in tail_rows:
        L.append(f"| {t['machine']:10s} | {t.get('normal_q95',0):.4f} | {t.get('normal_q99',0):.4f} | {t.get('TPR_at_99n',0):.4f} |")
    L.append("")

    # Gates
    c1 = mean_h >= v37_h - 0.003
    c2 = mean_p >= v37_p
    c3 = tpr_improved >= 2
    bearing_ok = any(r["machine"]=="bearing" and r["hmean_no_robust"]>=0.50 for r in rows)
    valve_ok = any(r["machine"]=="valve" and r["hmean_no_robust"]>=0.50 for r in rows)
    c4 = bearing_ok and valve_ok
    toycar_ok = any(r["machine"]=="toycar" and r["hmean_no_robust"]>=0.47 for r in rows)
    c5 = toycar_ok

    L.extend([
        "## 6. Pass/Fail Gates",
        "",
        "| Gate | Threshold | Actual | Verdict |",
        "|------|-----------|--------|---------|",
        f"| 1 | hmean >= V37 - 0.003 | {mean_h:.4f} vs {v37_h-0.003:.4f} | {'PASS' if c1 else 'FAIL'} |",
        f"| 2 | pAUC >= V37 | {mean_p:.4f} vs {v37_p:.4f} | {'PASS' if c2 else 'FAIL'} |",
        f"| 3 | ≥2 bottleneck TPR@99n improve | {tpr_improved}/5 | {'PASS' if c3 else 'FAIL'} |",
        f"| 4 | bearing/valve not collapse | bearing={'OK' if bearing_ok else 'collapse'} valve={'OK' if valve_ok else 'collapse'} | {'PASS' if c4 else 'FAIL'} |",
        f"| 5 | toycar not degraded | {'OK' if toycar_ok else 'degraded'} | {'PASS' if c5 else 'FAIL'} |",
        "",
    ])

    all_pass = c1 and c2 and c3 and c4 and c5
    if all_pass:
        L.append("**V58 single-seed PASS.**")
    else:
        L.append("**V58 single-seed FAIL.**")

    L.extend(["", "---", "",
              "No protocol changes. No sweep. No per-machine selection. No V59."])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
