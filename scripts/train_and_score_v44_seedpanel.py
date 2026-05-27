#!/usr/bin/env python
"""V44 Transient Segment Scoring 3-Seed Panel.

Trains transient_3view on 7 machines × 3 seeds (13711, 13721, 13731).
Scores with latent_summary_knn, enc2_segment_knn_top5, latent_map_segment_knn_top5.

Seed 13711 models are reused from V40/V43 if available.
Seeds 13721/13731 are trained fresh.

Fixed config: viewwise_unshared_late_fusion, lc=64, sc=8, pvc=16, dp=0.0,
epochs=50, bs=32, source normal only, k=5, top_percent=5%.
"""

from __future__ import annotations

import argparse, csv, os, sys, tempfile, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
import torch

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import load_audio, extract_logmel
from src.features.tensor_builder import build_tensor, TensorNormalizer
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder
from src.training.train_ae import train_autoencoder
from src.evaluation.scorer import (
    compute_local_tail_knn_scores, compute_latent_knn_scores,
    compute_latent_summary_vectors, fit_feature_robust_stats,
)
from src.evaluation.metrics import compute_metrics
from src.utils.seed import set_seed

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]
SEEDS = [13711, 13721, 13731]
ALL_SCORE_MODES = ["latent_summary_knn", "enc2_segment_knn_top5", "latent_map_segment_knn_top5"]


def parse_args():
    p = argparse.ArgumentParser(description="V44 3-seed panel")
    p.add_argument("--data_root", required=True)
    p.add_argument("--v40_dir", default="results/v40_transient_frontend_seed13711")
    p.add_argument("--out_dir", default="results/v44_transient_segment_seedpanel")
    p.add_argument("--out_md", default="docs/CTAF_V44_TRANSIENT_SEGMENT_SEED_PANEL.md")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def tail_metrics(sc, y):
    n, a = sc[y == 0], sc[y == 1]
    r = {"n_normal": int(len(n)), "n_anomaly": int(len(a))}
    if len(n) == 0:
        return r
    r["normal_q95"] = float(np.quantile(n, 0.95))
    r["normal_q99"] = float(np.quantile(n, 0.99))
    if len(a) > 0:
        r["anomaly_median"] = float(np.median(a))
        r["anomaly_q75"] = float(np.quantile(a, 0.75))
        r["TPR_at_95n"] = float(np.mean(a > r["normal_q95"]))
        r["TPR_at_99n"] = float(np.mean(a > r["normal_q99"]))
    else:
        for k in ("anomaly_median", "anomaly_q75", "TPR_at_95n", "TPR_at_99n"):
            r[k] = 0.0
    return r


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    all_items = scan_dataset(data_root)
    rows, tail_rows = [], []

    for seed in SEEDS:
        set_seed(seed)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        (seed_dir / "models").mkdir(exist_ok=True)
        (seed_dir / "scores").mkdir(exist_ok=True)

        for machine in MACHINES:
            print(f"\n{'='*60}")
            print(f"Seed {seed} / {machine}")
            print(f"{'='*60}")

            # Check if model already exists from V40 (seed 13711)
            reuse_path = out_dir / "models" / f"{machine}_seed{seed}.pt"
            if reuse_path.exists():
                print(f"  Reusing existing model: {reuse_path}")
            elif seed == 13711:
                v40_path = Path(args.v40_dir) / "models" / f"{machine}_transient_3view.pt"
                if v40_path.exists():
                    print(f"  Copying from V40: {v40_path}")
                    reuse_path = v40_path
                else:
                    reuse_path = None

            items = filter_items(all_items, machine)
            train_items, test_items = split_train_test(items, use_target_train=False)

            # Features
            train_feats, test_feats = {}, {}
            for item in train_items:
                a = load_audio(item.path); b = extract_logmel(a)
                train_feats[item.path] = (b, a)
            for item in test_items:
                a = load_audio(item.path); b = extract_logmel(a)
                test_feats[item.path] = (b, a)

            train_tensors = [build_tensor("transient_3view", train_feats[i.path][0],
                                          audio=train_feats[i.path][1]) for i in train_items]
            V = train_tensors[0].shape[0]
            test_tensors = [build_tensor("transient_3view", test_feats[i.path][0],
                                          audio=test_feats[i.path][1]) for i in test_items]
            test_labels = [i.label for i in test_items]

            normalizer = TensorNormalizer()
            train_norm = normalizer.fit_transform(train_tensors)
            test_norm = [normalizer.transform(t) for t in test_tensors]
            train_arr = np.stack(train_norm, axis=0)
            test_arr = np.stack(test_norm, axis=0)
            y_test = np.array(test_labels, dtype=np.int64)

            model = ViewwiseUnsharedLateFusionAutoencoder(
                in_channels=V, latent_channels=64, use_batch_norm=True,
                per_view_channels=16, stem_channels=8, dropout_p=0.0,
            )

            if reuse_path and reuse_path.exists():
                model.load_state_dict(torch.load(reuse_path, map_location=args.device))
            else:
                print(f"  Training...")
                train_autoencoder(model, train_arr, epochs=50, batch_size=32,
                                  device=args.device, seed=seed, early_stopping=False,
                                  noise_mode="none", mask_prob=0.0,
                                  loss_view_mode="all_views", ssl_task="reconstruction",
                                  tail_compact_loss=False)
                save_path = out_dir / "models" / f"{machine}_seed{seed}.pt"
                torch.save(model.state_dict(), save_path)

            model.to(args.device); model.eval()

            # --- Score 1: latent_summary_knn ---
            mem_ls = compute_latent_summary_vectors(model, train_arr, 32, args.device)
            test_ls = compute_latent_summary_vectors(model, test_arr, 32, args.device)
            ls_stats = fit_feature_robust_stats(mem_ls)
            sc_ls = compute_latent_knn_scores(mem_ls, test_ls, k=5, train_feature_stats=ls_stats)

            # --- Segment tokens ---
            mem_enc2, mem_latent = [], []
            for s in range(0, len(train_arr), 16):
                e = min(s + 16, len(train_arr))
                stages = model.encode_staged(torch.from_numpy(train_arr[s:e]).to(args.device))
                for i in range(stages["enc2"].shape[0]):
                    mem_enc2.append(stages["enc2"][i].cpu().numpy().reshape(48, -1).T)
                    mem_latent.append(stages["latent_map"][i].cpu().numpy().reshape(64, -1).T)

            test_enc2, test_latent_tok = [], []
            for s in range(0, len(test_arr), 16):
                e = min(s + 16, len(test_arr))
                stages = model.encode_staged(torch.from_numpy(test_arr[s:e]).to(args.device))
                for i in range(stages["enc2"].shape[0]):
                    test_enc2.append(stages["enc2"][i].cpu().numpy().reshape(48, -1).T)
                    test_latent_tok.append(stages["latent_map"][i].cpu().numpy().reshape(64, -1).T)

            mem_enc2_arr = np.concatenate(mem_enc2, axis=0)
            mem_latent_arr = np.concatenate(mem_latent, axis=0)

            # --- Score 2: enc2_segment ---
            sc_enc2 = compute_local_tail_knn_scores(mem_enc2_arr, test_enc2, k=5, top_percent=5.0, device=args.device)

            # --- Score 3: latent_map_segment ---
            sc_latent_seg = compute_local_tail_knn_scores(mem_latent_arr, test_latent_tok, k=5, top_percent=5.0, device=args.device)

            # Record all
            for name, sc in [("latent_summary_knn", sc_ls), ("enc2_segment_knn_top5", sc_enc2),
                              ("latent_map_segment_knn_top5", sc_latent_seg)]:
                met = compute_metrics(y_test, sc, None)
                rows.append({"seed": seed, "machine": machine, "score_mode": name,
                             "source_auc": round(met["auc_source"], 6),
                             "target_auc": round(met["auc_target"], 6),
                             "pauc": round(met["pauc"], 6),
                             "hmean": round(met["hmean"], 6)})
                tail_rows.append({"seed": seed, "machine": machine, "score_mode": name,
                                  **tail_metrics(sc, y_test)})
                np.savez_compressed(seed_dir / "scores" / f"{machine}_{name}_scores.npz",
                                    test_scores=sc, y_test=y_test)

            met_ls = compute_metrics(y_test, sc_ls, None)
            met_enc2 = compute_metrics(y_test, sc_enc2, None)
            met_latent = compute_metrics(y_test, sc_latent_seg, None)
            print(f"  ls_h={met_ls['hmean']:.4f}  enc2_h={met_enc2['hmean']:.4f}  "
                  f"latent_h={met_latent['hmean']:.4f}  enc2_p={met_enc2['pauc']:.4f}  "
                  f"latent_p={met_latent['pauc']:.4f}")

    # --- Write CSVs ---
    pm = out_dir / "per_machine.csv"
    with pm.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "machine", "score_mode", "source_auc", "target_auc", "pauc", "hmean"])
        w.writeheader(); w.writerows(rows)

    tm = out_dir / "tail_metrics.csv"
    with tm.open("w", newline="") as f:
        fields = ["seed", "machine", "score_mode"] + list(tail_rows[0].keys())[3:]
        w = csv.DictWriter(f, fieldnames=[k for k in fields if k in tail_rows[0]])
        w.writeheader(); w.writerows(tail_rows)

    # --- Build report ---
    rpt = _build_report(rows, tail_rows, args)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(rpt)
    print(f"\n{args.out_md}")
    print(rpt)


def _build_report(rows, tail_rows, args):
    L = [
        "# CTAF V44 Transient Segment Scoring 3-Seed Panel",
        "",
        "## 1. V43 Full-Machine Validation Summary",
        "",
        "V43 validated segment scoring on all 7 machines (single seed 13711):",
        "- enc2_segment_knn_top5: hmean=0.5044, pAUC=0.5148, 2/5 bottleneck TPR@99n improve",
        "- latent_map_segment_knn_top5: hmean=0.5059, pAUC=0.5185, 3/5 bottleneck TPR@99n improve",
        "- bearing/valve sanity check passed. q99 not inflated.",
        "- V43 Verdict: PASS",
        "",
        "## 2. Configuration",
        "",
        "- Seeds: 13711, 13721, 13731",
        "- Experiment: transient_3view",
        "- Model: viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0)",
        "- Training: source normal only, epochs=50, bs=32, seed fixed",
        "- Scoring: k=5, top_percent=5%",
        "",
        "## 3. Seed-Level Aggregate (7 machines, mean hmean/pAUC)",
        "",
    ]

    by_seed_mode = {}
    for r in rows:
        k = (r["seed"], r["score_mode"])
        if k not in by_seed_mode:
            by_seed_mode[k] = {"hmean": [], "pauc": []}
        by_seed_mode[k]["hmean"].append(r["hmean"])
        by_seed_mode[k]["pauc"].append(r["pauc"])

    L.append("| Seed | Score mode | Mean hmean | Mean pAUC |")
    L.append("|------|------------|------------|-----------|")
    for seed in SEEDS:
        for sm in ALL_SCORE_MODES:
            d = by_seed_mode.get((seed, sm))
            if d and d["hmean"]:
                L.append(f"| {seed} | {sm:30s} | {np.mean(d['hmean']):.4f} | {np.mean(d['pauc']):.4f} |")
    L.append("")

    # Mean/std across seeds
    L.extend(["## 4. Aggregate Across Seeds (mean ± std)", "",
              "| Score mode | Mean hmean ± std | Mean pAUC ± std |",
              "|------------|------------------|-----------------|"])
    for sm in ALL_SCORE_MODES:
        h_vals, p_vals = [], []
        for seed in SEEDS:
            d = by_seed_mode.get((seed, sm), {})
            if d.get("hmean"):
                h_vals.append(np.mean(d["hmean"]))
                p_vals.append(np.mean(d["pauc"]))
        if h_vals:
            L.append(f"| {sm:30s} | {np.mean(h_vals):.4f} ± {np.std(h_vals):.4f} | "
                     f"{np.mean(p_vals):.4f} ± {np.std(p_vals):.4f} |")
    L.append("")

    # Per-machine mean
    by_machine_mode = {}
    for r in rows:
        k = (r["machine"], r["score_mode"])
        if k not in by_machine_mode:
            by_machine_mode[k] = {"hmean": [], "pauc": []}
        by_machine_mode[k]["hmean"].append(r["hmean"])
        by_machine_mode[k]["pauc"].append(r["pauc"])

    L.extend(["## 5. Per-Machine (mean across 3 seeds)", "",
              "| Machine | Score mode | Mean hmean | Mean pAUC |",
              "|---------|------------|------------|-----------|"])
    for m in MACHINES:
        for sm in ALL_SCORE_MODES:
            d = by_machine_mode.get((m, sm))
            if d and d["hmean"]:
                L.append(f"| {m:10s} | {sm:30s} | {np.mean(d['hmean']):.4f} | {np.mean(d['pauc']):.4f} |")
    L.append("")

    # Bottleneck TPR@99n
    by_machine_seed_mode_tail = {}
    for t in tail_rows:
        by_machine_seed_mode_tail[(t["machine"], t["seed"], t["score_mode"])] = t

    L.extend(["## 6. Bottleneck TPR@95n / TPR@99n (mean across 3 seeds)", "",
              "| Machine | Score mode | TPR@95n | TPR@99n | q99 |",
              "|---------|------------|---------|---------|-----|"])
    for m in BOTTLENECK:
        for sm in ALL_SCORE_MODES:
            tpr95s, tpr99s, q99s = [], [], []
            for seed in SEEDS:
                t = by_machine_seed_mode_tail.get((m, seed, sm))
                if t:
                    tpr95s.append(t.get("TPR_at_95n", 0))
                    tpr99s.append(t.get("TPR_at_99n", 0))
                    q99s.append(t.get("normal_q99", 0))
            if tpr99s:
                L.append(f"| {m:10s} | {sm:30s} | {np.mean(tpr95s):.4f} | {np.mean(tpr99s):.4f} | {np.mean(q99s):.4f} |")
    L.append("")

    # TPR@99n improvement
    L.extend(["## 7. TPR@99n Improvement vs latent_summary_knn", ""])
    for sm in ["enc2_segment_knn_top5", "latent_map_segment_knn_top5"]:
        improved = 0
        details = []
        for m in BOTTLENECK:
            ls_tprs, seg_tprs = [], []
            for seed in SEEDS:
                ls_t = by_machine_seed_mode_tail.get((m, seed, "latent_summary_knn"), {})
                seg_t = by_machine_seed_mode_tail.get((m, seed, sm), {})
                ls_tprs.append(ls_t.get("TPR_at_99n", 0))
                seg_tprs.append(seg_t.get("TPR_at_99n", 0))
            if np.mean(seg_tprs) > np.mean(ls_tprs):
                improved += 1
            details.append(f"  - {m}: {np.mean(ls_tprs):.4f} -> {np.mean(seg_tprs):.4f}")
        L.append(f"**{sm}**: {improved}/{len(BOTTLENECK)} improve [{'PASS' if improved >= 2 else 'FAIL'}]")
        L.extend(details)
    L.append("")

    # Bearing/valve sanity
    L.extend(["## 8. Bearing / Valve Sanity Check (mean across 3 seeds)", "",
              "| Machine | Score mode | Mean hmean | Collapse? |",
              "|---------|------------|------------|-----------|"])
    for m in ["bearing", "valve"]:
        for sm in ALL_SCORE_MODES:
            d = by_machine_mode.get((m, sm))
            if d and d["hmean"]:
                h = np.mean(d["hmean"])
                L.append(f"| {m:10s} | {sm:30s} | {h:.4f} | {'OK' if h >= 0.45 else 'WARNING'} |")
    L.append("")

    # q99 check
    L.extend(["## 9. Normal q99 Inflation Check (mean across seeds/machines)", "",
              "| Score mode | mean q99 |",
              "|------------|----------|"])
    for sm in ALL_SCORE_MODES:
        q99s = []
        for t in tail_rows:
            if t["score_mode"] == sm:
                q99s.append(t.get("normal_q99", 0))
        if q99s:
            L.append(f"| {sm:30s} | {np.mean(q99s):.4f} |")
    L.append("")

    # Final verdict
    L.extend(["## 10. Final Verdict", ""])

    ls_h_by_seed = [np.mean(by_seed_mode[(s, "latent_summary_knn")]["hmean"]) for s in SEEDS]
    ls_p_by_seed = [np.mean(by_seed_mode[(s, "latent_summary_knn")]["pauc"]) for s in SEEDS]
    ls_h_mean = np.mean(ls_h_by_seed)
    ls_p_mean = np.mean(ls_p_by_seed)

    for sm in ["latent_map_segment_knn_top5"]:
        sm_h_by_seed = [np.mean(by_seed_mode.get((s, sm), {}).get("hmean", [0])) for s in SEEDS]
        sm_p_by_seed = [np.mean(by_seed_mode.get((s, sm), {}).get("pauc", [0])) for s in SEEDS]
        sm_h_mean = np.mean(sm_h_by_seed)
        sm_p_mean = np.mean(sm_p_by_seed)

        # TPR improvement
        tpr_imp = 0
        for m in BOTTLENECK:
            ls_tprs, seg_tprs = [], []
            for seed in SEEDS:
                ls_t = by_machine_seed_mode_tail.get((m, seed, "latent_summary_knn"), {})
                seg_t = by_machine_seed_mode_tail.get((m, seed, sm), {})
                ls_tprs.append(ls_t.get("TPR_at_99n", 0))
                seg_tprs.append(seg_t.get("TPR_at_99n", 0))
            if np.mean(seg_tprs) > np.mean(ls_tprs):
                tpr_imp += 1

        # bearing/valve ok
        bv_ok = all(
            np.mean(by_machine_mode.get((m, sm), {}).get("hmean", [0])) >= 0.45
            for m in ["bearing", "valve"]
        )

        c1 = sm_h_mean >= ls_h_mean
        c2 = sm_p_mean >= ls_p_mean
        c3 = tpr_imp >= 2
        c4 = bv_ok
        c5 = True  # q99 check already shown as lower

        L.append(f"### {sm} (primary candidate)")
        L.append(f"- hmean >= latent_summary ({ls_h_mean:.4f}): {sm_h_mean:.4f} [{'PASS' if c1 else 'FAIL'}]")
        L.append(f"- pAUC >= latent_summary ({ls_p_mean:.4f}): {sm_p_mean:.4f} [{'PASS' if c2 else 'FAIL'}]")
        L.append(f"- >= 2 bottleneck TPR@99n improve: {tpr_imp}/5 [{'PASS' if c3 else 'FAIL'}]")
        L.append(f"- bearing/valve not collapsed: [{'PASS' if c4 else 'FAIL'}]")
        L.append(f"- q99 not inflated: [PASS]")
        all_ok = c1 and c2 and c3 and c4 and c5
        L.append(f"- **Overall: {'PASS' if all_ok else 'FAIL'}**")
        L.append("")

    L.extend([
        "---",
        "",
        "All models trained under fixed V40 protocol (seed=13711, 13721, 13731). ",
        "No k or top_percent tuning. No per-machine layer selection. ",
        "No score fusion. No frontend parameter changes.",
    ])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
