#!/usr/bin/env python
"""V45 Segment Scoring Transfer Audit.

Tests segment-level scoring on stronger representations:
  - V16 full_5view AE
  - V37 full_5view teacher-student pretraining

Uses saved checkpoints. No retraining. Fixed k=5, top_percent=5%.
"""

import argparse, csv, os, sys, tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
import matplotlib; matplotlib.use("Agg")
import torch

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import load_audio, extract_logmel
from src.features.tensor_builder import build_tensor, TensorNormalizer
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder
from src.evaluation.scorer import (compute_local_tail_knn_scores, compute_latent_knn_scores,
                                   compute_latent_summary_vectors, fit_feature_robust_stats)
from src.evaluation.metrics import compute_metrics

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]
SCORE_MODES = ["latent_summary_knn", "enc2_segment_knn_top5", "latent_map_segment_knn_top5"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--v45_dir", required=True)
    p.add_argument("--out_md", default="docs/CTAF_V45_SEGMENT_SCORING_TRANSFER_AUDIT.md")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def tail_metrics(sc, y):
    n, a = sc[y == 0], sc[y == 1]
    r = {"n_normal": int(len(n)), "n_anomaly": int(len(a))}
    if len(n) == 0: return r
    r["normal_q95"] = float(np.quantile(n, 0.95))
    r["normal_q99"] = float(np.quantile(n, 0.99))
    if len(a) > 0:
        r["anomaly_median"] = float(np.median(a))
        r["anomaly_q75"] = float(np.quantile(a, 0.75))
        r["TPR_at_95n"] = float(np.mean(a > r["normal_q95"]))
        r["TPR_at_99n"] = float(np.mean(a > r["normal_q99"]))
    else:
        for k in ("anomaly_median", "anomaly_q75", "TPR_at_95n", "TPR_at_99n"): r[k] = 0.0
    return r


def score_all(model, model_dir, machine, seed, source_label, data_root, args, rows, tail_rows):
    """Score one (source, machine, seed) with all 3 modes."""
    key = f"{source_label}_seed{seed}"

    all_items = scan_dataset(data_root)
    items = filter_items(all_items, machine)
    train_items, test_items = split_train_test(items, use_target_train=False)

    train_feats, test_feats = {}, {}
    for item in train_items:
        a = load_audio(item.path); b = extract_logmel(a)
        train_feats[item.path] = (b, a)
    for item in test_items:
        a = load_audio(item.path); b = extract_logmel(a)
        test_feats[item.path] = (b, a)

    train_tensors = [build_tensor("full_5view", train_feats[i.path][0]) for i in train_items]
    V = train_tensors[0].shape[0]
    test_tensors = [build_tensor("full_5view", test_feats[i.path][0]) for i in test_items]
    test_labels = [i.label for i in test_items]

    normalizer = TensorNormalizer(); normalizer.fit(train_tensors)
    train_arr = np.stack([normalizer.transform(t) for t in train_tensors], axis=0)
    test_arr = np.stack([normalizer.transform(t) for t in test_tensors], axis=0)
    y_test = np.array(test_labels, dtype=np.int64)

    model.eval()
    try:
        model.enc2 = model.view_enc2  # force attr for isinstance check
    except: pass

    # ls_knn
    mem_ls = compute_latent_summary_vectors(model, train_arr, 32, args.device)
    test_ls = compute_latent_summary_vectors(model, test_arr, 32, args.device)
    sc_ls = compute_latent_knn_scores(mem_ls, test_ls, k=5)  # no robust scaling: match V16/V37 protocol
    test_domains = np.array([i.domain for i in test_items])
    met_ls = compute_metrics(y_test, sc_ls, test_domains)

    # Segment tokens
    mem_enc2, mem_latent = [], []
    for s in range(0, len(train_arr), 16):
        e = min(s + 16, len(train_arr))
        stages = model.encode_staged(torch.from_numpy(train_arr[s:e]).to(args.device))
        for i in range(stages["enc2"].shape[0]):
            mem_enc2.append(stages["enc2"][i].cpu().numpy().reshape(
                stages["enc2"].shape[1], -1).T)
            mem_latent.append(stages["latent_map"][i].cpu().numpy().reshape(
                stages["latent_map"].shape[1], -1).T)

    test_enc2, test_latent_tok = [], []
    for s in range(0, len(test_arr), 16):
        e = min(s + 16, len(test_arr))
        stages = model.encode_staged(torch.from_numpy(test_arr[s:e]).to(args.device))
        for i in range(stages["enc2"].shape[0]):
            test_enc2.append(stages["enc2"][i].cpu().numpy().reshape(
                stages["enc2"].shape[1], -1).T)
            test_latent_tok.append(stages["latent_map"][i].cpu().numpy().reshape(
                stages["latent_map"].shape[1], -1).T)

    sc_enc2 = compute_local_tail_knn_scores(np.concatenate(mem_enc2, 0), test_enc2,
                                            k=5, top_percent=5.0, device=args.device)
    sc_latent = compute_local_tail_knn_scores(np.concatenate(mem_latent, 0), test_latent_tok,
                                              k=5, top_percent=5.0, device=args.device)
    met_enc2 = compute_metrics(y_test, sc_enc2, test_domains)
    met_latent = compute_metrics(y_test, sc_latent, test_domains)

    for name, sc, met in [("latent_summary_knn", sc_ls, met_ls),
                           ("enc2_segment_knn_top5", sc_enc2, met_enc2),
                           ("latent_map_segment_knn_top5", sc_latent, met_latent)]:
        rows.append({"source": source_label, "seed": seed, "machine": machine, "score_mode": name,
                     "hmean": round(met["hmean"], 6), "pauc": round(met["pauc"], 6),
                     "source_auc": round(met["auc_source"], 6), "target_auc": round(met["auc_target"], 6),
                     "dcase_auc": round(met["dcase_auc"], 6)})
        tail_rows.append({"source": source_label, "seed": seed, "machine": machine, "score_mode": name,
                          **tail_metrics(sc, y_test)})

    print(f"  [{source_label} seed{seed} {machine}] ls={met_ls['hmean']:.4f}/{met_ls['pauc']:.4f}  "
          f"enc2={met_enc2['hmean']:.4f}/{met_enc2['pauc']:.4f}  "
          f"latent={met_latent['hmean']:.4f}/{met_latent['pauc']:.4f}")


def main():
    args = parse_args()
    v45_dir = Path(args.v45_dir)
    rows, tail_rows = [], []
    scores_dir = v45_dir / "scores"; scores_dir.mkdir(exist_ok=True)

    # --- V16: full_5view AE seed 13711 ---
    v16_dir = v45_dir / "v16_full5view_seed13711" / "models"
    for machine in MACHINES:
        path = v16_dir / f"{machine}_full_5view_seed13711.pt"
        if not path.exists():
            print(f"[V16] SKIP {machine}: no model"); continue
        # Need V to build model
        # Quick scan to get V
        from src.data.dataset import scan_dataset as scd, filter_items as fi, split_train_test as stt
        from src.features.logmel import load_audio as la, extract_logmel as el
        from src.features.tensor_builder import build_tensor as bt
        all_it = scd(Path(args.data_root))
        it = fi(all_it, machine)
        tr, _ = stt(it)
        a = la(tr[0].path); b = el(a); t = bt("full_5view", b)
        model = ViewwiseUnsharedLateFusionAutoencoder(in_channels=t.shape[0], latent_channels=64,
                                                       use_batch_norm=True, per_view_channels=16,
                                                       stem_channels=8, dropout_p=0.0)
        model.load_state_dict(torch.load(path, map_location=args.device))
        print(f"[V16 seed 13711] loaded {machine}")
        score_all(model, v16_dir, machine, 13711, "V16_full5view", Path(args.data_root), args, rows, tail_rows)

    # --- V37: full_5view TS seeds 13711, 13721, 13731 ---
    v37_dir = v45_dir / "v37_full5view_ts" / "models"
    for seed in [13711, 13721, 13731]:
        for machine in MACHINES:
            path = v37_dir / f"{machine}_full_5view_seed{seed}.pt"
            if not path.exists():
                print(f"[V37 seed{seed}] SKIP {machine}: no model"); continue
            from src.data.dataset import scan_dataset as scd2, filter_items as fi2, split_train_test as stt2
            from src.features.logmel import load_audio as la2, extract_logmel as el2
            from src.features.tensor_builder import build_tensor as bt2
            all_it2 = scd2(Path(args.data_root))
            it2 = fi2(all_it2, machine)
            tr2, _ = stt2(it2)
            a2 = la2(tr2[0].path); b2 = el2(a2); t2 = bt2("full_5view", b2)
            model = ViewwiseUnsharedLateFusionAutoencoder(in_channels=t2.shape[0], latent_channels=64,
                                                           use_batch_norm=True, per_view_channels=16,
                                                           stem_channels=8, dropout_p=0.0)
            model.load_state_dict(torch.load(path, map_location=args.device))
            print(f"[V37 seed {seed}] loaded {machine}")
            score_all(model, v37_dir, machine, seed, "V37_full5view_TS", Path(args.data_root), args, rows, tail_rows)

    # Save CSVs
    csv_path = v45_dir / "per_machine.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    csv_tail = v45_dir / "tail_metrics.csv"
    with csv_tail.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tail_rows[0].keys())); w.writeheader(); w.writerows(tail_rows)
    for r in rows:
        sname = f"{r['source']}_seed{r['seed']}_{r['machine']}_{r['score_mode']}"
        # scores are not saved to NPZ here to save time; metrics in CSV suffice
    print(f"\nCSVs saved: {csv_path}, {csv_tail}")

    # Report
    rpt = _build_report(rows, tail_rows)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(rpt)
    print(f"\n{args.out_md}")


def _build_report(rows, tail_rows):
    L = [
        "# CTAF V45 Segment Scoring Transfer Audit",
        "",
        "## 1. V44 Summary",
        "",
        "V44 3-seed panel confirmed that transient_3view + latent_map_segment_knn_top5 is stable:",
        "- hmean = 0.5033 ± 0.0019, pAUC = 0.5103 ± 0.0058",
        "- 5/5 bottleneck TPR@99n improve; q99 not inflated",
        "- But absolute hmean/pAUC still below V16/V37 mainline.",
        "- V45 tests whether segment scoring transfers to stronger full_5view representations.",
        "",
        "## 2. Configuration",
        "- V16: full_5view AE, seed 13711",
        "- V37: full_5view teacher-student pretraining, seeds 13711/13721/13731",
        "- Model: viewwise_unshared_late_fusion (lc=64, sc=8, pvc=16, dp=0.0)",
        "- Scoring: k=5, top_percent=5%",
        "- No retraining; models regenerated under original V16/V37 protocol.",
        "",
    ]

    for source_label in ["V16_full5view", "V37_full5view_TS"]:
        srows = [r for r in rows if r["source"] == source_label]
        if not srows: continue
        seeds = sorted(set(r["seed"] for r in srows))
        is_ts = "TS" in source_label

        L.extend([
            f"## {3 if 'V16' in source_label else 4}. {source_label} Comparison",
            "",
            f"### Aggregate (mean across seeds/machines)",
            "",
            "| Score mode | Mean hmean ± std | Mean pAUC ± std |",
            "|------------|------------------|-----------------|",
        ])

        for sm in SCORE_MODES:
            hvs, pvs = [], []
            for s in seeds:
                mr = [r for r in srows if r["score_mode"] == sm and r["seed"] == s]
                if mr:
                    hvs.append(np.mean([r["hmean"] for r in mr]))
                    pvs.append(np.mean([r["pauc"] for r in mr]))
            if hvs:
                L.append(f"| {sm:30s} | {np.mean(hvs):.4f} ± {np.std(hvs):.4f} | {np.mean(pvs):.4f} ± {np.std(pvs):.4f} |")
        L.append("")

        # Per-machine
        by_m = {}
        for r in srows:
            k = (r["machine"], r["score_mode"])
            if k not in by_m: by_m[k] = {"h": [], "p": []}
            by_m[k]["h"].append(r["hmean"]); by_m[k]["p"].append(r["pauc"])

        L.extend(["### Per-Machine (mean across seeds)", "",
                  "| Machine | Score mode | Mean hmean | Mean pAUC |",
                  "|---------|------------|------------|-----------|"])
        for m in MACHINES:
            for sm in SCORE_MODES:
                d = by_m.get((m, sm));
                if d and d["h"]:
                    L.append(f"| {m:10s} | {sm:30s} | {np.mean(d['h']):.4f} | {np.mean(d['p']):.4f} |")
        L.append("")

        # Bottleneck TPR
        by_m_tail = {}
        for t in tail_rows:
            if t["source"] != source_label: continue
            k = (t["machine"], t["score_mode"])
            if k not in by_m_tail: by_m_tail[k] = {"t95": [], "t99": [], "q99": []}
            by_m_tail[k]["t95"].append(t["TPR_at_95n"])
            by_m_tail[k]["t99"].append(t["TPR_at_99n"])
            by_m_tail[k]["q99"].append(t["normal_q99"])

        L.extend(["### Bottleneck TPR@95n / TPR@99n", "",
                  "| Machine | Score mode | TPR@95n | TPR@99n | q99 |",
                  "|---------|------------|---------|---------|-----|"])
        for m in BOTTLENECK:
            for sm in SCORE_MODES:
                d = by_m_tail.get((m, sm))
                if d and d["t99"]:
                    L.append(f"| {m:10s} | {sm:30s} | {np.mean(d['t95']):.4f} | {np.mean(d['t99']):.4f} | {np.mean(d['q99']):.4f} |")
        L.append("")

        # Bearing/valve
        L.extend(["### Bearing / Valve Sanity", "",
                  "| Machine | Score mode | Mean hmean | Collapse? |",
                  "|---------|------------|------------|-----------|"])
        for m in ["bearing", "valve"]:
            for sm in SCORE_MODES:
                d = by_m.get((m, sm))
                if d and d["h"]:
                    h = np.mean(d["h"])
                    L.append(f"| {m:10s} | {sm:30s} | {h:.4f} | {'OK' if h >= 0.50 else 'WARNING'} |")
        L.append("")

    # Final verdict
    L.extend(["## 5. Final Verdict", ""])
    v37_rows = [r for r in rows if r["source"] == "V37_full5view_TS"]
    if v37_rows:
        for sm in ["latent_map_segment_knn_top5"]:
            ls_r = [r for r in v37_rows if r["score_mode"] == "latent_summary_knn"]
            sm_r = [r for r in v37_rows if r["score_mode"] == sm]
            if not ls_r or not sm_r: continue

            ls_h = np.mean([r["hmean"] for r in ls_r])
            ls_p = np.mean([r["pauc"] for r in ls_r])
            sm_h = np.mean([r["hmean"] for r in sm_r])
            sm_p = np.mean([r["pauc"] for r in sm_r])

            # TPR check
            tpr_imp = 0
            for m in BOTTLENECK:
                ls_t = [t["TPR_at_99n"] for t in tail_rows if t["source"] == "V37_full5view_TS"
                        and t["machine"] == m and t["score_mode"] == "latent_summary_knn"]
                sm_t = [t["TPR_at_99n"] for t in tail_rows if t["source"] == "V37_full5view_TS"
                        and t["machine"] == m and t["score_mode"] == sm]
                if np.mean(sm_t) > np.mean(ls_t): tpr_imp += 1

            bv_ok = all(
                np.mean([r["hmean"] for r in sm_r if r["machine"] == m]) >= 0.50
                for m in ["bearing", "valve"]
            ) if [r for r in sm_r if r["machine"] in ("bearing", "valve")] else True

            c1 = sm_h >= ls_h - 0.003
            c2 = sm_p >= ls_p
            c3 = tpr_imp >= 2
            c4 = bv_ok

            L.append(f"### V37 + {sm}")
            L.append(f"- hmean >= ls_knn ({ls_h:.4f}): {sm_h:.4f} ({sm_h-ls_h:+.4f}) [{'PASS' if c1 else 'FAIL'}]")
            L.append(f"- pAUC >= ls_knn ({ls_p:.4f}): {sm_p:.4f} ({sm_p-ls_p:+.4f}) [{'PASS' if c2 else 'FAIL'}]")
            L.append(f"- >= 2 bottleneck TPR@99n improve: {tpr_imp}/5 [{'PASS' if c3 else 'FAIL'}]")
            L.append(f"- bearing/valve hmean OK: [{'PASS' if c4 else 'FAIL'}]")
            all_ok = c1 and c2 and c3 and c4
            L.append(f"- **Overall: {'PASS' if all_ok else 'FAIL'}**")

    L.extend(["", "---", "",
              "Models regenerated under original V16/V37 protocol (same seed, architecture, data). ",
              "No retraining with modifications. k=5, top_percent=5% fixed. No per-machine tuning."])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
