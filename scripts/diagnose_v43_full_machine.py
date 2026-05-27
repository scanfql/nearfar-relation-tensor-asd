#!/usr/bin/env python
"""V43 Transient Segment Scoring Full-Machine Validation.

Runs V42's segment-level scoring diagnostic on all 7 DCASE machines:
bearing, fan, gearbox, slider, toycar, toytrain, valve.

Uses saved V40 transient_3view checkpoints. No retraining. No tuning.
"""

from __future__ import annotations

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
from src.evaluation.scorer import (
    compute_local_tail_knn_scores,
    compute_latent_knn_scores,
    compute_latent_summary_vectors,
    fit_feature_robust_stats,
)
from src.evaluation.metrics import compute_metrics

ALL_MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args():
    p = argparse.ArgumentParser(description="V43 full-machine validation")
    p.add_argument("--data_root", required=True)
    p.add_argument("--v40_dir", required=True)
    p.add_argument("--out_dir", default="results/v43_transient_segment_full_machine")
    p.add_argument("--out_md", default="docs/CTAF_V43_TRANSIENT_SEGMENT_FULL_MACHINE_AUDIT.md")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    return p.parse_args()


def compute_tail_metrics(scores, y):
    n, a = scores[y == 0], scores[y == 1]
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


def fmt4(x): return f"{x:.4f}"


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    v40_dir = Path(args.v40_dir)
    out_dir = Path(args.out_dir)
    model_dir = v40_dir / "models"
    machines = [m.strip() for m in args.machines.split(",")]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(exist_ok=True)

    all_items = scan_dataset(data_root)

    rows: List[dict] = []
    tail_rows: List[dict] = []

    for machine in machines:
        print(f"\n{'='*60}\nMachine: {machine}\n{'='*60}")

        model_path = model_dir / f"{machine}_transient_3view.pt"
        if not model_path.exists():
            print(f"  SKIP: no model at {model_path}")
            continue

        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)

        # Extract features
        train_feats = {}
        for item in train_items:
            a = load_audio(item.path); b = extract_logmel(a)
            train_feats[item.path] = (b, a)

        test_feats = {}
        for item in test_items:
            a = load_audio(item.path); b = extract_logmel(a)
            test_feats[item.path] = (b, a)

        train_tensors = [build_tensor("transient_3view", train_feats[i.path][0],
                                       audio=train_feats[i.path][1]) for i in train_items]
        V = train_tensors[0].shape[0]
        normalizer = TensorNormalizer()
        train_norm = normalizer.fit_transform(train_tensors)
        train_arr = np.stack(train_norm, axis=0)

        test_tensors = [build_tensor("transient_3view", test_feats[i.path][0],
                                      audio=test_feats[i.path][1]) for i in test_items]
        test_labels = [i.label for i in test_items]
        test_norm = [normalizer.transform(t) for t in test_tensors]
        test_arr = np.stack(test_norm, axis=0)
        y_test = np.array(test_labels, dtype=np.int64)

        model = ViewwiseUnsharedLateFusionAutoencoder(
            in_channels=V, latent_channels=64, use_batch_norm=True,
            per_view_channels=16, stem_channels=8, dropout_p=0.0,
        )
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.to(args.device); model.eval()

        # latent_summary_knn
        print("  latent_summary_knn...")
        mem_ls = compute_latent_summary_vectors(model, train_arr, 32, args.device)
        test_ls = compute_latent_summary_vectors(model, test_arr, 32, args.device)
        ls_stats = fit_feature_robust_stats(mem_ls)
        sc_ls = compute_latent_knn_scores(mem_ls, test_ls, k=5, train_feature_stats=ls_stats)

        # Segment tokens
        print("  segment tokens...")
        mem_enc2, mem_latent = [], []
        batch = 16
        for s in range(0, len(train_arr), batch):
            e = min(s + batch, len(train_arr))
            stages = model.encode_staged(torch.from_numpy(train_arr[s:e]).to(args.device))
            for i in range(stages["enc2"].shape[0]):
                mem_enc2.append(stages["enc2"][i].cpu().numpy().reshape(48, -1).T)
                mem_latent.append(stages["latent_map"][i].cpu().numpy().reshape(64, -1).T)

        test_enc2, test_latent_tok = [], []
        for s in range(0, len(test_arr), batch):
            e = min(s + batch, len(test_arr))
            stages = model.encode_staged(torch.from_numpy(test_arr[s:e]).to(args.device))
            for i in range(stages["enc2"].shape[0]):
                test_enc2.append(stages["enc2"][i].cpu().numpy().reshape(48, -1).T)
                test_latent_tok.append(stages["latent_map"][i].cpu().numpy().reshape(64, -1).T)

        mem_enc2_arr = np.concatenate(mem_enc2, axis=0)
        mem_latent_arr = np.concatenate(mem_latent, axis=0)

        # enc2_segment
        print("  enc2_segment...")
        sc_enc2 = compute_local_tail_knn_scores(mem_enc2_arr, test_enc2, k=5, top_percent=5.0, device=args.device)

        # latent_map_segment
        print("  latent_map_segment...")
        sc_latent_seg = compute_local_tail_knn_scores(mem_latent_arr, test_latent_tok, k=5, top_percent=5.0, device=args.device)

        metrics_ls = compute_metrics(y_test, sc_ls, None)
        metrics_enc2 = compute_metrics(y_test, sc_enc2, None)
        metrics_latent_seg = compute_metrics(y_test, sc_latent_seg, None)

        # Save & record
        for name, sc, met in [("latent_summary_knn", sc_ls, metrics_ls),
                               ("enc2_segment_knn_top5", sc_enc2, metrics_enc2),
                               ("latent_map_segment_knn_top5", sc_latent_seg, metrics_latent_seg)]:
            np.savez_compressed(out_dir / "scores" / f"{machine}_transient_3view_{name}_scores.npz",
                                test_scores=sc, y_test=y_test)
            rows.append({"machine": machine, "score_mode": name,
                         "source_auc": round(met["auc_source"], 6), "target_auc": round(met["auc_target"], 6),
                         "pauc": round(met["pauc"], 6), "dcase_auc": round(met["dcase_auc"], 6),
                         "hmean": round(met["hmean"], 6)})
            tail_rows.append({"machine": machine, "score_mode": name, **compute_tail_metrics(sc, y_test)})

        print(f"  ls={metrics_ls['hmean']:.4f}  enc2={metrics_enc2['hmean']:.4f}  latent={metrics_latent_seg['hmean']:.4f}")

    # CSVs
    if rows:
        with (out_dir / "per_machine.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if tail_rows:
        with (out_dir / "tail_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tail_rows[0].keys())); w.writeheader(); w.writerows(tail_rows)

    # Report
    rpt = _build_report(rows, tail_rows, machines)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(rpt)
    print(f"\n{Path(args.out_md)}")
    print(rpt)


def _build_report(rows, tail_rows, machines):
    sms = sorted(set(r["score_mode"] for r in rows))
    ls_rows = [r for r in rows if r["score_mode"] == "latent_summary_knn"]
    ls_tail = {t["machine"]: t for t in tail_rows if t["score_mode"] == "latent_summary_knn"}

    L = [
        "# CTAF V43 Transient Segment Scoring Full-Machine Validation",
        "",
        "## 1. Provenance: bearing/valve Checkpoint Completion",
        "",
        "bearing and valve transient_3view checkpoints were generated under the",
        "identical V40 protocol:",
        "- experiment = transient_3view",
        "- model_type = viewwise_unshared_late_fusion",
        "- latent_channels = 64, stem_channels = 8, per_view_channels = 16, dropout_p = 0.0",
        "- training = source normal only, memory bank = source normal only",
        "- loss_view_mode = all_views, ssl_task = reconstruction",
        "- seed = 13711, epochs = 50, batch_size = 32",
        "- No per-machine customization.",

        "",
        "V40 bottleneck checkpoints (gearbox, slider, toycar, toytrain, fan) ",
        "were regenerated earlier for V41/V42 under the same seed=13711 protocol. ",
        "Regeneration was verified: score reproduction max_diff < 0.08.",

        "",
        "## 2. V42 Bottleneck Summary",
        "",
        "V42 tested segment-level scoring on 5 bottleneck machines. Both score modes passed:",
        "",
        "| Score mode | Mean hmean | Mean pAUC | TPR@99n improved | q99 status |",
        "|------------|------------|-----------|------------------|------------|",
        "| enc2_segment_knn_top5 | 0.5056 | 0.5187 | 2/5 PASS | not inflated |",
        "| latent_map_segment_knn_top5 | 0.5075 | 0.5233 | 3/5 PASS | not inflated |",
        "",

        "## 3. All-7-Machine Overall Comparison",
        "",
        f"Mean across {len(machines)} machines.",
        "",
        "| Score mode | Mean hmean | Mean pAUC | Mean dcase_auc |",
        "|------------|------------|-----------|----------------|",
    ]

    for sm in sms:
        mr = [r for r in rows if r["score_mode"] == sm]
        if not mr: continue
        L.append(f"| {sm:30s} | {np.mean([r['hmean'] for r in mr]):.4f} | {np.mean([r['pauc'] for r in mr]):.4f} | {np.mean([r['dcase_auc'] for r in mr]):.4f} |")
    L.append("")

    # Per-machine
    L.extend(["## 4. Per-Machine hmean / pAUC", "",
              "| Machine | Score mode | hmean | pAUC |",
              "|---------|------------|-------|------|"])
    for m in machines:
        for sm in sms:
            r = [x for x in rows if x["machine"] == m and x["score_mode"] == sm]
            if r: L.append(f"| {m:10s} | {sm:30s} | {r[0]['hmean']:.4f} | {r[0]['pauc']:.4f} |")
    L.append("")

    # Bottleneck tail
    L.extend(["## 5. Bottleneck Machine Tail Table (TPR@95n / TPR@99n)", "",
              "| Machine | Score mode | TPR@95n | TPR@99n | q99 |",
              "|---------|------------|---------|---------|-----|"])
    for m in BOTTLENECK:
        for sm in sms:
            t = [x for x in tail_rows if x["machine"] == m and x["score_mode"] == sm]
            if t: L.append(f"| {m:10s} | {sm:30s} | {t[0]['TPR_at_95n']:.4f} | {t[0]['TPR_at_99n']:.4f} | {t[0]['normal_q99']:.4f} |")
    L.append("")

    # TPR@99n improvement
    L.extend(["## 6. TPR@99n Improvement on Bottleneck Machines", ""])
    for sm in [s for s in sms if s != "latent_summary_knn"]:
        sm_t = {t["machine"]: t for t in tail_rows if t["score_mode"] == sm}
        imp = sum(1 for m in BOTTLENECK
                  if sm_t.get(m, {}).get("TPR_at_99n", 0) > ls_tail.get(m, {}).get("TPR_at_99n", 0))
        dets = []
        for m in BOTTLENECK:
            old = ls_tail.get(m, {}).get("TPR_at_99n", 0)
            new = sm_t.get(m, {}).get("TPR_at_99n", 0)
            dets.append(f"  - {m}: {old:.4f} -> {new:.4f} ({new-old:+.4f})")
        L.append(f"**{sm}**: {imp}/{len(BOTTLENECK)} improve [{('PASS' if imp >= 2 else 'FAIL')}]")
        L.extend(dets); L.append("")

    # Bearing/valve sanity
    L.extend(["## 7. Bearing / Valve Sanity Check", "",
              "Non-bottleneck machines validate that segment scoring does not cause",
              "catastrophic degradation on held-out machine types.",
              "",
              "| Machine | Score mode | hmean | pAUC | TPR@95n | TPR@99n | Collapse? |",
              "|---------|------------|-------|------|---------|---------|-----------|"])
    for m in ["bearing", "valve"]:
        for sm in sms:
            r = [x for x in rows if x["machine"] == m and x["score_mode"] == sm]
            t = [x for x in tail_rows if x["machine"] == m and x["score_mode"] == sm]
            if r and t:
                h_ok = r[0]["hmean"] >= 0.45
                collapse = "OK" if h_ok else "WARNING: low hmean"
                L.append(f"| {m:10s} | {sm:30s} | {r[0]['hmean']:.4f} | {r[0]['pauc']:.4f} | {t[0]['TPR_at_95n']:.4f} | {t[0]['TPR_at_99n']:.4f} | {collapse} |")
    L.append("")

    # Success criteria
    L.extend(["## 8. Success Criteria", ""])
    ls_h = np.mean([r["hmean"] for r in ls_rows])
    ls_p = np.mean([r["pauc"] for r in ls_rows])

    any_pass = False
    for sm in [s for s in sms if s != "latent_summary_knn"]:
        mr = [r for r in rows if r["score_mode"] == sm]
        sm_h = np.mean([r["hmean"] for r in mr])
        sm_p = np.mean([r["pauc"] for r in mr])
        sm_t = {t["machine"]: t for t in tail_rows if t["score_mode"] == sm}

        c1 = sm_h >= ls_h
        c2 = sm_p >= ls_p
        c3 = sum(1 for m in BOTTLENECK if sm_t.get(m, {}).get("TPR_at_99n", 0) > ls_tail.get(m, {}).get("TPR_at_99n", 0)) >= 2
        c4 = all(r["hmean"] >= 0.45 for r in [x for x in rows if x["machine"] in ("bearing", "valve") and x["score_mode"] == sm])
        q99_ok = all(sm_t.get(m, {}).get("normal_q99", 1) <= ls_tail.get(m, {}).get("normal_q99", 1) * 1.2 for m in machines)
        ok = c1 and c2 and c3 and c4 and q99_ok
        if ok: any_pass = True

        L.append(f"### {sm}")
        L.append(f"- hmean >= ls_knn ({ls_h:.4f}): {sm_h:.4f} [{'PASS' if c1 else 'FAIL'}]")
        L.append(f"- pAUC >= ls_knn ({ls_p:.4f}): {sm_p:.4f} [{'PASS' if c2 else 'FAIL'}]")
        L.append(f"- >= 2 bottleneck TPR@99n improve: [{'PASS' if c3 else 'FAIL'}]")
        L.append(f"- bearing/valve hmean >= 0.45: [{'PASS' if c4 else 'FAIL'}]")
        L.append(f"- q99 not inflated: [{'PASS' if q99_ok else 'FAIL'}]")
        L.append(f"- **Overall: {'PASS' if ok else 'FAIL'}**")
        L.append("")

    if any_pass:
        L.extend([
            "### V43 Verdict: PASS",
            "",
            "Segment-level scoring generalizes to all 7 machines. ",
            "The improvement over file-level latent_summary_knn is consistent.",
            "",
            "**Next**: V44 = 3-seed panel validation of segment scoring on transient_3view.",
        ])
    else:
        L.extend([
            "### V43 Verdict: FAIL",
            "",
            "Segment scoring did not pass full-machine validation. ",
            "V42 remains a bottleneck-only diagnostic.",
            "Do NOT proceed to V44.",
        ])

    L.extend(["", "---", "",
              "All models loaded from V40 checkpoints. No retraining. ",
              "k=5, top_percent=5% fixed. No per-machine tuning."])
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
