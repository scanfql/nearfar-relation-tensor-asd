#!/usr/bin/env python
"""V42 Transient Segment-Level Scoring Audit.

Tests whether local segment tokens (from V40 transient_3view encoder)
produce better low-FPR tail separation than file-level latent_summary_knn.

Uses saved V40 checkpoints. No retraining. No parameter tuning.

Score modes:
  - latent_summary_knn  (V40 baseline)
  - enc2_segment_knn_top5
  - latent_map_segment_knn_top5
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib
matplotlib.use("Agg")
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

BOTTLENECK_MACHINES = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V42 segment-level scoring audit")
    p.add_argument("--data_root", required=True)
    p.add_argument("--v40_dir", required=True)
    p.add_argument("--out_dir", default="results/v42_transient_segment_scoring_audit")
    p.add_argument("--out_md", default="docs/CTAF_V42_TRANSIENT_SEGMENT_SCORING_AUDIT.md")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--machines", default="gearbox,slider,toycar,toytrain,fan")
    p.add_argument("--experiment", default="transient_3view")
    return p.parse_args()


# --------------------------------------------------------------------------
# Tail metrics helper
# --------------------------------------------------------------------------

def compute_tail_metrics(
    test_scores: np.ndarray, y_test: np.ndarray,
) -> dict:
    normal = test_scores[y_test == 0]
    anomaly = test_scores[y_test == 1]
    result = {"n_normal": int(len(normal)), "n_anomaly": int(len(anomaly))}
    if len(normal) == 0:
        return result
    q95 = float(np.quantile(normal, 0.95))
    q99 = float(np.quantile(normal, 0.99))
    result["normal_q95"] = q95
    result["normal_q99"] = q99
    if len(anomaly) > 0:
        result["anomaly_median"] = float(np.median(anomaly))
        result["anomaly_q75"] = float(np.quantile(anomaly, 0.75))
        result["TPR_at_95n"] = float(np.mean(anomaly > q95))
        result["TPR_at_99n"] = float(np.mean(anomaly > q99))
    else:
        for k in ("anomaly_median", "anomaly_q75", "TPR_at_95n", "TPR_at_99n"):
            result[k] = 0.0
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    v40_dir = Path(args.v40_dir)
    out_dir = Path(args.out_dir)
    exp_name = args.experiment

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    model_dir = v40_dir / "models"
    machines = [m.strip() for m in args.machines.split(",")]

    all_items = scan_dataset(data_root)

    rows: List[dict] = []
    tail_rows: List[dict] = []

    for machine in machines:
        print(f"\n{'='*60}")
        print(f"Machine: {machine}")
        print(f"{'='*60}")

        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)

        # Load model
        model_path = model_dir / f"{machine}_{exp_name}.pt"
        if not model_path.exists():
            print(f"  SKIP: no model at {model_path}")
            continue

        # Extract features
        train_feats: Dict[Path, tuple] = {}
        for item in train_items:
            audio = load_audio(item.path)
            base = extract_logmel(audio)
            train_feats[item.path] = (base, audio)

        test_feats: Dict[Path, tuple] = {}
        for item in test_items:
            audio = load_audio(item.path)
            base = extract_logmel(audio)
            test_feats[item.path] = (base, audio)

        # Build tensors & normalizer
        train_tensors_all = []
        for item in train_items:
            base, audio_t = train_feats[item.path]
            t = build_tensor(exp_name, base, audio=audio_t)
            train_tensors_all.append(t)

        V = train_tensors_all[0].shape[0]

        normalizer = TensorNormalizer()
        train_norm_all = normalizer.fit_transform(train_tensors_all)
        train_arr = np.stack(train_norm_all, axis=0)

        test_tensors_all = []
        test_labels = []
        for item in test_items:
            base, audio_t = test_feats[item.path]
            t = build_tensor(exp_name, base, audio=audio_t)
            test_tensors_all.append(t)
            test_labels.append(item.label)
        test_norm_all = [normalizer.transform(t) for t in test_tensors_all]
        test_arr = np.stack(test_norm_all, axis=0)
        y_test = np.array(test_labels, dtype=np.int64)

        # Build model
        model = ViewwiseUnsharedLateFusionAutoencoder(
            in_channels=V, latent_channels=64,
            use_batch_norm=True, per_view_channels=16,
            stem_channels=8, dropout_p=0.0,
        )
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.to(args.device)
        model.eval()

        # ---- Score 1: latent_summary_knn (V40 baseline) ----
        print("  Computing latent_summary_knn...")
        memory_ls = compute_latent_summary_vectors(
            model, train_arr, batch_size=32, device=args.device,
        )
        test_ls = compute_latent_summary_vectors(
            model, test_arr, batch_size=32, device=args.device,
        )
        ls_stats = fit_feature_robust_stats(memory_ls)
        test_scores_ls = compute_latent_knn_scores(
            memory_ls, test_ls, k=5, train_feature_stats=ls_stats,
        )
        metrics_ls = compute_metrics(y_test, test_scores_ls, None)

        # ---- Extract segment tokens for enc2 and latent_map ----
        print("  Extracting segment tokens...")

        # Train-normal tokens
        memory_enc2: List[np.ndarray] = []
        memory_latent: List[np.ndarray] = []

        batch_size = 16
        for start in range(0, len(train_arr), batch_size):
            end = min(start + batch_size, len(train_arr))
            x = torch.from_numpy(train_arr[start:end]).to(args.device)
            stages = model.encode_staged(x)

            for i in range(x.shape[0]):
                # enc2: [48, F/4, T/4] -> [F/4*T/4, 48] tokens
                e2 = stages["enc2"][i].cpu().numpy()  # [48, H2, W2]
                e2_tokens = e2.reshape(e2.shape[0], -1).T  # [H2*W2, 48]
                memory_enc2.append(e2_tokens)

                # latent_map: [64, F/8, T/8] -> [F/8*T/8, 64] tokens
                lm = stages["latent_map"][i].cpu().numpy()  # [64, H3, W3]
                lm_tokens = lm.reshape(lm.shape[0], -1).T  # [H3*W3, 64]
                memory_latent.append(lm_tokens)

        memory_enc2_arr = np.concatenate(memory_enc2, axis=0)
        memory_latent_arr = np.concatenate(memory_latent, axis=0)
        print(f"    enc2 memory tokens: {memory_enc2_arr.shape}")
        print(f"    latent_map memory tokens: {memory_latent_arr.shape}")

        # Test tokens per file
        test_enc2_tokens: List[np.ndarray] = []
        test_latent_tokens: List[np.ndarray] = []

        for start in range(0, len(test_arr), batch_size):
            end = min(start + batch_size, len(test_arr))
            x = torch.from_numpy(test_arr[start:end]).to(args.device)
            stages = model.encode_staged(x)

            for i in range(x.shape[0]):
                e2 = stages["enc2"][i].cpu().numpy()
                e2_tokens = e2.reshape(e2.shape[0], -1).T
                test_enc2_tokens.append(e2_tokens)

                lm = stages["latent_map"][i].cpu().numpy()
                lm_tokens = lm.reshape(lm.shape[0], -1).T
                test_latent_tokens.append(lm_tokens)

        # ---- Score 2: enc2_segment_knn_top5 ----
        print("  Computing enc2_segment_knn_top5...")
        test_scores_enc2 = compute_local_tail_knn_scores(
            memory_enc2_arr, test_enc2_tokens,
            k=5, top_percent=5.0, device=args.device,
        )
        metrics_enc2 = compute_metrics(y_test, test_scores_enc2, None)

        # ---- Score 3: latent_map_segment_knn_top5 ----
        print("  Computing latent_map_segment_knn_top5...")
        test_scores_latent_seg = compute_local_tail_knn_scores(
            memory_latent_arr, test_latent_tokens,
            k=5, top_percent=5.0, device=args.device,
        )
        metrics_latent_seg = compute_metrics(y_test, test_scores_latent_seg, None)

        # Save scores
        for name, scores in [
            ("transient_3view_latent_summary_knn", test_scores_ls),
            ("transient_3view_enc2_segment_knn_top5", test_scores_enc2),
            ("transient_3view_latent_map_segment_knn_top5", test_scores_latent_seg),
        ]:
            npz_path = scores_dir / f"{machine}_{name}_scores.npz"
            np.savez_compressed(npz_path, test_scores=scores, y_test=y_test)

        # Collect results
        for mode_name, metrics, scores in [
            ("latent_summary_knn", metrics_ls, test_scores_ls),
            ("enc2_segment_knn_top5", metrics_enc2, test_scores_enc2),
            ("latent_map_segment_knn_top5", metrics_latent_seg, test_scores_latent_seg),
        ]:
            rows.append({
                "machine": machine,
                "score_mode": mode_name,
                "source_auc": round(metrics["auc_source"], 6),
                "target_auc": round(metrics["auc_target"], 6),
                "pauc": round(metrics["pauc"], 6),
                "dcase_auc": round(metrics["dcase_auc"], 6),
                "hmean": round(metrics["hmean"], 6),
            })

            tail = compute_tail_metrics(scores, y_test)
            tail_rows.append({
                "machine": machine,
                "score_mode": mode_name,
                **tail,
            })

        print(f"  ls_knn: hmean={metrics_ls['hmean']:.4f} pAUC={metrics_ls['pauc']:.4f}")
        print(f"  enc2_seg: hmean={metrics_enc2['hmean']:.4f} pAUC={metrics_enc2['pauc']:.4f}")
        print(f"  latent_seg: hmean={metrics_latent_seg['hmean']:.4f} pAUC={metrics_latent_seg['pauc']:.4f}")

    # ---- Write CSVs ----
    if rows:
        with (out_dir / "per_machine.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if tail_rows:
        with (out_dir / "tail_metrics.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(tail_rows[0].keys()))
            w.writeheader()
            w.writerows(tail_rows)

    # ---- Build report ----
    report = _build_md(rows, tail_rows, machines)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report)
    print(f"\nReport: {out_md}")
    print(report)


# --------------------------------------------------------------------------
# Report builder
# --------------------------------------------------------------------------

def _build_md(rows: list, tail_rows: list, machines: list) -> str:
    lines = [
        "# CTAF V42 Transient Segment-Level Scoring Audit",
        "",
        "## 1. V40 Failure Summary",
        "",
        "V40 single-seed transient_3view FS train:",
        "",
        "- Criteria 1-3, 5 PASS (hmean > logmel, > full_5view, dcase_auc > V16 baseline, KS OK)",
        "- Criteria 4 FAIL: pAUC = 0.5213 < V16 baseline 0.5423",
        "- Criteria 6 FAIL: only 1/5 bottleneck machines improve TPR@99n (toytrain)",
        "- Verdict: SINGLE-SEED FAIL.",
        "",
        "## 2. V41 Signal Preservation Summary",
        "",
        "V41 tracked robust_d_missed through 6 encoder stages:",
        "",
        "- Signal does NOT decay through encoder. Mean |rd| goes from 0.999 (input) to 1.771 (latent_summary).",
        "- toycar: encoder destroys signal (rd: 1.67 -> -0.22)",
        "- gearbox/slider/fan: signal amplifies but scoring fails to capture it at low FPR",
        "- toytrain: clear signal -> successful separation pattern",
        "- Conclusion: problem is in scoring/aggregation, not in encoder/frontend.",
        "",
        "## 3. Score Comparison",
        "",
    ]

    # Overall means
    score_modes = sorted(set(r["score_mode"] for r in rows))
    lines.append("### Overall (mean across 5 bottleneck machines)")
    lines.append("")
    lines.append("| Score mode | Mean hmean | Mean pAUC | Mean dcase_auc |")
    lines.append("|------------|------------|-----------|----------------|")

    for sm in score_modes:
        mr = [r for r in rows if r["score_mode"] == sm]
        if not mr:
            continue
        mh = np.mean([r["hmean"] for r in mr])
        mp = np.mean([r["pauc"] for r in mr])
        md = np.mean([r["dcase_auc"] for r in mr])
        lines.append(f"| {sm:30s} | {mh:.4f} | {mp:.4f} | {md:.4f} |")
    lines.append("")

    # Per-machine
    lines.append("### Per-Machine hmean / pAUC")
    lines.append("")
    lines.append("| Machine | Score mode | hmean | pAUC |")
    lines.append("|---------|------------|-------|------|")

    for machine in machines:
        for sm in score_modes:
            mr = [r for r in rows if r["machine"] == machine and r["score_mode"] == sm]
            if mr:
                r = mr[0]
                lines.append(f"| {machine:10s} | {sm:30s} | {r['hmean']:.4f} | {r['pauc']:.4f} |")
    lines.append("")

    # Bottleneck tail table
    lines.extend([
        "## 4. Bottleneck-Machine Tail Table",
        "",
        "| Machine | Score mode | q95 | q99 | Anom.med | Anom.q75 | TPR@95n | TPR@99n |",
        "|---------|------------|-----|-----|----------|----------|---------|---------|",
    ])

    for machine in machines:
        for sm in score_modes:
            tr = [t for t in tail_rows if t["machine"] == machine and t["score_mode"] == sm]
            if not tr:
                continue
            t = tr[0]
            lines.append(
                f"| {machine:10s} | {sm:30s} | {t['normal_q95']:.4f} | {t['normal_q99']:.4f} | "
                f"{t['anomaly_median']:.4f} | {t['anomaly_q75']:.4f} | "
                f"{t['TPR_at_95n']:.4f} | {t['TPR_at_99n']:.4f} |"
            )
    lines.append("")

    # TPR@99n improvement analysis
    lines.extend([
        "## 5. TPR@99n Improvement Analysis",
        "",
    ])

    ls_tail = {t["machine"]: t for t in tail_rows if t["score_mode"] == "latent_summary_knn"}
    for sm in [s for s in score_modes if s != "latent_summary_knn"]:
        sm_tail = {t["machine"]: t for t in tail_rows if t["score_mode"] == sm}
        improved = 0
        details = []
        for m in machines:
            ls_tpr = ls_tail.get(m, {}).get("TPR_at_99n", 0.0)
            sm_tpr = sm_tail.get(m, {}).get("TPR_at_99n", 0.0)
            if sm_tpr > ls_tpr:
                improved += 1
                details.append(f"  - {m}: {ls_tpr:.4f} -> {sm_tpr:.4f} (+{sm_tpr-ls_tpr:.4f})")
            else:
                details.append(f"  - {m}: {ls_tpr:.4f} -> {sm_tpr:.4f} ({sm_tpr-ls_tpr:+.4f})")
        status = "PASS" if improved >= 2 else "FAIL"
        lines.append(f"**{sm}**: {improved}/{len(machines)} machines improve TPR@99n [{status}]")
        for d in details:
            lines.append(d)
        lines.append("")

    # Normal q99 inflation check
    lines.append("## 6. Normal q99 Inflation Check")
    lines.append("")
    lines.append("| Score mode | gearbox | slider | toycar | toytrain | fan |")
    lines.append("|------------|---------|--------|--------|----------|-----|")

    for sm in score_modes:
        vals = []
        for m in machines:
            tr = [t for t in tail_rows if t["machine"] == m and t["score_mode"] == sm]
            vals.append(f"{tr[0]['normal_q99']:.4f}" if tr else "--")
        lines.append(f"| {sm:30s} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | {vals[4]} |")
    lines.append("")

    # Final verdict
    lines.extend([
        "## 7. Final Verdict",
        "",
    ])

    # Check success criteria per segment mode
    ls_mr = [r for r in rows if r["score_mode"] == "latent_summary_knn"]
    ls_hmean = np.mean([r["hmean"] for r in ls_mr])
    ls_pauc = np.mean([r["pauc"] for r in ls_mr])

    any_pass = False
    for sm in [s for s in score_modes if s != "latent_summary_knn"]:
        sm_rows = [r for r in rows if r["score_mode"] == sm]
        sm_hmean = np.mean([r["hmean"] for r in sm_rows])
        sm_pauc = np.mean([r["pauc"] for r in sm_rows])

        sm_tail = {t["machine"]: t for t in tail_rows if t["score_mode"] == sm}
        tpr_improved = sum(
            1 for m in machines
            if sm_tail.get(m, {}).get("TPR_at_99n", 0) >
               ls_tail.get(m, {}).get("TPR_at_99n", 0)
        )

        q99_ok = all(
            sm_tail.get(m, {}).get("normal_q99", 1.0) <=
            ls_tail.get(m, {}).get("normal_q99", 0.0) * 1.2
            for m in machines
        )

        c1 = sm_hmean >= ls_hmean
        c2 = sm_pauc >= ls_pauc
        c3 = tpr_improved >= 2
        c4 = q99_ok

        all_ok = c1 and c2 and c3 and c4
        if all_ok:
            any_pass = True

        lines.append(f"### {sm}")
        lines.append(f"- hmean >= V40 baseline ({ls_hmean:.4f}): {sm_hmean:.4f} [{'PASS' if c1 else 'FAIL'}]")
        lines.append(f"- pAUC >= V40 baseline ({ls_pauc:.4f}): {sm_pauc:.4f} [{'PASS' if c2 else 'FAIL'}]")
        lines.append(f"- TPR@99n improved on >= 2 machines: {tpr_improved}/5 [{'PASS' if c3 else 'FAIL'}]")
        lines.append(f"- Normal q99 not inflated: [{'PASS' if c4 else 'FAIL'}]")
        lines.append(f"- **Overall: {'PASS' if all_ok else 'FAIL'}**")
        lines.append("")

    if any_pass:
        lines.extend([
            "**Segment-level scoring provides improvement over file-level latent_summary_knn.**",
            "This confirms V41's hypothesis that the encoder preserves signal but global pooling loses it.",
        ])
    else:
        lines.extend([
            "**No segment-level score met all success criteria.**",
            "",
            "V42 is recorded as a segment-scoring diagnostic failure.",
            "Do NOT tune top fraction, k, or do score fusion.",
            "The encoder may be producing representations that are not kNN-separable "
            "even at the local token level. Reconsider whether transient frontend + "
            "reconstruction AE is the right path, or whether an alternative training "
            "objective (contrastive, normalizing flow) is needed.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "Models were loaded from saved V40 checkpoints. No retraining was performed. "
        "No scoring parameters (k=5, top_percent=5%) were tuned. "
        "No per-machine selection was applied.",
    ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
