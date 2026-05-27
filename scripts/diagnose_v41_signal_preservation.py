#!/usr/bin/env python
"""V41 Transient Signal Preservation Diagnostic.

Tracks V39 Stage-0 input-level separation signal through each encoder stage
of a trained V40 ViewwiseUnsharedLateFusionAutoencoder.

Loads saved V40 models (no retraining). Computes robust_d at 6 representation
stages: input, stem, enc2, fused_pre, latent_map, latent_summary.

Outputs:
  results/v41_transient_signal_preservation/
    per_machine.csv
  docs/CTAF_V41_TRANSIENT_SIGNAL_PRESERVATION_DIAGNOSTIC.md
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
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


STAGE_NAMES = ["input", "stem", "enc2", "fused_pre", "latent_map", "latent_summary"]
STAGE_LABELS = {
    "input": "Input (raw tensor)",
    "stem": "Per-view stem (enc1, V*8, F/2, T/2)",
    "enc2": "Per-view encoder (enc2, V*16, F/4, T/4)",
    "fused_pre": "Fused multi-view (before fusion_enc, V*16, F/4, T/4)",
    "latent_map": "Final latent map (fusion_enc, 64, F/8, T/8)",
    "latent_summary": "Latent summary (mean+std+max, 192)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V41 signal preservation diagnostic")
    p.add_argument("--data_root", required=True)
    p.add_argument("--v40_dir", required=True,
                   help="V40 output dir (contains models/ and scores/)")
    p.add_argument("--out_dir", default="results/v41_transient_signal_preservation")
    p.add_argument("--out_md", default="docs/CTAF_V41_TRANSIENT_SIGNAL_PRESERVATION_DIAGNOSTIC.md")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--machines", default="gearbox,slider,toycar,toytrain,fan")
    p.add_argument("--experiment", default="transient_3view")
    p.add_argument("--score_mode", default="latent_summary_knn")
    p.add_argument("--normal_tail_q_low", type=float, default=0.95)
    p.add_argument("--normal_tail_q_high", type=float, default=0.99)
    return p.parse_args()


# --------------------------------------------------------------------------
# robust_d
# --------------------------------------------------------------------------

def compute_robust_d(dist_a: np.ndarray, dist_b: np.ndarray,
                     eps: float = 1e-8) -> float:
    if len(dist_a) == 0 or len(dist_b) == 0:
        return 0.0
    med_a = float(np.median(dist_a))
    med_b = float(np.median(dist_b))
    iqr_a = float(np.subtract(*np.percentile(dist_a, [75, 25])))
    iqr_b = float(np.subtract(*np.percentile(dist_b, [75, 25])))
    denom = 0.5 * (iqr_a + iqr_b) + eps
    return (med_a - med_b) / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    v40_dir = Path(args.v40_dir)
    out_dir = Path(args.out_dir)
    machines = [m.strip() for m in args.machines.split(",")]
    exp_name = args.experiment

    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = v40_dir / "models"

    all_items = scan_dataset(data_root)

    # Per-stage records
    rows: List[dict] = []

    for machine in machines:
        print(f"\n{'='*60}")
        print(f"Machine: {machine}")
        print(f"{'='*60}")

        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)

        # Load scores
        score_label = f"{exp_name}_{args.score_mode}"
        npz_path = v40_dir / "scores" / f"{machine}_{score_label}_scores.npz"
        if not npz_path.exists():
            print(f"  SKIP: no scores at {npz_path}")
            continue
        arr = np.load(npz_path, allow_pickle=True)
        test_scores = arr["test_scores"].astype(np.float64)
        y_test = arr["y_test"].astype(np.int64)

        # Classify
        normal_scores = test_scores[y_test == 0]
        q_low = float(np.quantile(normal_scores, args.normal_tail_q_low))
        q_high = float(np.quantile(normal_scores, args.normal_tail_q_high))

        normal_idx = np.where(y_test == 0)[0]
        anomaly_idx = np.where(y_test == 1)[0]

        tail_mask = ((test_scores[normal_idx] > q_low) &
                     (test_scores[normal_idx] <= q_high))
        tail_idx = normal_idx[tail_mask]

        missed_mask = test_scores[anomaly_idx] <= q_high
        missed_idx = anomaly_idx[missed_mask]

        detected_mask = test_scores[anomaly_idx] > q_high
        detected_idx = anomaly_idx[detected_mask]

        print(f"  q_low={q_low:.6f}  q_high={q_high:.6f}")
        print(f"  normal_tail={len(tail_idx)}  missed={len(missed_idx)}  "
              f"detected={len(detected_idx)}")

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
            train_feats[item.path] = (base, audio, item.label, item.domain)

        test_feats: Dict[Path, tuple] = {}
        for item in test_items:
            audio = load_audio(item.path)
            base = extract_logmel(audio)
            test_feats[item.path] = (base, audio, item.label, item.domain)

        # Build tensors & normalizer
        train_tensors = []
        for item in train_items:
            base, audio_t, _, _ = train_feats[item.path]
            t = build_tensor(exp_name, base, audio=audio_t)
            train_tensors.append(t)

        V = train_tensors[0].shape[0]

        normalizer = TensorNormalizer()
        normalizer.fit(train_tensors)

        # Build model
        model = ViewwiseUnsharedLateFusionAutoencoder(
            in_channels=V, latent_channels=64,
            use_batch_norm=True, per_view_channels=16,
            stem_channels=8, dropout_p=0.0,
        )
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        model.to(args.device)
        model.eval()

        # Collect test tensors
        # Build test_tensors in same order as test_items
        test_tensors_all = []
        for item in test_items:
            base, audio_t, _, _ = test_feats[item.path]
            t = build_tensor(exp_name, base, audio=audio_t)
            test_tensors_all.append(t)

        # Helper: extract staged representations for a list of indices
        def extract_staged(idx_list: np.ndarray) -> Dict[str, List[np.ndarray]]:
            out: Dict[str, List[np.ndarray]] = {s: [] for s in STAGE_NAMES}
            for idx in idx_list:
                raw = test_tensors_all[idx]
                t = normalizer.transform(raw)
                x = torch.from_numpy(t[None, ...]).to(args.device)
                stages = model.encode_staged(x)
                for sname in STAGE_NAMES:
                    vec = stages[sname].cpu().numpy().ravel()
                    out[sname].append(vec.astype(np.float64))
            return out

        # Extract representations
        tail_repr = extract_staged(tail_idx)
        missed_repr = extract_staged(missed_idx) if len(missed_idx) > 0 else {s: [] for s in STAGE_NAMES}
        detected_repr = extract_staged(detected_idx) if len(detected_idx) > 0 else {s: [] for s in STAGE_NAMES}

        # For each stage, compute centroid + distances
        for sname in STAGE_NAMES:
            tail_vecs = tail_repr[sname]
            if len(tail_vecs) == 0:
                continue
            centroid = np.mean(np.stack(tail_vecs, axis=0), axis=0)

            # Distances from centroid
            tail_dists = np.array([np.linalg.norm(v - centroid) for v in tail_vecs])

            missed_vecs = missed_repr[sname]
            missed_dists = (np.array([np.linalg.norm(v - centroid) for v in missed_vecs])
                            if missed_vecs else np.array([]))

            detected_vecs = detected_repr[sname]
            detected_dists = (np.array([np.linalg.norm(v - centroid) for v in detected_vecs])
                              if detected_vecs else np.array([]))

            rd_missed = compute_robust_d(missed_dists, tail_dists)
            rd_detected = compute_robust_d(detected_dists, tail_dists)

            mean_tail = float(np.mean(tail_dists)) if len(tail_dists) > 0 else 0.0
            mean_missed = float(np.mean(missed_dists)) if len(missed_dists) > 0 else 0.0
            mean_detected = float(np.mean(detected_dists)) if len(detected_dists) > 0 else 0.0

            rows.append({
                "machine": machine,
                "stage": sname,
                "stage_label": STAGE_LABELS[sname],
                "robust_d_missed": round(rd_missed, 6),
                "robust_d_detected": round(rd_detected, 6),
                "mean_dist_tail": round(mean_tail, 6),
                "mean_dist_missed": round(mean_missed, 6),
                "mean_dist_detected": round(mean_detected, 6),
                "n_missed": len(missed_idx),
                "n_detected": len(detected_idx),
                "n_normal_tail": len(tail_idx),
            })

            print(f"  {sname:16s}  rd_missed={rd_missed:.4f}  "
                  f"rd_detected={rd_detected:.4f}  "
                  f"d_tail={mean_tail:.4f}  d_missed={mean_missed:.4f}")

    # Write CSV
    csv_path = out_dir / "per_machine.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {csv_path}")

    # Build report
    report = _build_md_report(args, rows, machines)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report)
    print(f"Report: {out_md}")
    print(report)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _build_md_report(args, rows, machines) -> str:
    lines = [
        "# CTAF V41 Transient Signal Preservation Diagnostic",
        "",
        "## 1. Purpose",
        "",
        "Diagnose where V39 Stage-0 input-level separation signal is lost "
        "through the V40 `ViewwiseUnsharedLateFusionAutoencoder` pipeline.",
        "",
        f"- Experiment: `{args.experiment}`",
        f"- Score mode: `{args.score_mode}`",
        f"- Normal-tail band: q{args.normal_tail_q_low} -- q{args.normal_tail_q_high}",
        "",
        "robust_d = (median(group) - median(normal_tail)) / "
        "(0.5*(IQR(group)+IQR(normal_tail)) + 1e-8)",
        "",
        "## 2. Per-Machine Per-Stage Separation",
        "",
    ]

    for machine in machines:
        m_rows = [r for r in rows if r["machine"] == machine]
        if not m_rows:
            continue

        lines.extend([
            f"### {machine}",
            "",
            f"n_missed={m_rows[0]['n_missed']}  "
            f"n_detected={m_rows[0]['n_detected']}  "
            f"n_normal_tail={m_rows[0]['n_normal_tail']}",
            "",
            "| Stage | robust_d_missed | robust_d_detected | d_tail | d_missed | d_detected | Signal? |",
            "|-------|-----------------|-------------------|--------|----------|------------|---------|",
        ])

        rd_prev = None
        for r in m_rows:
            rd_m = r["robust_d_missed"]
            rd_d = r["robust_d_detected"]

            # Signal tracking: compare robust_d_missed with previous stage
            if rd_prev is not None:
                if rd_m > rd_prev * 0.9:
                    signal = "preserved"
                elif rd_m > rd_prev * 0.5:
                    signal = "decaying"
                else:
                    signal = "LOST"
            else:
                signal = "baseline" if rd_m >= 0.30 else "weak"
            rd_prev = rd_m

            lines.append(
                f"| {r['stage']:12s} | {rd_m:.4f} | {rd_d:.4f} | "
                f"{r['mean_dist_tail']:.4f} | {r['mean_dist_missed']:.4f} | "
                f"{r['mean_dist_detected']:.4f} | {signal} |"
            )
        lines.append("")

    # Cross-machine summary
    lines.extend([
        "## 3. Cross-Machine Stage Summary",
        "",
        "Mean |robust_d_missed| across 5 bottleneck machines, per stage.",
        "",
        "| Stage | Mean |rd_missed| | Min |rd_missed| | Max |rd_missed| | Signal retention |",
        "|-------|-----------------|-----------------|-----------------|-------------------|",
    ])

    for sname in STAGE_NAMES:
        vals = [r["robust_d_missed"] for r in rows if r["stage"] == sname]
        if not vals:
            continue
        mean_v = np.mean(np.abs(vals))
        min_v = np.min(np.abs(vals))
        max_v = np.max(np.abs(vals))
        if mean_v >= 0.25:
            retention = "strong"
        elif mean_v >= 0.10:
            retention = "partial"
        else:
            retention = "lost"
        lines.append(
            f"| {sname:12s} | {mean_v:.4f} | {min_v:.4f} | {max_v:.4f} | {retention} |"
        )
    lines.append("")

    # Analysis: where does signal drop?
    lines.extend([
        "## 4. Signal Drop-off Analysis",
        "",
    ])

    # Compute per-stage mean robust_d across machines
    stage_means = {}
    for sname in STAGE_NAMES:
        vals = [r["robust_d_missed"] for r in rows if r["stage"] == sname]
        stage_means[sname] = np.mean(np.abs(vals)) if vals else 0.0

    input_rd = stage_means.get("input", 0.0)
    stem_rd = stage_means.get("stem", 0.0)
    enc2_rd = stage_means.get("enc2", 0.0)
    latent_rd = stage_means.get("latent_map", 0.0)
    summary_rd = stage_means.get("latent_summary", 0.0)

    # Identify the primary loss stage
    losses = [
        ("input -> stem", input_rd - stem_rd),
        ("stem -> enc2", stem_rd - enc2_rd),
        ("enc2 -> latent_map", enc2_rd - latent_rd),
        ("latent_map -> latent_summary", latent_rd - summary_rd),
    ]
    losses.sort(key=lambda x: x[1], reverse=True)
    primary_loss = losses[0]

    lines.append(f"- **Primary signal loss stage**: `{primary_loss[0]}` "
                 f"(drop = {primary_loss[1]:.4f})")
    lines.append(f"- Input |robust_d|: {input_rd:.4f}")
    lines.append(f"- Stem |robust_d|: {stem_rd:.4f}")
    lines.append(f"- Enc2 |robust_d|: {enc2_rd:.4f}")
    lines.append(f"- Latent map |robust_d|: {latent_rd:.4f}")
    lines.append(f"- Latent summary |robust_d|: {summary_rd:.4f}")
    lines.append("")

    # Per-machine drop pattern
    lines.append("### Per-machine drop pattern")
    lines.append("")
    for machine in machines:
        m_rows = [r for r in rows if r["machine"] == machine]
        if not m_rows:
            continue
        m_rd = {r["stage"]: abs(r["robust_d_missed"]) for r in m_rows}
        input_v = m_rd.get("input", 0)
        summary_v = m_rd.get("latent_summary", 0)
        retention = summary_v / max(input_v, 1e-8)
        lines.append(f"- **{machine}**: input={input_v:.4f} -> summary={summary_v:.4f} "
                     f"(retention={retention:.1%})")
    lines.append("")

    # Toytrain analysis
    toytrain_rows = [r for r in rows if r["machine"] == "toytrain"]
    if toytrain_rows:
        lines.extend([
            "## 5. Toytrain: Why Signal Survives",
            "",
        ])
        tt_rd = {r["stage"]: abs(r["robust_d_missed"]) for r in toytrain_rows}
        lines.append(f"- Input signal: {tt_rd.get('input', 0):.4f}")
        lines.append(f"- Stem output: {tt_rd.get('stem', 0):.4f}")
        lines.append(f"- Latent summary: {tt_rd.get('latent_summary', 0):.4f}")
        if tt_rd.get("stem", 0) > 0.5 and tt_rd.get("latent_summary", 0) > 0.3:
            lines.append("- Signal survives encoder training on this machine.")
            lines.append("- Toytrain's transient view may encode a physically meaningful "
                         "signature (e.g., speed variation in train audio) that "
                         "the encoder's per-view stem captures effectively.")
        lines.append("")

    # Failure machines analysis
    lines.extend([
        "## 6. Failure Machines: Where Signal Dies",
        "",
    ])
    failure_machines = [m for m in machines if m != "toytrain"]
    for machine in failure_machines:
        m_rows = [r for r in rows if r["machine"] == machine]
        if not m_rows:
            continue
        m_rd = {r["stage"]: abs(r["robust_d_missed"]) for r in m_rows}
        input_v = m_rd.get("input", 0)
        stem_v = m_rd.get("stem", 0)
        summary_v = m_rd.get("latent_summary", 0)

        # Determine where the main drop happens
        if stem_v < input_v * 0.3:
            drop = "stem"
        elif m_rd.get("enc2", 0) < stem_v * 0.3:
            drop = "enc2"
        elif m_rd.get("latent_map", 0) < m_rd.get("enc2", 0) * 0.3:
            drop = "latent_map"
        elif summary_v < m_rd.get("latent_map", 0) * 0.3:
            drop = "latent_summary aggregation"
        else:
            drop = "scoring"

        lines.append(f"- **{machine}**: input={input_v:.4f} -> stem={stem_v:.4f} "
                     f"-> summary={summary_v:.4f}. "
                     f"Primary loss at: `{drop}`.")
    lines.append("")

    # Next steps
    lines.extend([
        "## 7. Next-Step Guidance",
        "",
    ])

    if primary_loss[0].startswith("input -> stem"):
        lines.extend([
            "- **Recommendation**: Transient-specific per-view stem.",
            "- The current per-view stem (enc1: 1->8, stride=2) is shared across all "
            "three views but is too aggressive for transient features.",
            "- Consider a transient-specific stem with smaller stride or dilated temporal "
            "convolutions to preserve transient events.",
        ])
    elif primary_loss[0].startswith("stem -> enc2"):
        lines.extend([
            "- **Recommendation**: Transient-specific per-view encoder.",
            "- Transient signal survives stem but is lost during enc2 compression.",
            "- Consider encoder architecture that preserves onset/temporal structure, "
            "such as fewer channels or wider temporal receptive field.",
        ])
    elif primary_loss[0].startswith("enc2 -> latent"):
        lines.extend([
            "- **Recommendation**: Modify fusion encoding or latent structure.",
            "- Multi-view fusion (48->64, stride=2) is the bottleneck.",
            "- Consider: skip connection from per-transient-view to latent, or "
            "increase latent_channels.",
        ])
    elif primary_loss[0].startswith("latent_map -> latent_summary"):
        lines.extend([
            "- **Recommendation**: Segment-level or patch-level aggregation.",
            "- Signal exists in the latent map but is lost during global pooling (mean+std+max).",
            "- Consider: local kNN on latent tokens instead of global summary, or "
            "attention-weighted pooling that preserves spatial structure.",
        ])
    else:
        lines.extend([
            "- **Recommendation**: Scoring pipeline, not frontend.",
            "- Signal survives all encoder stages but fails at scoring.",
            "- Consider: density scoring, per-token scoring, or feature-space distance "
            "metrics beyond simple kNN on global summary vectors.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "Models were loaded from saved V40 checkpoints trained under the fixed V40 "
        "protocol. No retraining was performed. No frontend parameters were tuned. "
        "No per-machine selection was applied.",
    ])

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
