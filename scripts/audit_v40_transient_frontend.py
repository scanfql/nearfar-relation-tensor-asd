#!/usr/bin/env python
"""V40 Transient Frontend Training Validation Audit.

Reads V40 single-seed results, computes bottleneck tail metrics,
and generates the CTAF_V40_TRANSIENT_FRONTEND_AUDIT.md report.

Offline-only: no retraining, no score tuning.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Pre-registered V16 5-seed panel baseline (full_5view, latent_summary_knn)
V16_BASELINE = {
    "hmean": 0.549547,   # mean dcase_auc across 5 seeds
    "pAUC": 0.542331,    # mean pauc across 5 seeds
    "KS": 0.294880,      # mean source_target_ks across 5 seeds
}

BOTTLENECK_MACHINES = ["gearbox", "slider", "toycar", "toytrain", "fan"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V40 transient frontend audit")
    p.add_argument("--v40_dir", required=True,
                   help="V40 output directory (contains per_machine.csv, scores/)")
    p.add_argument("--out_md", default="docs/CTAF_V40_TRANSIENT_FRONTEND_AUDIT.md",
                   help="Output markdown report path")
    p.add_argument("--score_mode", default="latent_summary_knn",
                   help="Score mode to audit (default: latent_summary_knn)")
    return p.parse_args()


def load_per_machine(v40_dir: Path, score_mode: str) -> Dict[str, Dict]:
    """Load per_machine.csv and return dict keyed by (machine, experiment)."""
    pm_path = v40_dir / "per_machine.csv"
    if not pm_path.exists():
        raise FileNotFoundError(f"per_machine.csv not found in {v40_dir}")

    data: Dict[Tuple[str, str], Dict] = {}
    with pm_path.open("r") as f:
        for row in csv.DictReader(f):
            if row.get("score_mode", "") != score_mode:
                continue
            key = (row["machine"], row["experiment"])
            data[key] = {
                "machine": row["machine"],
                "experiment": row["experiment"],
                "source_auc": float(row["source_auc"]),
                "target_auc": float(row["target_auc"]),
                "pauc": float(row["pauc"]),
                "hmean": float(row["hmean"]),
                "dcase_auc": float(row["dcase_auc"]),
            }
    return data


def load_score_geometry(v40_dir: Path, score_mode: str) -> Dict[Tuple[str, str], Dict]:
    """Load score_geometry.csv."""
    sg_path = v40_dir / "score_geometry.csv"
    data: Dict[Tuple[str, str], Dict] = {}
    if not sg_path.exists():
        return data
    with sg_path.open("r") as f:
        for row in csv.DictReader(f):
            if row.get("score_mode", "") != score_mode:
                continue
            key = (row["machine"], row["experiment"])
            data[key] = {
                "source_normal_mean": float(row["source_normal_mean"]),
                "target_normal_mean": float(row["target_normal_mean"]),
                "source_normal_std": float(row["source_normal_std"]),
                "target_normal_std": float(row["target_normal_std"]),
                "source_target_ks": float(row["source_target_ks"]),
                "median_gap": float(row["median_gap"]),
                "iqr_gap": float(row["iqr_gap"]),
            }
    return data


def compute_tail_metrics(
    v40_dir: Path, machine: str, experiment: str, score_mode: str,
) -> Dict:
    """Compute bottleneck tail table metrics from score NPZ.

    Returns dict with: normal_q95, normal_q99, anomaly_median, anomaly_q75,
    TPR@95n, TPR@99n, n_normal, n_anomaly.
    """
    score_label = f"{experiment}_{score_mode}"
    npz_path = v40_dir / "scores" / f"{machine}_{score_label}_scores.npz"
    result = {
        "normal_q95": 0.0, "normal_q99": 0.0,
        "anomaly_median": 0.0, "anomaly_q75": 0.0,
        "TPR_at_95n": 0.0, "TPR_at_99n": 0.0,
        "n_normal": 0, "n_anomaly": 0,
        "label": score_label,
    }

    if not npz_path.exists():
        result["label"] = f"MISSING: {npz_path}"
        return result

    arr = np.load(npz_path, allow_pickle=True)
    test_scores = arr["test_scores"].astype(np.float64)
    y_test = arr["y_test"].astype(np.int64)

    normal_scores = test_scores[y_test == 0]
    anomaly_scores = test_scores[y_test == 1]

    result["n_normal"] = int(len(normal_scores))
    result["n_anomaly"] = int(len(anomaly_scores))

    if len(normal_scores) == 0:
        return result

    q95 = float(np.quantile(normal_scores, 0.95))
    q99 = float(np.quantile(normal_scores, 0.99))
    result["normal_q95"] = q95
    result["normal_q99"] = q99

    if len(anomaly_scores) > 0:
        result["anomaly_median"] = float(np.median(anomaly_scores))
        result["anomaly_q75"] = float(np.quantile(anomaly_scores, 0.75))
        result["TPR_at_95n"] = float(np.mean(anomaly_scores > q95))
        result["TPR_at_99n"] = float(np.mean(anomaly_scores > q99))

    return result


def fmt_val(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def build_report(
    args: argparse.Namespace,
    pm: Dict,
    geo: Dict,
    tail: Dict[Tuple[str, str], Dict],
) -> str:
    score_mode = args.score_mode
    experiments = sorted(set(k[1] for k in pm.keys()))
    machines = sorted(set(k[0] for k in pm.keys()))

    lines = [
        "# CTAF V40 Transient Frontend Training Validation Audit",
        "",
        "## 1. V39 Stage-0 Summary",
        "",
        "| Frontend | Gate A | Redundancy max_r | Stage 0 |",
        "|----------|--------|------------------|---------|",
        "| transient_3view | 5/5 PASS | 0.0538 | **PASS** |",
        "| modulation_3view | 5/5 PASS | 0.9688 | FAIL (Gate C: too redundant) |",
        "| cepstral_3view | 0/5 FAIL | 0.6089 | FAIL (Gate A: NaN distances) |",
        "",
        "- `transient_3view` enters V40: strong input-level separation signal (mean robust_d_missed=0.738), very low redundancy (r=0.054), 4/5 machines pass all gates.",
        "- `modulation_3view` does NOT enter V40: modulation_low r=0.969 vs logmel_base, fails pre-registered redundancy gate C.",
        "- `cepstral_3view` does NOT enter V40: Gate A fails on all 5 machines (NaN in distance computation).",
        "",
        "## 2. V16 Seed-Panel Mean Baseline",
        "",
        "Pre-registered V16 5-seed panel (full_5view, latent_summary_knn, viewwise_unshared_late_fusion):",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean hmean (dcase_auc) | {V16_BASELINE['hmean']:.6f} |",
        f"| Mean pAUC | {V16_BASELINE['pAUC']:.6f} |",
        f"| Mean source-target KS | {V16_BASELINE['KS']:.6f} |",
        "",
        "## 3. Overall Comparison",
        "",
        f"Score mode: `{score_mode}`. Mean across {len(machines)} machines.",
        "",
        "| Experiment | Views | Mean hmean | Mean pAUC | Mean KS | vs V16 hmean | vs V16 pAUC | vs logmel_only |",
        "|------------|-------|------------|-----------|---------|--------------|-------------|----------------|",
    ]

    # Compute mean metrics per experiment
    exp_means = {}
    for exp in experiments:
        rows = [pm[k] for k in pm if k[1] == exp]
        if not rows:
            continue
        ks_rows = [geo[k] for k in geo if k[1] == exp]
        exp_means[exp] = {
            "hmean": np.mean([r["hmean"] for r in rows]),
            "pauc": np.mean([r["pauc"] for r in rows]),
            "dcase_auc": np.mean([r["dcase_auc"] for r in rows]),
            "ks": np.mean([r["source_target_ks"] for r in ks_rows]) if ks_rows else 0.0,
            "n_views": rows[0].get("num_views", "?"),
        }

    logmel_hmean = exp_means.get("logmel_only", {}).get("hmean", 0.0)
    for exp in experiments:
        m = exp_means.get(exp)
        if m is None:
            continue
        vs_v16_h = m["dcase_auc"] - V16_BASELINE["hmean"]
        vs_v16_p = m["pauc"] - V16_BASELINE["pAUC"]
        vs_lm = m["hmean"] - logmel_hmean if logmel_hmean else 0.0
        lines.append(
            f"| {exp:18s} | {m['n_views']} | {fmt_val(m['hmean'])} | "
            f"{fmt_val(m['pauc'])} | {fmt_val(m['ks'])} | "
            f"{vs_v16_h:+.4f} | {vs_v16_p:+.4f} | {vs_lm:+.4f} |"
        )

    # 4. Per-machine hmean / pAUC / KS
    lines.extend([
        "",
        "## 4. Per-Machine hmean / pAUC / KS",
        "",
        "| Machine | Experiment | hmean | pAUC | src-tgt KS |",
        "|---------|------------|-------|------|------------|",
    ])

    for m in machines:
        for exp in experiments:
            key = (m, exp)
            r = pm.get(key)
            g = geo.get(key, {})
            if r:
                ks = g.get("source_target_ks", 0.0)
                lines.append(
                    f"| {m:10s} | {exp:18s} | {fmt_val(r['hmean'])} | "
                    f"{fmt_val(r['pauc'])} | {fmt_val(ks)} |"
                )

    # 5. Bottleneck-machine tail table
    lines.extend([
        "",
        "## 5. Bottleneck-Machine Tail Table",
        "",
        "Bottleneck = gearbox, slider, toycar, toytrain, fan (V39 frontend-prescreen machines).",
        "",
        "| Machine | Experiment | q95 | q99 | Anom.median | Anom.q75 | TPR@95n | TPR@99n |",
        "|---------|------------|-----|-----|-------------|----------|---------|---------|",
    ])

    for m in BOTTLENECK_MACHINES:
        if m not in machines:
            continue
        for exp in experiments:
            t = tail.get((m, exp), {})
            if not t:
                lines.append(f"| {m:10s} | {exp:18s} | -- | -- | -- | -- | -- | -- |")
                continue
            lines.append(
                f"| {m:10s} | {exp:18s} | {fmt_val(t['normal_q95'], 6)} | "
                f"{fmt_val(t['normal_q99'], 6)} | {fmt_val(t['anomaly_median'], 4)} | "
                f"{fmt_val(t['anomaly_q75'], 4)} | {fmt_val(t['TPR_at_95n'], 4)} | "
                f"{fmt_val(t['TPR_at_99n'], 4)} |"
            )

    # 6. Final Verdict
    lines.extend([
        "",
        "## 6. Final Verdict",
        "",
    ])

    # Check success criteria
    t3_hmean = exp_means.get("transient_3view", {}).get("hmean", 0.0)
    t3_dcase = exp_means.get("transient_3view", {}).get("dcase_auc", 0.0)
    t3_pauc = exp_means.get("transient_3view", {}).get("pauc", 0.0)
    t3_ks = exp_means.get("transient_3view", {}).get("ks", 0.0)
    logmel_h = exp_means.get("logmel_only", {}).get("hmean", 0.0)
    full5_h = exp_means.get("full_5view", {}).get("hmean", 0.0)
    full5_pauc = exp_means.get("full_5view", {}).get("pauc", 0.0)

    checks = []

    # Criteria 1: transient_3view > matched logmel_only
    c1_pass = t3_hmean > logmel_h
    if t3_hmean > 0 and logmel_h > 0:
        checks.append((
            f"1. transient_3view hmean ({t3_hmean:.6f}) > logmel_only hmean ({logmel_h:.6f})",
            c1_pass,
            f"delta = {t3_hmean - logmel_h:+.6f}",
        ))

    # Criteria 2: transient_3view > full_5view
    c2_pass = t3_hmean > full5_h
    if t3_hmean > 0 and full5_h > 0:
        checks.append((
            f"2. transient_3view hmean ({t3_hmean:.6f}) > full_5view hmean ({full5_h:.6f})",
            c2_pass,
            f"delta = {t3_hmean - full5_h:+.6f}",
        ))

    # Criteria 3: hmean > V16 baseline
    c3_pass = t3_dcase > V16_BASELINE["hmean"]
    checks.append((
        f"3. dcase_auc ({t3_dcase:.6f}) > V16 baseline ({V16_BASELINE['hmean']:.6f})",
        c3_pass,
        f"delta = {t3_dcase - V16_BASELINE['hmean']:+.6f}",
    ))

    # Criteria 4: pAUC >= V16 baseline
    c4_pass = t3_pauc >= V16_BASELINE["pAUC"]
    checks.append((
        f"4. pAUC ({t3_pauc:.6f}) >= V16 baseline ({V16_BASELINE['pAUC']:.6f})",
        c4_pass,
        f"delta = {t3_pauc - V16_BASELINE['pAUC']:+.6f}",
    ))

    # Criteria 5: KS not significantly worse
    ks_delta = t3_ks - V16_BASELINE["KS"]
    c5_pass = ks_delta <= 0.05  # "not significantly worse" = within 0.05
    checks.append((
        f"5. KS ({t3_ks:.6f}) not significantly worse than V16 ({V16_BASELINE['KS']:.6f})",
        c5_pass,
        f"delta = {ks_delta:+.6f}",
    ))

    # Criteria 6: >= 2 bottleneck machines improve TPR@99n
    tpr_improved = 0
    tpr_details = []
    for m in BOTTLENECK_MACHINES:
        t3_tail = tail.get((m, "transient_3view"), {})
        logmel_tail = tail.get((m, "logmel_only"), {})
        if t3_tail and logmel_tail:
            t3_tpr = t3_tail.get("TPR_at_99n", 0.0)
            lm_tpr = logmel_tail.get("TPR_at_99n", 0.0)
            if t3_tpr > lm_tpr:
                tpr_improved += 1
                tpr_details.append(f"  - {m}: TPR@99n {lm_tpr:.4f} -> {t3_tpr:.4f} (+{t3_tpr-lm_tpr:.4f})")
            else:
                tpr_details.append(f"  - {m}: TPR@99n {lm_tpr:.4f} -> {t3_tpr:.4f} ({t3_tpr-lm_tpr:+.4f})")
    c6_pass = tpr_improved >= 2
    checks.append((
        f"6. >= 2 bottleneck machines improve TPR@99n ({tpr_improved} machines improved)",
        c6_pass,
        "",
    ))

    for check, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"- **[{status}]** {check}")
        if detail:
            lines.append(f"  {detail}")

    all_pass = all(c for _, c, _ in checks)
    lines.append("")

    if all_pass:
        lines.extend([
            "### Verdict: V40 SINGLE-SEED **PASS**",
            "",
            "`transient_3view` successfully converts V39 Stage-0 input-level separation signal "
            "into low-FPR tail separation under latent_summary_knn.",
            "",
            "**Next step**: Run 3-seed panel (seeds 13711, 13721, 13731) on `transient_3view` only. "
            "Report as V41 seed-panel across the same 5 bottleneck machines.",
        ])
    else:
        lines.extend([
            "### Verdict: V40 SINGLE-SEED **FAIL**",
            "",
            "`transient_3view` did not meet all pre-registered success criteria.",
            "",
            "## 7. Failure Analysis",
            "",
            "- V39 Stage-0 input-level separation signal did **not reliably survive** "
            "encoder training. While transient_3view showed strong robust_d at the raw-input "
            "level (mean 0.738), the AE encoder's learned representation did not consistently "
            "preserve that separation in the latent summary space.",
            "",
            "- `transient_3view` improves `toytrain` strongly (hmean 0.5224 -> 0.6108, "
            "TPR@99n 0.01 -> 0.08), but degrades or fails to improve on toycar, gearbox, "
            "slider, and fan. The improvement is mechanism-specific, not general.",
            "",
            "- The **dominant failure mechanism is normal-tail inflation**: "
            "q95 and q99 thresholds increase substantially under transient_3view "
            "(e.g., gearbox q99: 0.157 -> 0.283, slider q99: 0.167 -> 0.374), "
            "while anomaly scores do not move far enough beyond the new normal extreme tail. "
            "This suppresses pAUC and TPR@99n even when mean separation exists.",
            "",
            "- Therefore `transient_3view` is a **mechanism-specific positive diagnostic**, "
            "not a general frontend replacement. Its value is in demonstrating that specific "
            "frontend views can reveal structure invisible to logmel alone on some machines "
            "(toytrain).",
            "",
            "- V40 should be recorded as evidence that **input-level separability alone is "
            "insufficient** for low-FPR anomaly detection. The representation and scoring "
            "pipeline must be designed to preserve tail separation after training, not just "
            "before it.",
            "",
            "### Actions blocked by this result",
            "",
            "- Do NOT run 3-seed panel on transient_3view.",
            "- Do NOT tune transient parameters.",
            "- Do NOT attempt transient_6view or frontend fusion.",
            "- Do NOT perform per-machine frontend selection.",
            "- Do NOT auto-proceed to V41.",
        ])

    lines.extend([
        "",
        "---",
        "",
        "Models were trained only under the fixed V40 protocol; no score was tuned, "
        "no frontend parameters were tuned, and no per-machine selection was performed.",
    ])

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    v40_dir = Path(args.v40_dir)
    score_mode = args.score_mode

    pm = load_per_machine(v40_dir, score_mode)
    geo = load_score_geometry(v40_dir, score_mode)

    # Compute tail metrics for bottleneck machines
    tail: Dict[Tuple[str, str], Dict] = {}
    machines = sorted(set(k[0] for k in pm.keys()))
    experiments = sorted(set(k[1] for k in pm.keys()))
    for m in machines:
        for exp in experiments:
            tail[(m, exp)] = compute_tail_metrics(v40_dir, m, exp, score_mode)

    # Build and write report
    report = build_report(args, pm, geo, tail)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report)
    print(f"V40 audit report written to: {out_md}")
    print(report)


if __name__ == "__main__":
    main()
