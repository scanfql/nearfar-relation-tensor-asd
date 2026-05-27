#!/usr/bin/env python
"""Aggregate fixed-seed CTAF result folders into a reproducible markdown audit."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate CTAF fixed-seed runs")
    p.add_argument("--seed_dirs", required=True,
                   help="Comma-separated result directories, one per seed")
    p.add_argument("--baseline_dir", required=True,
                   help="Reference result directory for the frozen baseline")
    p.add_argument("--output_md", required=True,
                   help="Markdown report path")
    p.add_argument("--focus_experiment", default="full_5view")
    p.add_argument("--matched_experiment", default="logmel_only")
    p.add_argument("--score_mode", default="latent_summary_knn")
    p.add_argument("--panel_name", default="V16 fixed 5-seed audit")
    return p.parse_args()


def read_csv(path: Path):
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def find_row(rows, experiment: str, score_mode: str):
    for row in rows:
        if row["experiment"] == experiment and row["score_mode"] == score_mode:
            return row
    raise KeyError(f"Missing row for experiment={experiment} score_mode={score_mode} in {rows}")


def to_float(row, key):
    return float(row[key])


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def fmt(x):
    return f"{x:.6f}"


def fmt_delta(x):
    return f"{x:+.6f}"


def main():
    args = parse_args()

    seed_dirs = [Path(x.strip()) for x in args.seed_dirs.split(",") if x.strip()]
    baseline_dir = Path(args.baseline_dir)
    output_md = Path(args.output_md)

    baseline_summary = read_csv(baseline_dir / "summary.csv")
    baseline_per_machine = read_csv(baseline_dir / "per_machine.csv")
    baseline_focus = find_row(baseline_summary, args.focus_experiment, args.score_mode)
    baseline_match = find_row(baseline_summary, args.matched_experiment, args.score_mode)

    seed_rows = []
    per_machine_by_seed = {}
    for seed_dir in seed_dirs:
        summary = read_csv(seed_dir / "summary.csv")
        per_machine = read_csv(seed_dir / "per_machine.csv")
        focus = find_row(summary, args.focus_experiment, args.score_mode)
        match = find_row(summary, args.matched_experiment, args.score_mode)
        seed_rows.append({
            "seed_dir": seed_dir,
            "seed": seed_dir.name.split("_")[-1],
            "focus": focus,
            "match": match,
        })
        per_machine_by_seed[seed_dir.name] = per_machine

    hmeans = [to_float(x["focus"], "mean_hmean") for x in seed_rows]
    paucs = [to_float(x["focus"], "mean_pauc") for x in seed_rows]
    kss = [to_float(x["focus"], "mean_source_target_ks") for x in seed_rows]
    match_hmeans = [to_float(x["match"], "mean_hmean") for x in seed_rows]
    match_paucs = [to_float(x["match"], "mean_pauc") for x in seed_rows]

    machine_stats = defaultdict(lambda: {
        "focus_hmean": [], "focus_pauc": [], "focus_ks": [],
        "match_hmean": [], "match_pauc": [], "match_ks": [],
    })
    for seed_name, rows in per_machine_by_seed.items():
        for row in rows:
            if row["score_mode"] != args.score_mode:
                continue
            machine = row["machine"]
            if row["experiment"] == args.focus_experiment:
                machine_stats[machine]["focus_hmean"].append(float(row["hmean"]))
                machine_stats[machine]["focus_pauc"].append(float(row["pauc"]))
                machine_stats[machine]["focus_ks"].append(float(row["source_auc"]) - float(row["target_auc"]))
            elif row["experiment"] == args.matched_experiment:
                machine_stats[machine]["match_hmean"].append(float(row["hmean"]))
                machine_stats[machine]["match_pauc"].append(float(row["pauc"]))
                machine_stats[machine]["match_ks"].append(float(row["source_auc"]) - float(row["target_auc"]))

    baseline_machine = {}
    for row in baseline_per_machine:
        if row["experiment"] == args.focus_experiment and row["score_mode"] == args.score_mode:
            baseline_machine[row["machine"]] = {
                "hmean": float(row["hmean"]),
                "pauc": float(row["pauc"]),
            }

    seeds_beating_match = sum(
        1 for row in seed_rows
        if to_float(row["focus"], "mean_hmean") > to_float(row["match"], "mean_hmean")
        and to_float(row["focus"], "mean_pauc") >= to_float(row["match"], "mean_pauc")
    )
    seeds_beating_baseline = sum(
        1 for row in seed_rows
        if to_float(row["focus"], "mean_hmean") > to_float(baseline_focus, "mean_hmean")
        and to_float(row["focus"], "mean_pauc") >= to_float(baseline_focus, "mean_pauc")
    )

    lines = []
    lines.append(f"# {args.panel_name}")
    lines.append("")
    lines.append("## Frozen Baseline")
    lines.append("")
    lines.append(f"- baseline dir: `{baseline_dir}`")
    lines.append(f"- focus: `{args.focus_experiment} / {args.score_mode}`")
    lines.append(f"- matched baseline: `{args.matched_experiment} / {args.score_mode}`")
    lines.append(f"- frozen hmean: `{fmt(to_float(baseline_focus, 'mean_hmean'))}`")
    lines.append(f"- frozen pAUC: `{fmt(to_float(baseline_focus, 'mean_pauc'))}`")
    lines.append(f"- frozen KS: `{float(baseline_focus['mean_source_target_ks']):.4f}`")
    lines.append("")
    lines.append("## Seed Panel")
    lines.append("")
    lines.append("| seed | logmel hmean | V16 hmean | delta | logmel pAUC | V16 pAUC | delta | V16 KS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in seed_rows:
        focus = row["focus"]
        match = row["match"]
        lines.append(
            f"| {row['seed']} | {fmt(to_float(match, 'mean_hmean'))} | {fmt(to_float(focus, 'mean_hmean'))} | "
            f"{fmt_delta(to_float(focus, 'mean_hmean') - to_float(match, 'mean_hmean'))} | "
            f"{fmt(to_float(match, 'mean_pauc'))} | {fmt(to_float(focus, 'mean_pauc'))} | "
            f"{fmt_delta(to_float(focus, 'mean_pauc') - to_float(match, 'mean_pauc'))} | "
            f"{float(focus['mean_source_target_ks']):.4f} |"
        )
    lines.append("")
    lines.append("## Aggregate Stability")
    lines.append("")
    lines.append("| metric | mean | std | min | max | baseline | delta(mean-baseline) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| hmean | {fmt(mean(hmeans))} | {fmt(std(hmeans))} | {fmt(min(hmeans))} | {fmt(max(hmeans))} | "
        f"{fmt(to_float(baseline_focus, 'mean_hmean'))} | {fmt_delta(mean(hmeans) - to_float(baseline_focus, 'mean_hmean'))} |"
    )
    lines.append(
        f"| pAUC | {fmt(mean(paucs))} | {fmt(std(paucs))} | {fmt(min(paucs))} | {fmt(max(paucs))} | "
        f"{fmt(to_float(baseline_focus, 'mean_pauc'))} | {fmt_delta(mean(paucs) - to_float(baseline_focus, 'mean_pauc'))} |"
    )
    lines.append(
        f"| KS | {fmt(mean(kss))} | {fmt(std(kss))} | {fmt(min(kss))} | {fmt(max(kss))} | "
        f"{float(baseline_focus['mean_source_target_ks']):.4f} | {fmt_delta(mean(kss) - float(baseline_focus['mean_source_target_ks']))} |"
    )
    lines.append("")
    lines.append("## Per-Machine Stability")
    lines.append("")
    lines.append("| machine | baseline hmean | seed-panel hmean mean | std | baseline pAUC | seed-panel pAUC mean | std | observation |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for machine in sorted(machine_stats):
        focus_hmean_mean = mean(machine_stats[machine]["focus_hmean"])
        focus_pauc_mean = mean(machine_stats[machine]["focus_pauc"])
        lines.append(
            f"| {machine} | {fmt(baseline_machine[machine]['hmean'])} | {fmt(focus_hmean_mean)} | {fmt(std(machine_stats[machine]['focus_hmean']))} | "
            f"{fmt(baseline_machine[machine]['pauc'])} | {fmt(focus_pauc_mean)} | {fmt(std(machine_stats[machine]['focus_pauc']))} | "
            f"hmean delta {fmt_delta(focus_hmean_mean - baseline_machine[machine]['hmean'])}, "
            f"pAUC delta {fmt_delta(focus_pauc_mean - baseline_machine[machine]['pauc'])} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"- seeds beating matched logmel_only on both hmean and pAUC: `{seeds_beating_match}/{len(seed_rows)}`"
    )
    lines.append(
        f"- seeds beating frozen V16 point estimate on both hmean and pAUC: `{seeds_beating_baseline}/{len(seed_rows)}`"
    )
    lines.append(
        f"- hmean std across seeds: `{fmt(std(hmeans))}`; pAUC std: `{fmt(std(paucs))}`; KS std: `{fmt(std(kss))}`"
    )
    if seeds_beating_match == len(seed_rows):
        lines.append("- V16 appears stable against the matched logmel baseline across the fixed seed panel.")
    else:
        lines.append("- V16 is not uniformly stable against the matched logmel baseline across the fixed seed panel.")
    if seeds_beating_baseline >= max(1, len(seed_rows) // 2 + 1):
        lines.append("- The original frozen V16 run does not look like a lucky outlier.")
    else:
        lines.append("- The original frozen V16 run may be optimistic relative to the fixed seed panel.")

    output_md.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
