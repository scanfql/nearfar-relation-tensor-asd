#!/usr/bin/env python
"""Low-FPR failure attribution diagnostic for CTAF bottleneck machines."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import load_audio, extract_logmel
from src.features.deltas import compute_delta, compute_delta_delta
from src.features.edge import compute_edge
from src.features.enhanced import compute_enhanced


DEFAULT_MACHINES = "gearbox,slider,toycar,toytrain,fan"


@dataclass
class SampleRecord:
    index: int
    score: float
    label: int
    domain: str
    path: Path
    nearest_idx: int
    nearest_path: Path
    nn_distance: float
    score_rank: int
    group: str


def parse_args():
    p = argparse.ArgumentParser(description="Diagnose low-FPR false negatives")
    p.add_argument("--data_root", required=True)
    p.add_argument("--reference_dir", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--machines", default=DEFAULT_MACHINES)
    p.add_argument("--max_per_group", type=int, default=20)
    return p.parse_args()


def align_time(x: np.ndarray, t_ref: int) -> np.ndarray:
    if x.shape[1] == t_ref:
        return x
    if x.shape[1] > t_ref:
        return x[:, :t_ref]
    pad = t_ref - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="edge")


def moving_average_time(x: np.ndarray, win: int = 9) -> np.ndarray:
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, x)


def compute_extra_views(audio: np.ndarray, logmel: np.ndarray) -> dict[str, np.ndarray]:
    modulation_low = moving_average_time(logmel, win=11).astype(np.float32)
    modulation_high = (logmel - modulation_low).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=40).astype(np.float32)
    flux = librosa.onset.onset_strength(y=audio, sr=16000, hop_length=512)
    flux = np.asarray(flux, dtype=np.float32)[None, :]
    flux = np.repeat(flux, logmel.shape[0], axis=0)
    flux = align_time(flux, logmel.shape[1])
    return {
        "modulation_low": modulation_low,
        "modulation_high": modulation_high,
        "cepstral_map": mfcc,
        "spectral_flux_map": flux,
    }


def load_primary_views(path: Path) -> dict[str, np.ndarray]:
    audio = load_audio(path)
    logmel = extract_logmel(audio)
    edge = compute_edge(logmel)
    return {
        "audio": audio,
        "logmel": logmel,
        "delta": compute_delta(logmel),
        "deltadelta": compute_delta_delta(logmel),
        "edge": edge,
        "enhanced": compute_enhanced(edge),
        **compute_extra_views(audio, logmel),
    }


def normalized_nn(memory: np.ndarray, queries: np.ndarray):
    mem_norm = memory / np.maximum(np.linalg.norm(memory, axis=1, keepdims=True), 1e-12)
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(q_norm * q_norm, axis=1, keepdims=True)
        + np.sum(mem_norm * mem_norm, axis=1)[None, :]
        - 2.0 * (q_norm @ mem_norm.T)
    )
    d2 = np.maximum(d2, 0.0)
    nn_idx = np.argmin(d2, axis=1)
    nn_dist = np.sqrt(d2[np.arange(len(q_norm)), nn_idx])
    return nn_idx.astype(int), nn_dist.astype(np.float64)


def select_group(indices: np.ndarray, scores: np.ndarray, max_n: int, descending: bool = True):
    order = np.argsort(scores[indices])
    if descending:
        order = order[::-1]
    picked = indices[order][:max_n]
    return picked.tolist()


def create_panel(record: SampleRecord, sample_views: dict, nearest_views: dict, out_path: Path):
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    axs = axes.ravel()

    t = np.arange(sample_views["audio"].shape[0]) / 16000.0
    axs[0].plot(t, sample_views["audio"], linewidth=0.7)
    axs[0].set_title("waveform")

    panels = [
        ("logMel", sample_views["logmel"]),
        ("delta", sample_views["delta"]),
        ("deltadelta", sample_views["deltadelta"]),
        ("edge", sample_views["edge"]),
        ("enhanced", sample_views["enhanced"]),
        ("nearest normal logMel", nearest_views["logmel"]),
        ("input difference map", np.abs(sample_views["logmel"] - align_time(nearest_views["logmel"], sample_views["logmel"].shape[1]))),
        ("modulation_low", sample_views["modulation_low"]),
        ("modulation_high", sample_views["modulation_high"]),
        ("cepstral map", sample_views["cepstral_map"]),
        ("spectral flux map", sample_views["spectral_flux_map"]),
    ]
    for ax, (title, arr) in zip(axs[1:], panels):
        ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{record.group} | score={record.score:.4f} | rank={record.score_rank} | "
        f"nn_dist={record.nn_distance:.4f}\n"
        f"sample={record.path.name} | nearest={record.nearest_path.name}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def group_stats(records: list[SampleRecord], diffs: list[float]):
    if not records:
        return {
            "count": 0,
            "mean_score": 0.0,
            "mean_nn_dist": 0.0,
            "mean_input_diff": 0.0,
        }
    return {
        "count": len(records),
        "mean_score": float(np.mean([r.score for r in records])),
        "mean_nn_dist": float(np.mean([r.nn_distance for r in records])),
        "mean_input_diff": float(np.mean(diffs)) if diffs else 0.0,
    }


def infer_attribution(missed, normal_tail, detected):
    if missed["count"] == 0:
        return "unclear", "No missed anomalies under the current q99 threshold."

    input_ratio = missed["mean_input_diff"] / max(normal_tail["mean_input_diff"], 1e-8)
    latent_ratio = missed["mean_nn_dist"] / max(normal_tail["mean_nn_dist"], 1e-8)
    detected_input_ratio = detected["mean_input_diff"] / max(missed["mean_input_diff"], 1e-8) if detected["count"] else 1.0
    detected_latent_ratio = detected["mean_nn_dist"] / max(missed["mean_nn_dist"], 1e-8) if detected["count"] else 1.0

    if input_ratio < 1.10 and latent_ratio < 1.10:
        return "A. input-limited", "Missed anomalies look too similar to normal-tail samples already at the input level."
    if input_ratio >= 1.15 and latent_ratio < 1.10:
        return "B. encoder-limited", "Input differences exist, but missed anomalies stay mixed with normal-tail samples in latent space."
    if input_ratio >= 1.15 and latent_ratio >= 1.10 and detected_latent_ratio <= 1.10:
        return "C. scoring-limited", "Latent separation exists, but the current score/threshold still fails to promote missed anomalies into the tail."
    return "D. unclear", "Evidence is mixed; missed anomalies differ from normal-tail samples, but not in a cleanly attributable way."


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ref_dir = Path(args.reference_dir)
    out_root = Path(args.output_root)
    doc_path = Path(args.doc_path)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    out_root.mkdir(parents=True, exist_ok=True)

    items = scan_dataset(data_root)
    doc_lines = [
        "# CTAF Low-FPR Failure Attribution",
        "",
        "Reference scoring source: `V37 seed_13711 / full_5view / latent_summary_knn`.",
        "",
    ]

    for machine in machines:
        machine_out = out_root / machine
        machine_out.mkdir(parents=True, exist_ok=True)
        for group in ["missed_anomaly", "normal_tail", "detected_anomaly"]:
            (machine_out / group).mkdir(exist_ok=True)

        m_items = filter_items(items, machine)
        train_items, test_items = split_train_test(m_items, use_target_train=False)
        memory_items = [x for x in m_items if x.split == "train" and x.label in (0, None) and x.domain == "source"]

        npz_path = ref_dir / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
        arr = np.load(npz_path, allow_pickle=True)
        test_scores = arr["test_scores"].astype(np.float64)
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype("U16")
        test_features = arr["test_latent_summaries"].astype(np.float64)
        memory_features = arr["memory_latent_summaries"].astype(np.float64)
        nn_idx, nn_dist = normalized_nn(memory_features, test_features)

        normal_scores = test_scores[y_test == 0]
        q95 = float(np.quantile(normal_scores, 0.95))
        q99 = float(np.quantile(normal_scores, 0.99))

        anomaly_idx = np.where(y_test == 1)[0]
        normal_idx = np.where(y_test == 0)[0]
        missed_idx = anomaly_idx[test_scores[anomaly_idx] <= q99]
        detected_idx = anomaly_idx[test_scores[anomaly_idx] > q99]
        normal_tail_idx = normal_idx[(test_scores[normal_idx] > q95) & (test_scores[normal_idx] <= q99)]

        selected = {
            "missed_anomaly": select_group(missed_idx, test_scores, args.max_per_group, descending=True),
            "normal_tail": select_group(normal_tail_idx, test_scores, args.max_per_group, descending=True),
            "detected_anomaly": select_group(detected_idx, test_scores, args.max_per_group, descending=True),
        }

        order = np.argsort(test_scores)
        rank_map = {idx: rank + 1 for rank, idx in enumerate(order[::-1])}
        metadata_rows = []
        grouped_records = {k: [] for k in selected}
        grouped_input_diff = {k: [] for k in selected}

        for group_name, idx_list in selected.items():
            for idx in idx_list:
                item = test_items[idx]
                nearest_item = memory_items[nn_idx[idx]]
                record = SampleRecord(
                    index=idx,
                    score=float(test_scores[idx]),
                    label=int(y_test[idx]),
                    domain=str(d_test[idx]),
                    path=item.path,
                    nearest_idx=int(nn_idx[idx]),
                    nearest_path=nearest_item.path,
                    nn_distance=float(nn_dist[idx]),
                    score_rank=int(rank_map[idx]),
                    group=group_name,
                )
                sample_views = load_primary_views(item.path)
                nearest_views = load_primary_views(nearest_item.path)
                aligned_nearest = align_time(nearest_views["logmel"], sample_views["logmel"].shape[1])
                input_diff = float(np.mean(np.abs(sample_views["logmel"] - aligned_nearest)))
                grouped_input_diff[group_name].append(input_diff)
                grouped_records[group_name].append(record)

                out_path = machine_out / group_name / f"{record.score_rank:03d}_{item.path.stem}.png"
                create_panel(record, sample_views, nearest_views, out_path)

                metadata_rows.append({
                    "group": group_name,
                    "score_rank": record.score_rank,
                    "score": f"{record.score:.6f}",
                    "nn_distance": f"{record.nn_distance:.6f}",
                    "domain": record.domain,
                    "sample_path": str(record.path),
                    "nearest_normal_path": str(record.nearest_path),
                    "input_diff_mean_abs_logmel": f"{input_diff:.6f}",
                })

        with (machine_out / "metadata.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else ["group"])
            writer.writeheader()
            writer.writerows(metadata_rows)

        missed_stats = group_stats(grouped_records["missed_anomaly"], grouped_input_diff["missed_anomaly"])
        normal_tail_stats = group_stats(grouped_records["normal_tail"], grouped_input_diff["normal_tail"])
        detected_stats = group_stats(grouped_records["detected_anomaly"], grouped_input_diff["detected_anomaly"])
        attribution, reason = infer_attribution(missed_stats, normal_tail_stats, detected_stats)

        candidate_frontend = "yes" if (
            missed_stats["mean_input_diff"] > normal_tail_stats["mean_input_diff"] * 1.20
            and detected_stats["mean_input_diff"] >= missed_stats["mean_input_diff"] * 0.95
        ) else "weak/unclear"

        visually_distinct = "yes" if missed_stats["mean_input_diff"] > normal_tail_stats["mean_input_diff"] * 1.15 else "weak"
        latent_mixed = "yes" if missed_stats["mean_nn_dist"] <= normal_tail_stats["mean_nn_dist"] * 1.10 else "no/partial"

        doc_lines.extend([
            f"## {machine}",
            "",
            f"- output dir: `{machine_out}`",
            f"- missed anomalies: `{len(missed_idx)}`",
            f"- normal-tail samples: `{len(normal_tail_idx)}`",
            f"- detected anomalies: `{len(detected_idx)}`",
            f"- normal q95 / q99: `{q95:.6f} / {q99:.6f}`",
            f"- missed anomaly visually distinguishable on 5-view input: `{visually_distinct}`",
            f"- missed anomaly mixed with normal-tail in latent space: `{latent_mixed}`",
            f"- candidate frontend evidence from modulation/cepstral/transient views: `{candidate_frontend}`",
            "",
            "Group diagnostics:",
            "",
            f"- missed anomaly: count={missed_stats['count']}, mean_score={missed_stats['mean_score']:.6f}, mean_nn_dist={missed_stats['mean_nn_dist']:.6f}, mean_input_diff={missed_stats['mean_input_diff']:.6f}",
            f"- normal tail: count={normal_tail_stats['count']}, mean_score={normal_tail_stats['mean_score']:.6f}, mean_nn_dist={normal_tail_stats['mean_nn_dist']:.6f}, mean_input_diff={normal_tail_stats['mean_input_diff']:.6f}",
            f"- detected anomaly: count={detected_stats['count']}, mean_score={detected_stats['mean_score']:.6f}, mean_nn_dist={detected_stats['mean_nn_dist']:.6f}, mean_input_diff={detected_stats['mean_input_diff']:.6f}",
            "",
            f"Attribution: `{attribution}`",
            "",
            f"- rationale: {reason}",
            "",
        ])

    overall_attr = []
    for line in doc_lines:
        if line.startswith("Attribution:"):
            overall_attr.append(line.split("`")[1])

    if overall_attr.count("A. input-limited") >= 2:
        next_step = "Prefer modulation / cepstral / transient frontend work."
    elif overall_attr.count("B. encoder-limited") >= 2:
        next_step = "Prefer local representation learning."
    elif overall_attr.count("C. scoring-limited") >= 2:
        next_step = "Scoring may still matter, but V38 already argues against ordinary density scoring."
    else:
        next_step = "Pause version search; failure attribution is still mixed."

    doc_lines.extend([
        "## Final Attribution",
        "",
        "- A = input-limited",
        "- B = encoder-limited",
        "- C = scoring-limited",
        "- D = unclear",
        "",
        f"Next-step suggestion: {next_step}",
        "",
        "This diagnostic is offline only. No new model was trained, no score was tuned, and no per-machine method was selected.",
    ])

    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(doc_lines) + "\n")


if __name__ == "__main__":
    main()
