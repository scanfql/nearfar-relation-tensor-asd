#!/usr/bin/env python
"""V55 frozen BEATs no-finetune audit."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.features.logmel import load_audio


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
BOTTLENECK = ["gearbox", "slider", "toycar", "toytrain", "fan"]
DEFAULT_BEATS_CKPTS = [
    "/data1/test/dacase2023_task2_baseline/pretrained_models/beats_official/BEATs_iter3_plus_AS2M.pt",
    "/data1/marine/MRLN_other_py/beats/BEATs_iter3_plus_AS2M.pt",
]


def parse_args():
    p = argparse.ArgumentParser(description="Audit frozen BEATs representation with kNN")
    p.add_argument("--data_root", required=True)
    p.add_argument("--score_seed_dir", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--machine", default="")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--beats_checkpoint", default="")
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def find_beats_checkpoint(explicit: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for cand in DEFAULT_BEATS_CKPTS:
        p = Path(cand)
        if p.exists():
            return p
    return None


def load_beats(ckpt_path: Path, device: str):
    beats_dir = ckpt_path.parent
    if str(beats_dir) not in sys.path:
        sys.path.insert(0, str(beats_dir))
    from BEATs import BEATs, BEATsConfig

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=requested)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.to(requested)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def beats_embed(model, waveform_list: list[np.ndarray], device: torch.device, batch_size: int = 16):
    embs = []
    with torch.no_grad():
        for start in range(0, len(waveform_list), batch_size):
            batch = waveform_list[start:start + batch_size]
            max_len = max(len(w) for w in batch)
            x = np.zeros((len(batch), max_len), dtype=np.float32)
            for i, w in enumerate(batch):
                x[i, :len(w)] = w.astype(np.float32)
            wav = torch.from_numpy(x).to(device)
            feats, _ = model.extract_features(wav)
            emb = feats.mean(dim=1)
            embs.append(emb.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float64)


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5):
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, len(train_n))
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray):
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95))
    q99 = float(np.quantile(normals, 0.99))
    return {
        "normal_q95": q95,
        "normal_q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "TPR_at_95n": float(np.mean(anomalies > q95)),
        "TPR_at_99n": float(np.mean(anomalies > q99)),
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x): return f"{x:.6f}"
def fmt4(x): return f"{x:.4f}"


def run_audit(args):
    ckpt = find_beats_checkpoint(args.beats_checkpoint)
    if ckpt is None:
        out_root = Path(args.output_root)
        out_root.mkdir(parents=True, exist_ok=True)
        missing = [{"kind": "beats_checkpoint", "path": args.beats_checkpoint or "search_failed"}]
        write_csv(out_root / "missing_paths.csv", missing)
        Path(args.doc_path).write_text(
            "# CTAF V55 Frozen BEATs Representation Audit\n\n"
            "BEATs checkpoint missing. Audit stopped without substitution.\n"
        )
        return False, None

    model, device = load_beats(ckpt, args.device)
    items = scan_dataset(Path(args.data_root))
    machines = [args.machine] if args.machine else MACHINES_ALL
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    baseline_summary = list(csv.DictReader((Path(args.score_seed_dir) / "summary.csv").open()))
    v37_full = next(r for r in baseline_summary if r["experiment"] == "full_5view" and r["score_mode"] == "latent_summary_knn")
    logmel_row = next(r for r in baseline_summary if r["experiment"] == "logmel_only" and r["score_mode"] == "latent_summary_knn")

    per_machine = []
    tail_rows = []
    for machine in machines:
        machine_items = filter_items(items, machine)
        train_items, test_items = split_train_test(machine_items, use_target_train=False)
        train_wavs = [load_audio(item.path) for item in train_items]
        test_wavs = [load_audio(item.path) for item in test_items]
        y_test = np.asarray([item.label for item in test_items], dtype=np.int64)
        d_test = np.asarray([item.domain for item in test_items])

        train_emb = beats_embed(model, train_wavs, device)
        test_emb = beats_embed(model, test_wavs, device)
        scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
        metrics = compute_metrics(y_test, scores, d_test)
        geometry = compute_score_geometry(scores, y_test, d_test)

        machine_dir = out_root / machine
        machine_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            machine_dir / f"{machine}_beats_embeddings.npz",
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            test_scores=scores,
            y_test=y_test,
            d_test=d_test.astype("U16"),
            beats_checkpoint=str(ckpt),
        )

        per_machine.append({
            "machine": machine,
            "hmean": metrics["hmean"],
            "pauc": metrics["pauc"],
            "source_auc": metrics["auc_source"],
            "target_auc": metrics["auc_target"],
            "ks": geometry["source_target_ks"],
        })
        tail_rows.append({"machine": machine, **tail_metrics(scores, y_test)})

    write_csv(out_root / "per_machine.csv", per_machine)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "summary.csv", [{
        "score_mode": "frozen_beats_knn",
        "mean_hmean": fmt(np.mean([r["hmean"] for r in per_machine])),
        "mean_pauc": fmt(np.mean([r["pauc"] for r in per_machine])),
        "mean_ks": fmt(np.mean([r["ks"] for r in per_machine])),
    }])

    beats_h = np.mean([r["hmean"] for r in per_machine])
    beats_p = np.mean([r["pauc"] for r in per_machine])
    beats_ks = np.mean([r["ks"] for r in per_machine])
    sanity_mode = len(machines) == 1
    improved_tpr = 0
    baseline_tail = {}
    for machine in machines:
        arr = np.load(Path(args.score_seed_dir) / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz", allow_pickle=True)
        baseline_tail[machine] = tail_metrics(arr["test_scores"].astype(np.float64), arr["y_test"].astype(np.int64))
        if next(r for r in tail_rows if r["machine"] == machine)["TPR_at_99n"] > baseline_tail[machine]["TPR_at_99n"] and machine in BOTTLENECK:
            improved_tpr += 1
    q99_infl = [m for m in machines if next(r for r in tail_rows if r["machine"] == m)["normal_q99"] > baseline_tail[m]["normal_q99"] * 1.05]
    pm = {r["machine"]: r for r in per_machine}
    # compare to V37 baseline per-machine
    base_pm_rows = list(csv.DictReader((Path(args.score_seed_dir) / "per_machine.csv").open()))
    base_pm = {r["machine"]: r for r in base_pm_rows if r["experiment"] == "full_5view" and r["score_mode"] == "latent_summary_knn"}
    bearing_collapse = ("bearing" in pm and "bearing" in base_pm and pm["bearing"]["hmean"] < float(base_pm["bearing"]["hmean"]) - 0.03)
    valve_collapse = ("valve" in pm and "valve" in base_pm and pm["valve"]["hmean"] < float(base_pm["valve"]["hmean"]) - 0.03)
    toycar_degrade = ("toycar" in pm and "toycar" in base_pm and pm["toycar"]["pauc"] < float(base_pm["toycar"]["pauc"]) - 0.01)
    positive = (
        beats_h >= float(v37_full["mean_hmean"]) - 0.003 and
        beats_p >= float(v37_full["mean_pauc"]) and
        improved_tpr >= 2 and
        not bearing_collapse and not valve_collapse and
        not toycar_degrade and
        not q99_infl
    )
    verdict = "V55 positive" if positive else "V55 negative"
    if sanity_mode:
        verdict = "sanity complete"

    lines = [
        "# CTAF V55 Frozen BEATs Representation Audit",
        "",
        "## 1. V54 prereg summary",
        "",
        "- Stage 10 tests a new external representation hypothesis",
        "- no fine-tuning, no LoRA, no adapter, no DLCL",
        "- backend is fixed kNN on frozen BEATs utterance embeddings",
        "",
        "## 2. BEATs checkpoint path and provenance",
        "",
        f"- checkpoint path: `{ckpt}`",
        "- provenance: official BEATs 90M AudioSet-pretrained checkpoint found on server",
        "",
        "## 3. Confirmation that BEATs is frozen",
        "",
        "- encoder loaded in eval mode",
        "- `torch.no_grad` used for feature extraction",
        "- no parameter updates",
        "- no classifier head training",
        "- no pooling head training",
        "",
        "## 4. Embedding extraction protocol",
        "",
        "- input: raw waveform",
        "- resample: 16 kHz via existing waveform loader",
        "- feature extraction: BEATs internal fbank preprocessing",
        "- embedding: final-layer frame embeddings",
        "- pooling: time mean pooling",
        "- backend: cosine-distance kNN with `k=5` on source train-normal memory only",
        "",
        "## 5. Overall comparison",
        "",
        "| system | mean hmean | mean pAUC | mean KS |",
        "|---|---:|---:|---:|",
        f"| V37 full_5view latent_summary_knn | {v37_full['mean_hmean']} | {v37_full['mean_pauc']} | {v37_full['mean_source_target_ks']} |",
        f"| logmel_only latent_summary_knn | {logmel_row['mean_hmean']} | {logmel_row['mean_pauc']} | {logmel_row['mean_source_target_ks']} |",
        f"| frozen BEATs + kNN | {fmt(beats_h)} | {fmt(beats_p)} | {fmt(beats_ks)} |",
        "",
        "## 6. Per-machine hmean / pAUC / KS",
        "",
        "| machine | hmean | pAUC | KS |",
        "|---|---:|---:|---:|",
    ]
    for machine in machines:
        row = pm[machine]
        lines.append(f"| {machine} | {fmt(row['hmean'])} | {fmt(row['pauc'])} | {fmt(row['ks'])} |")
    lines.extend([
        "",
        "## 7. Tail table",
        "",
        "| machine | normal q95 | normal q99 | anomaly median | anomaly q75 | TPR@95n | TPR@99n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in tail_rows:
        lines.append(
            f"| {row['machine']} | {fmt4(row['normal_q95'])} | {fmt4(row['normal_q99'])} | "
            f"{fmt4(row['anomaly_median'])} | {fmt4(row['anomaly_q75'])} | {fmt4(row['TPR_at_95n'])} | {fmt4(row['TPR_at_99n'])} |"
        )
    lines.extend([
        "",
        "## 8. q99 inflation check",
        "",
        f"- machines with >5% q99 inflation relative to V37 baseline: `{', '.join(q99_infl) if q99_infl else 'none'}`",
        "",
        "## 9. Final verdict",
        "",
        f"`{verdict}`",
    ])
    if sanity_mode:
        lines.extend([
            "",
            "This run is a 1-machine sanity check only.",
            "",
            "- it verifies checkpoint discovery",
            "- it verifies frozen BEATs embedding extraction",
            "- it verifies kNN scoring on a single machine",
            "- it verifies output file generation",
            "",
            "No BEATs fine-tuning occurred. No protocol substitutions were made.",
        ])
    else:
        lines.extend([
            "",
            "Success gates:",
            "",
            f"- mean hmean >= V37 - 0.003: `{'PASS' if beats_h >= float(v37_full['mean_hmean']) - 0.003 else 'FAIL'}`",
            f"- mean pAUC >= V37: `{'PASS' if beats_p >= float(v37_full['mean_pauc']) else 'FAIL'}`",
            f"- >=2 bottleneck machines improve TPR@99n: `{improved_tpr}/5`",
            f"- bearing / valve do not collapse: `{'PASS' if not (bearing_collapse or valve_collapse) else 'FAIL'}`",
            f"- toycar does not clearly degrade: `{'PASS' if not toycar_degrade else 'FAIL'}`",
            f"- normal q99 does not clearly inflate: `{'PASS' if not q99_infl else 'FAIL'}`",
            "",
            "No BEATs fine-tuning occurred. No protocol substitutions were made. No V56 was opened.",
        ])
    Path(args.doc_path).write_text("\n".join(lines) + "\n")
    return True, str(ckpt)


def main():
    args = parse_args()
    ok, ckpt = run_audit(args)
    if not ok:
        print("BEATs checkpoint missing. Audit stopped.")
    else:
        print(f"BEATs checkpoint used: {ckpt}")
        print(f"Results saved to: {args.output_root}")
        print(f"Report saved to: {args.doc_path}")


if __name__ == "__main__":
    main()
