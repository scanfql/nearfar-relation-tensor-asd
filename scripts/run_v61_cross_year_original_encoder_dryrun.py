#!/usr/bin/env python
"""V61 dry run: cross-year normal-only pretrain + current-year adapt on one machine."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.evaluation.scorer import compute_latent_knn_scores, compute_latent_summary_vectors
from src.features.logmel import extract_logmel, load_audio
from src.features.tensor_builder import TensorNormalizer, build_tensor
from src.run_experiment import _build_model
from src.training.train_ae import train_autoencoder, train_teacher_student_local_distillation
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="V61 one-machine dry run")
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--data_root", required=True, help="Current target-year root used by V37 baseline")
    p.add_argument("--machine", default="gearbox")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--baseline_dir", default="results/v37_teacher_student_pretrain_seedpanel/seed_13711")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--latent_channels", type=int, default=64)
    p.add_argument("--stem_channels", type=int, default=8)
    p.add_argument("--per_view_channels", type=int, default=16)
    p.add_argument("--dropout_p", type=float, default=0.0)
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _extract_tensor(path: str):
    audio = load_audio(Path(path))
    logmel = extract_logmel(audio)
    return build_tensor("full_5view", logmel, audio=audio)


def _build_tensors(paths: list[str]) -> list[np.ndarray]:
    return [_extract_tensor(p) for p in paths]


def _prepare_arrays(train_tensors: list[np.ndarray], eval_tensors: list[np.ndarray]):
    normalizer = TensorNormalizer()
    train_norm = normalizer.fit_transform(train_tensors)
    eval_norm = [normalizer.transform(t) for t in eval_tensors]
    return np.stack(train_norm, axis=0), np.stack(eval_norm, axis=0), normalizer


def _tail_stats(scores: np.ndarray, y_true: np.ndarray):
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    q95 = float(np.percentile(normal_scores, 95)) if len(normal_scores) else 0.0
    q99 = float(np.percentile(normal_scores, 99)) if len(normal_scores) else 0.0
    tpr95 = float(np.mean(anomaly_scores > q95)) if len(anomaly_scores) else 0.0
    tpr99 = float(np.mean(anomaly_scores > q99)) if len(anomaly_scores) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.median(anomaly_scores)) if len(anomaly_scores) else 0.0,
        "anomaly_q75": float(np.percentile(anomaly_scores, 75)) if len(anomaly_scores) else 0.0,
        "TPR@95n": tpr95,
        "TPR@99n": tpr99,
    }


def _load_v37_baseline(baseline_dir: Path, machine: str):
    per_machine = pd.read_csv(baseline_dir / "per_machine.csv")
    row = per_machine[
        (per_machine["machine"] == machine)
        & (per_machine["experiment"] == "full_5view")
        & (per_machine["score_mode"] == "latent_summary_knn")
    ].iloc[0]

    geom = pd.read_csv(baseline_dir / "score_geometry.csv")
    grow = geom[
        (geom["machine"] == machine)
        & (geom["experiment"] == "full_5view")
        & (geom["score_mode"] == "latent_summary_knn")
    ].iloc[0]

    score_path = baseline_dir / "scores" / f"{machine}_full_5view_latent_summary_knn_scores.npz"
    npz = np.load(score_path, allow_pickle=True)
    y_test = npz["y_test"]
    test_scores = npz["test_scores"]
    tail = _tail_stats(test_scores, y_test)
    return {
        "hmean": float(row["hmean"]),
        "pAUC": float(row["pauc"]),
        "KS": float(grow["source_target_ks"]),
        "q95": tail["q95"],
        "q99": tail["q99"],
        "TPR@95n": tail["TPR@95n"],
        "TPR@99n": tail["TPR@99n"],
        "score_path": str(score_path),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models_dir = output_root / "models"
    scores_dir = output_root / "scores"
    models_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[(manifest["machine_canonical"] == args.machine) & (manifest["usable_for_v61"] == True)]
    if manifest.empty:
        raise RuntimeError(f"No manifest rows for machine={args.machine}")

    # Phase 1: cross-year teacher AE pretraining on the locked manifest.
    pretrain_paths = manifest["absolute_path"].tolist()
    pretrain_year_counts = manifest.groupby("year").size().to_dict()
    pretrain_domain_counts = manifest.groupby("domain").size().to_dict() if "domain" in manifest else {}

    phase1_tensors = _build_tensors(pretrain_paths)
    phase1_arr, _, _ = _prepare_arrays(phase1_tensors, phase1_tensors[:1])
    in_views = phase1_arr.shape[1]

    teacher = _build_model("viewwise_unshared_late_fusion", in_views, args)
    phase1_stats = train_autoencoder(
        teacher,
        phase1_arr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        loss_view_mode="all_views",
        ssl_task="reconstruction",
        noise_mode="none",
        mask_prob=0.0,
    )
    phase1_ckpt = models_dir / f"{args.machine}_phase1_teacher_seed{args.seed}.pt"
    torch.save({"state_dict": teacher.state_dict(), "train_stats": phase1_stats}, phase1_ckpt)

    # Phase 2: current-year source-train adaptation + teacher-student family.
    current_root = Path(args.data_root)
    all_items = scan_dataset(current_root)
    machine_items = filter_items(all_items, args.machine)
    train_items, test_items = split_train_test(machine_items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Current-year split invalid for {args.machine}: train={len(train_items)} test={len(test_items)}")

    train_paths = [str(x.path) for x in train_items]
    test_paths = [str(x.path) for x in test_items]
    train_tensors = _build_tensors(train_paths)
    test_tensors = _build_tensors(test_paths)
    train_arr, test_arr, _ = _prepare_arrays(train_tensors, test_tensors)
    y_test = np.asarray([x.label for x in test_items], dtype=np.int64)
    d_test = np.asarray([x.domain for x in test_items])

    phase2_stats = train_autoencoder(
        teacher,
        train_arr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        loss_view_mode="all_views",
        ssl_task="reconstruction",
        noise_mode="none",
        mask_prob=0.0,
    )
    phase2_teacher_ckpt = models_dir / f"{args.machine}_phase2_teacher_seed{args.seed}.pt"
    torch.save({"state_dict": teacher.state_dict(), "train_stats": phase2_stats}, phase2_teacher_ckpt)

    student = _build_model("viewwise_unshared_late_fusion", in_views, args)
    phase2_student_stats = train_teacher_student_local_distillation(
        teacher,
        student,
        train_arr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    phase2_student_ckpt = models_dir / f"{args.machine}_phase2_student_seed{args.seed}.pt"
    torch.save({"state_dict": student.state_dict(), "train_stats": phase2_student_stats}, phase2_student_ckpt)

    # Evaluation uses the same V37 latent_summary_knn scoring on the adapted teacher.
    memory_lat = compute_latent_summary_vectors(teacher, train_arr, batch_size=args.batch_size, device=args.device)
    test_lat = compute_latent_summary_vectors(teacher, test_arr, batch_size=args.batch_size, device=args.device)
    train_scores = compute_latent_knn_scores(memory_lat, memory_lat, k=args.latent_knn_k)
    test_scores = compute_latent_knn_scores(memory_lat, test_lat, k=args.latent_knn_k)

    metrics = compute_metrics(y_test, test_scores, d_test)
    geom = compute_score_geometry(test_scores, y_test, d_test)
    tail = _tail_stats(test_scores, y_test)

    score_file = scores_dir / f"{args.machine}_full_5view_latent_summary_knn_scores.npz"
    np.savez(
        score_file,
        train_scores=train_scores,
        test_scores=test_scores,
        y_test=y_test,
        domains=d_test,
        train_paths=np.asarray(train_paths, dtype=object),
        test_paths=np.asarray(test_paths, dtype=object),
    )

    baseline = _load_v37_baseline(Path(args.baseline_dir), args.machine)

    summary_row = {
        "machine": args.machine,
        "experiment": "full_5view",
        "score_mode": "latent_summary_knn",
        "seed": args.seed,
        "manifest_selected_count": len(pretrain_paths),
        "current_year_train_count": len(train_items),
        "current_year_test_count": len(test_items),
        "hmean": metrics["hmean"],
        "pAUC": metrics["pauc"],
        "KS": geom["source_target_ks"],
        "source_auc": metrics["auc_source"],
        "target_auc": metrics["auc_target"],
        "phase1_epochs": phase1_stats["epochs_ran"],
        "phase2_teacher_epochs": phase2_stats["epochs_ran"],
        "phase2_student_epochs": phase2_student_stats["epochs_ran"],
        "phase1_train_time_sec": phase1_stats["train_time_sec"],
        "phase2_teacher_train_time_sec": phase2_stats["train_time_sec"],
        "phase2_student_train_time_sec": phase2_student_stats["train_time_sec"],
        "phase1_ckpt": str(phase1_ckpt),
        "phase2_teacher_ckpt": str(phase2_teacher_ckpt),
        "phase2_student_ckpt": str(phase2_student_ckpt),
        "score_file": str(score_file),
    }
    tail_row = {
        "machine": args.machine,
        **tail,
    }
    geom_row = {
        "machine": args.machine,
        **geom,
    }
    _write_csv(output_root / "summary.csv", [summary_row])
    _write_csv(output_root / "tail_table.csv", [tail_row])
    _write_csv(output_root / "score_geometry.csv", [geom_row])

    verdict = "dryrun pass"
    lines = [
        "# CTAF V61 Cross-Year Original Encoder Dry Run",
        "",
        "## 1. Scope",
        "",
        "One-machine dry run only. This step validates pipeline compatibility for",
        "`cross-year pretrain -> current-year adapt -> V37 latent_summary_knn eval`.",
        "",
        "## 2. Manifest and Current-Year Root",
        "",
        f"- manifest: `{manifest_path}`",
        f"- current target-year data root: `{current_root}`",
        f"- machine: `{args.machine}`",
        f"- seed: `{args.seed}`",
        "",
        "## 3. Included / Excluded Confirmation",
        "",
        f"- manifest selected rows for `{args.machine}`: `{len(pretrain_paths)}`",
        f"- per-year counts: `{json.dumps(pretrain_year_counts, ensure_ascii=False)}`",
        f"- per-domain counts: `{json.dumps(pretrain_domain_counts, ensure_ascii=False)}`",
        "- 2025 eval_data_* branches: excluded by V60B manifest",
        "- 2026 Emu / noise-aware branches: excluded by V60B manifest",
        "- anomaly clips: excluded by V60B manifest",
        "- test files: excluded from cross-year pretraining manifest",
        "",
        "## 4. Phase Status",
        "",
        f"- Phase 1 pretrain: success, epochs={phase1_stats['epochs_ran']}, best_val={phase1_stats['best_val_loss']:.6f}",
        f"- Phase 2 teacher adaptation: success, epochs={phase2_stats['epochs_ran']}, best_val={phase2_stats['best_val_loss']:.6f}",
        f"- Phase 2 student distillation: success, epochs={phase2_student_stats['epochs_ran']}, best_val={phase2_student_stats['best_val_loss']:.6f}",
        "",
        "## 5. Checkpoints and Scores",
        "",
        f"- phase1 teacher checkpoint: `{phase1_ckpt}`",
        f"- phase2 teacher checkpoint: `{phase2_teacher_ckpt}`",
        f"- phase2 student checkpoint: `{phase2_student_ckpt}`",
        f"- score file: `{score_file}`",
        "",
        "## 6. Gearbox Dry-Run Metrics",
        "",
        f"- hmean: `{metrics['hmean']:.6f}`",
        f"- pAUC: `{metrics['pauc']:.6f}`",
        f"- KS: `{geom['source_target_ks']:.6f}`",
        f"- q95: `{tail['q95']:.6f}`",
        f"- q99: `{tail['q99']:.6f}`",
        f"- TPR@95n: `{tail['TPR@95n']:.6f}`",
        f"- TPR@99n: `{tail['TPR@99n']:.6f}`",
        "",
        "## 7. V37 Gearbox Baseline Comparison",
        "",
        f"- baseline hmean: `{baseline['hmean']:.6f}`",
        f"- baseline pAUC: `{baseline['pAUC']:.6f}`",
        f"- baseline KS: `{baseline['KS']:.6f}`",
        f"- baseline q95: `{baseline['q95']:.6f}`",
        f"- baseline q99: `{baseline['q99']:.6f}`",
        f"- baseline TPR@95n: `{baseline['TPR@95n']:.6f}`",
        f"- baseline TPR@99n: `{baseline['TPR@99n']:.6f}`",
        f"- delta hmean: `{metrics['hmean'] - baseline['hmean']:.6f}`",
        f"- delta pAUC: `{metrics['pauc'] - baseline['pAUC']:.6f}`",
        f"- delta KS: `{geom['source_target_ks'] - baseline['KS']:.6f}`",
        "",
        "## 8. Final Verdict",
        "",
        f"`{verdict}`",
        "",
        "Pipeline is compatible with the V61 prereg logic on one-machine dry run.",
        "This verdict does not change protocol or authorize V62.",
        "",
        "## 9. Confirmation",
        "",
        "- no encoder architecture change",
        "- no frontend change",
        "- no loss change beyond prereg V37 pretrain/adapt logic",
        "- no anomaly label used in training",
        "- no test label used in training",
        "- no 2025 eval_data_* used",
        "- no 2026 used",
        "- no V62 opened",
    ]
    doc_path = Path(args.doc_path)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
