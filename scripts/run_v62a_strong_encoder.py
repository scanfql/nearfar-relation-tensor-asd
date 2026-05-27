#!/usr/bin/env python
"""V62A strong encoder utility: shape_check, dryrun_train, full_train."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.evaluation.scorer import compute_latent_knn_scores, compute_latent_summary_vectors
from src.features.logmel import extract_logmel, load_audio
from src.features.tensor_builder import TensorNormalizer, build_tensor
from src.models.conv_ae import Full5ViewConformerAutoencoder
from src.training.train_ae import train_autoencoder, train_teacher_student_local_distillation
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="V62A strong encoder utility")
    p.add_argument("--mode", required=True, choices=["shape_check", "dryrun_train", "full_train"])
    p.add_argument("--manifest_path", default="results/v60b_cross_year_normal_manifest/manifest_v61_clean.csv")
    p.add_argument("--data_root", default="/data1/test/LAES_2025/data/dcase")
    p.add_argument("--machine", default="gearbox")
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", required=True)
    p.add_argument("--baseline_dir", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--conformer_conv_kernel", type=int, default=15)
    p.add_argument("--input_freq", type=int, default=128)
    p.add_argument("--input_time", type=int, default=313)
    p.add_argument("--shape_batch_size", type=int, default=2)
    p.add_argument("--mini_train_step", action="store_true")
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


def _build_model(args) -> Full5ViewConformerAutoencoder:
    return Full5ViewConformerAutoencoder(
        in_channels=5,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        conv_kernel=args.conformer_conv_kernel,
        stem_channels=16,
        per_view_channels=32,
        dropout_p=0.0,
        use_batch_norm=True,
    )


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
    return np.stack(train_norm, axis=0), np.stack(eval_norm, axis=0)


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


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_manifest_paths(manifest_path: str, machine: str) -> list[str]:
    with Path(manifest_path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [
        row["absolute_path"]
        for row in rows
        if row.get("machine_canonical") == machine and _truthy(row.get("usable_for_v61", ""))
    ]
    return selected


def _shape_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    model = _build_model(args)
    params = sum(p.numel() for p in model.parameters())

    x = torch.randn(args.shape_batch_size, 5, args.input_freq, args.input_time)
    with torch.no_grad():
        latent = model.encode(x)
        recon = model(x)
        mean = latent.mean(dim=(2, 3))
        std = latent.std(dim=(2, 3), unbiased=False)
        max_vals = latent.amax(dim=(2, 3))
        latent_summary = torch.cat([mean, std, max_vals], dim=1)

    mini_train_loss = None
    if args.mini_train_step:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        pred = model(x)
        loss = ((pred - x) ** 2).mean()
        loss.backward()
        opt.step()
        mini_train_loss = float(loss.item())

    result = {
        "mode": "shape_check",
        "parameter_count": int(params),
        "input_shape": list(x.shape),
        "latent_map_shape": list(latent.shape),
        "output_shape": list(recon.shape),
        "latent_summary_shape": list(latent_summary.shape),
        "latent_summary_dim": int(latent_summary.shape[1]),
        "mini_train_loss": mini_train_loss,
        "target_param_range_ok": bool(1_000_000 <= params <= 5_000_000),
    }
    (out_root / "shape_check_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _run_machine(args, machine: str):
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    models_dir = output_root / "models"
    scores_dir = output_root / "scores"
    models_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    pretrain_paths = _read_manifest_paths(args.manifest_path, machine)
    if not pretrain_paths:
        raise RuntimeError(f"No manifest rows for machine={machine}")

    pretrain_tensors = _build_tensors(pretrain_paths)
    phase1_arr, _ = _prepare_arrays(pretrain_tensors, pretrain_tensors[:1])

    teacher = _build_model(args)
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
    phase1_ckpt = models_dir / f"{machine}_phase1_teacher_seed{args.seed}.pt"
    torch.save({"state_dict": teacher.state_dict(), "train_stats": phase1_stats}, phase1_ckpt)

    all_items = scan_dataset(Path(args.data_root))
    machine_items = filter_items(all_items, machine)
    train_items, test_items = split_train_test(machine_items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Invalid current-year split for {machine}: train={len(train_items)} test={len(test_items)}")

    train_paths = [str(x.path) for x in train_items]
    test_paths = [str(x.path) for x in test_items]
    train_tensors = _build_tensors(train_paths)
    test_tensors = _build_tensors(test_paths)
    train_arr, test_arr = _prepare_arrays(train_tensors, test_tensors)
    y_test = np.asarray([x.label for x in test_items], dtype=np.int64)
    d_test = np.asarray([x.domain for x in test_items])

    phase2_teacher_stats = train_autoencoder(
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
    phase2_teacher_ckpt = models_dir / f"{machine}_phase2_teacher_seed{args.seed}.pt"
    torch.save({"state_dict": teacher.state_dict(), "train_stats": phase2_teacher_stats}, phase2_teacher_ckpt)

    student = _build_model(args)
    phase2_student_stats = train_teacher_student_local_distillation(
        teacher,
        student,
        train_arr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    phase2_student_ckpt = models_dir / f"{machine}_phase2_student_seed{args.seed}.pt"
    torch.save({"state_dict": student.state_dict(), "train_stats": phase2_student_stats}, phase2_student_ckpt)

    memory_lat = compute_latent_summary_vectors(teacher, train_arr, batch_size=args.batch_size, device=args.device)
    test_lat = compute_latent_summary_vectors(teacher, test_arr, batch_size=args.batch_size, device=args.device)
    train_scores = compute_latent_knn_scores(memory_lat, memory_lat, k=args.latent_knn_k)
    test_scores = compute_latent_knn_scores(memory_lat, test_lat, k=args.latent_knn_k)
    metrics = compute_metrics(y_test, test_scores, d_test)
    geom = compute_score_geometry(test_scores, y_test, d_test)
    tail = _tail_stats(test_scores, y_test)

    score_file = scores_dir / f"{machine}_full_5view_latent_summary_knn_scores.npz"
    np.savez(
        score_file,
        train_scores=train_scores,
        test_scores=test_scores,
        y_test=y_test,
        domains=d_test,
        train_paths=np.asarray(train_paths, dtype=object),
        test_paths=np.asarray(test_paths, dtype=object),
    )

    summary_row = {
        "machine": machine,
        "seed": args.seed,
        "manifest_selected_count": len(pretrain_paths),
        "current_year_train_count": len(train_items),
        "current_year_test_count": len(test_items),
        "hmean": metrics["hmean"],
        "pAUC": metrics["pauc"],
        "KS": geom["source_target_ks"],
        "phase1_ckpt": str(phase1_ckpt),
        "phase2_teacher_ckpt": str(phase2_teacher_ckpt),
        "phase2_student_ckpt": str(phase2_student_ckpt),
        "score_file": str(score_file),
    }
    tail_row = {"machine": machine, **tail}
    geom_row = {"machine": machine, **geom}
    return summary_row, tail_row, geom_row


def _run_training_mode(args, machines: list[str]):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    tail_rows = []
    geom_rows = []
    for machine in machines:
        s, t, g = _run_machine(args, machine)
        summary_rows.append(s)
        tail_rows.append(t)
        geom_rows.append(g)
    _write_csv(out_root / "summary.csv", summary_rows)
    _write_csv(out_root / "tail_table.csv", tail_rows)
    _write_csv(out_root / "score_geometry.csv", geom_rows)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.mode == "shape_check":
        _shape_check(args)
        return
    if args.mode == "dryrun_train":
        _run_training_mode(args, [args.machine])
        return
    if args.mode == "full_train":
        machines = [m.strip() for m in args.machines.split(",") if m.strip()]
        _run_training_mode(args, machines)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
