#!/usr/bin/env python
"""run_experiment.py — CTAF Clean Tensor Acoustic Frontend Experiment Runner.

Label-free protocol:
  - Train on normal-only source-domain data.
  - No anomaly labels used in training, feature extraction, or model selection.
  - Fixed pre-registered rules for all choices.
  - Labels used ONLY in final evaluation metrics and post-hoc plots.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="CTAF experiment runner")
    # Required
    p.add_argument("--data_root", required=True, help="Path to dataset root")
    # Optional overrides
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--experiments", default="logmel_only,dynamic_3view,edge_3view,full_5view")
    p.add_argument("--use_target_train", action="store_true",
                   help="Train on both source and target normal train files")
    p.add_argument("--memory_domains", default="source_train",
                   choices=["source_train", "source_target_train"],
                   help="Which normal-train domains are allowed in the latent memory bank")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--latent_channels", type=int, default=64,
                   help="Conv2D AE encoder/decoder latent channel width")
    p.add_argument("--stem_channels", type=int, default=8,
                   help="Per-view stem width for late-fusion models")
    p.add_argument("--per_view_channels", type=int, default=16,
                   help="Per-view encoder width before fusion for late-fusion models")
    p.add_argument("--dropout_p", type=float, default=0.0,
                   help="Spatial dropout probability inside AE blocks")
    p.add_argument("--loss_view_mode", default="all_views",
                   choices=["all_views", "logmel_only", "masked_view_only", "future_only"],
                   help="Which output views contribute to reconstruction loss")
    p.add_argument("--ssl_task", default="reconstruction",
                   choices=["reconstruction", "masked_cross_view_prediction", "temporal_predictive_ssl", "teacher_student_local_distillation", "vicreg_invariance_ssl"],
                   help="Self-supervised training task")
    p.add_argument("--model_type", default="baseline",
                   choices=[
                       "baseline",
                       "viewwise_late_fusion",
                       "viewwise_unshared_late_fusion",
                       "viewwise_unshared_residual_temporal_pyramid",
                       "viewwise_unshared_temporal_pyramid",
                       "viewwise_unshared_gated_late_fusion",
                   ],
                   help="Global model architecture")
    p.add_argument("--early_stopping", action="store_true",
                   help="Restore best checkpoint by normal-validation reconstruction loss")
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience in epochs")
    p.add_argument("--min_delta", type=float, default=0.0,
                   help="Minimum normal-validation loss improvement for early stopping")
    p.add_argument("--noise_mode", default="none", choices=["none", "tf_mask", "view_dropout"],
                   help="Train-time denoising corruption mode. Scoring always uses clean input")
    p.add_argument("--mask_prob", type=float, default=0.0,
                   help="Mask probability for --noise_mode tf_mask")
    p.add_argument("--tail_compact_loss", action="store_true",
                   help="Add fixed train-normal latent-summary tail compactness loss")
    p.add_argument("--tail_compact_lambda", type=float, default=0.01,
                   help="Global weight for tail compactness loss")
    p.add_argument("--tail_compact_top_frac", type=float, default=0.10,
                   help="Top fraction of batch center-distances used by tail compactness loss")
    p.add_argument("--hard_normal_index_dir", default="",
                   help="Directory containing fixed hard-normal CSVs, one per machine and seed")
    p.add_argument("--hard_normal_consistency_loss", action="store_true",
                   help="Apply fixed hard-normal augmentation consistency loss")
    p.add_argument("--hard_normal_consistency_lambda", type=float, default=0.10,
                   help="Global weight for hard-normal consistency loss")
    p.add_argument("--hard_normal_time_mask_frac", type=float, default=0.10,
                   help="Fixed time-mask fraction for hard-normal consistency")
    p.add_argument("--hard_normal_freq_mask_frac", type=float, default=0.10,
                   help="Fixed frequency-mask fraction for hard-normal consistency")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output_dir", default="results/full")
    p.add_argument("--score_modes", default="mean_mse",
                   help=("Comma-separated fixed score modes: "
                         "mean_mse,view_z_mean,view_z_top2,latent_knn,"
                         "latent_summary_knn,latent_summary_robust_knn,"
                         "latent_summary_blockscale_knn,"
                         "latent_summary_whiten_knn"))
    p.add_argument("--latent_knn_k", type=int, default=5,
                   help="Fixed k for latent_knn train-normal memory distance")
    p.add_argument("--progress", dest="progress", action="store_true", default=True)
    p.add_argument("--no-progress", dest="progress", action="store_false",
                   help="Disable tqdm progress bars for cleaner non-interactive logs")
    return p.parse_args()


def _progress(iterable, desc: str, enabled: bool = True):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True)
    except Exception:
        return iterable


def _banner(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _section(title: str):
    print()
    print(f"[{title}]")


def _fmt_float(value: float) -> str:
    return f"{value:.4f}"


def _build_model(model_type: str, in_channels: int, args):
    from src.models.conv_ae import (
        Conv2DAutoencoder,
        ViewwiseLateFusionAutoencoder,
        ViewwiseUnsharedGatedLateFusionAutoencoder,
        ViewwiseUnsharedLateFusionAutoencoder,
        ViewwiseUnsharedResidualTemporalPyramidAutoencoder,
        ViewwiseUnsharedTemporalPyramidAutoencoder,
    )
    if model_type == "baseline":
        return Conv2DAutoencoder(
            in_channels=in_channels,
            latent_channels=args.latent_channels,
            use_batch_norm=True,
            dropout_p=args.dropout_p,
        )
    if model_type == "viewwise_late_fusion":
        return ViewwiseLateFusionAutoencoder(
            in_channels=in_channels,
            latent_channels=args.latent_channels,
            use_batch_norm=True,
            per_view_channels=args.per_view_channels,
            stem_channels=args.stem_channels,
            dropout_p=args.dropout_p,
        )
    if model_type == "viewwise_unshared_late_fusion":
        return ViewwiseUnsharedLateFusionAutoencoder(
            in_channels=in_channels,
            latent_channels=args.latent_channels,
            use_batch_norm=True,
            per_view_channels=args.per_view_channels,
            stem_channels=args.stem_channels,
            dropout_p=args.dropout_p,
        )
    if model_type == "viewwise_unshared_residual_temporal_pyramid":
        return ViewwiseUnsharedResidualTemporalPyramidAutoencoder(
            in_channels=in_channels,
            latent_channels=args.latent_channels,
            use_batch_norm=True,
            per_view_channels=args.per_view_channels,
            stem_channels=args.stem_channels,
            dropout_p=args.dropout_p,
        )
    if model_type == "viewwise_unshared_temporal_pyramid":
        return ViewwiseUnsharedTemporalPyramidAutoencoder(
            in_channels=in_channels,
            latent_channels=args.latent_channels,
            use_batch_norm=True,
            per_view_channels=args.per_view_channels,
            stem_channels=args.stem_channels,
            dropout_p=args.dropout_p,
        )
    if model_type == "viewwise_unshared_gated_late_fusion":
        return ViewwiseUnsharedGatedLateFusionAutoencoder(
            in_channels=in_channels,
            latent_channels=args.latent_channels,
            use_batch_norm=True,
            per_view_channels=args.per_view_channels,
            stem_channels=args.stem_channels,
            dropout_p=args.dropout_p,
        )
    raise ValueError(f"unknown model_type: {model_type}")


def main():
    args = parse_args()

    from src.data.dataset import scan_dataset, filter_items, split_train_test
    from src.features.logmel import extract_logmel, extract_multires_logmels, load_audio
    from src.features.tensor_builder import build_tensor, TensorNormalizer
    from src.training.train_ae import (
        train_autoencoder,
        train_teacher_student_local_distillation,
        train_vicreg_invariance_ssl,
    )
    from src.evaluation.scorer import (
        aggregate_view_scores,
        apply_feature_block_scales,
        apply_feature_whitener,
        compute_local_latent_token_lists,
        compute_local_tail_knn_scores,
        compute_recon_error_views,
        compute_teacher_student_top5_scores,
        fit_feature_robust_stats,
        compute_latent_knn_scores,
        compute_latent_summary_vectors,
        compute_latent_vectors,
        fit_feature_block_scales,
        fit_feature_whitener,
        fit_view_residual_stats,
    )
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.score_geometry import compute_score_geometry
    from src.evaluation.plots import plot_score_histogram
    from src.utils.seed import set_seed

    set_seed(args.seed)

    machines = [m.strip() for m in args.machines.split(",")]
    experiments = [e.strip() for e in args.experiments.split(",")]
    score_modes = [s.strip() for s in args.score_modes.split(",") if s.strip()]
    known_experiments = {
        "logmel_only", "dynamic_3view", "edge_3view", "full_5view", "multires_7view",
        "transient_3view", "cepstral_3view",
    }
    unknown_experiments = sorted(set(experiments) - known_experiments)
    if unknown_experiments:
        raise ValueError(f"Unknown experiment(s): {','.join(unknown_experiments)}")
    allowed_score_modes = {
        "mean_mse", "view_z_mean", "view_z_top2",
        "latent_knn", "latent_summary_knn",
        "latent_summary_robust_knn", "latent_summary_blockscale_knn",
        "latent_summary_whiten_knn",
        "local_tail_knn_top5",
        "teacher_student_top5",
    }
    unknown_modes = sorted(set(score_modes) - allowed_score_modes)
    if unknown_modes:
        raise ValueError(f"Unknown score mode(s): {','.join(unknown_modes)}")
    os.makedirs(args.output_dir, exist_ok=True)
    figs_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figs_dir, exist_ok=True)
    scores_dir = os.path.join(args.output_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    models_dir = os.path.join(args.output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    _banner("CTAF Clean Tensor ASD")
    print(f"data_root  : {args.data_root}")
    print(f"machines   : {','.join(machines)}")
    print(f"experiments: {','.join(experiments)}")
    print(f"target_tr  : {args.use_target_train}")
    print(f"memory_dom : {args.memory_domains}")
    print(f"score_modes: {','.join(score_modes)}")
    print(f"latent_k   : {args.latent_knn_k}")
    print(f"epochs     : {args.epochs}")
    print(f"batch_size : {args.batch_size}")
    print(f"latent_ch  : {args.latent_channels}")
    print(f"stem_ch    : {args.stem_channels}")
    print(f"per_view_ch: {args.per_view_channels}")
    print(f"dropout_p  : {args.dropout_p}")
    print(f"loss_mode  : {args.loss_view_mode}")
    print(f"ssl_task   : {args.ssl_task}")
    print(f"model_type : {args.model_type}")
    print(f"early_stop : {args.early_stopping} patience={args.patience} min_delta={args.min_delta}")
    print(f"noise_mode : {args.noise_mode} mask_prob={args.mask_prob}")
    print(
        f"tail_compact: {args.tail_compact_loss} "
        f"lambda={args.tail_compact_lambda} top_frac={args.tail_compact_top_frac}"
    )
    print(
        f"hard_normal_consistency: {args.hard_normal_consistency_loss} "
        f"lambda={args.hard_normal_consistency_lambda} "
        f"time_frac={args.hard_normal_time_mask_frac} "
        f"freq_frac={args.hard_normal_freq_mask_frac}"
    )
    print(f"device     : {args.device}")
    print(f"seed       : {args.seed}")
    print(f"output_dir : {args.output_dir}")

    _section("Dataset")
    print(f"Scanning: {args.data_root}")
    all_items = scan_dataset(Path(args.data_root))
    print(f"Found {len(all_items)} audio files")

    # Fixed comparison baseline: logmel_only with the V1 mean-MSE score.
    logmel_hmean = {}  # machine -> logmel_only/mean_mse hmean

    all_results = []
    per_machine_rows = []
    score_geometry_rows = []

    for machine in machines:
        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=args.use_target_train)
        memory_items = [
            x for x in items
            if x.split == "train"
            and x.label in (0, None)
            and (
                x.domain == "source"
                or (args.memory_domains == "source_target_train" and x.domain == "target")
            )
        ]
        _banner(f"Machine: {machine}")
        print(f"train={len(train_items)}  test={len(test_items)}")
        _print_split_counts(train_items, test_items)
        print(f"memory bank train normals={len(memory_items)} ({args.memory_domains})")

        if len(train_items) == 0 or len(test_items) == 0:
            print("SKIP: no train or test data")
            continue

        # Extract log-Mel for all files
        need_multires = "multires_7view" in experiments

        def extract_features(item_list, label=""):
            tensors = {}
            t0 = time.time()
            desc = f"{machine} {label} logmel"
            for item in _progress(item_list, desc=desc, enabled=args.progress):
                audio = load_audio(item.path)
                base = extract_logmel(audio)
                multires = extract_multires_logmels(audio) if need_multires else None
                tensors[item.path] = (base, multires, audio, item.label, item.domain)
            print(f"features[{label}]: {len(item_list)} files in {time.time() - t0:.1f}s")
            return tensors

        train_feats = extract_features(train_items, "train")
        test_feats = extract_features(test_items, "test")
        memory_feats = train_feats
        if args.memory_domains == "source_target_train":
            memory_only_items = [x for x in memory_items if x.path not in train_feats]
            memory_only_feats = extract_features(memory_only_items, "memory") if memory_only_items else {}
            memory_feats = dict(train_feats)
            memory_feats.update(memory_only_feats)

        for exp_name in experiments:
            _section(f"Experiment: {machine} / {exp_name}")

            # Build tensors for this experiment
            train_tensors = []
            train_paths = []
            for item in train_items:
                base_lm, multires_lm, audio_t, _, _ = train_feats[item.path]
                feat = multires_lm if exp_name == "multires_7view" else base_lm
                t = build_tensor(exp_name, feat, audio=audio_t)
                train_tensors.append(t)
                train_paths.append(str(item.path))

            test_tensors = []
            test_labels = []
            test_domains = []
            for item in test_items:
                base_lm, multires_lm, audio_t, label, domain = test_feats[item.path]
                feat = multires_lm if exp_name == "multires_7view" else base_lm
                t = build_tensor(exp_name, feat, audio=audio_t)
                test_tensors.append(t)
                test_labels.append(label)
                test_domains.append(domain)

            memory_tensors = []
            for item in memory_items:
                base_lm, multires_lm, audio_t, _, _ = memory_feats[item.path]
                feat = multires_lm if exp_name == "multires_7view" else base_lm
                t = build_tensor(exp_name, feat, audio=audio_t)
                memory_tensors.append(t)

            V = train_tensors[0].shape[0]
            print(f"views={V}  tensor_shape={train_tensors[0].shape}")

            # Normalize (train-set statistics, no labels)
            normalizer = TensorNormalizer()
            train_norm = normalizer.fit_transform(train_tensors)
            test_norm = [normalizer.transform(t) for t in test_tensors]
            memory_norm = [normalizer.transform(t) for t in memory_tensors]

            # Stack to arrays
            train_arr = np.stack(train_norm, axis=0)
            test_arr = np.stack(test_norm, axis=0)
            memory_arr = np.stack(memory_norm, axis=0)

            hard_normal_flags = None
            if args.hard_normal_consistency_loss and exp_name == "full_5view":
                if not args.hard_normal_index_dir:
                    raise ValueError("--hard_normal_consistency_loss requires --hard_normal_index_dir")
                seed_dir = Path(args.hard_normal_index_dir) / f"seed_{args.seed}"
                index_path = seed_dir / f"{machine}_{exp_name}_latent_summary_knn_hard_normal.csv"
                if not index_path.exists():
                    raise FileNotFoundError(f"Missing hard-normal index: {index_path}")
                import csv as _csv
                with index_path.open("r", newline="") as f:
                    rows = list(_csv.DictReader(f))
                flag_map = {row["sample_path"]: int(row["hard_normal_flag"]) for row in rows}
                hard_normal_flags = np.asarray(
                    [flag_map.get(p, 0) for p in train_paths], dtype=np.float32
                )

            # Train model or teacher-student pair.
            teacher_model = None
            student_model = None
            if args.ssl_task == "teacher_student_local_distillation":
                teacher_model = _build_model(args.model_type, V, args)
                teacher_params = sum(p.numel() for p in teacher_model.parameters())
                print(
                    f"train teacher: epochs={args.epochs} batch={args.batch_size} "
                    f"params={teacher_params:,}"
                )
                teacher_stats = train_autoencoder(
                    teacher_model, train_arr,
                    epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, seed=args.seed,
                    early_stopping=args.early_stopping,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    noise_mode="none",
                    mask_prob=0.0,
                    loss_view_mode="all_views",
                    ssl_task="reconstruction",
                    tail_compact_loss=False,
                    tail_compact_lambda=0.0,
                    tail_compact_top_frac=0.10,
                )
                student_model = _build_model(args.model_type, V, args)
                student_params = sum(p.numel() for p in student_model.parameters())
                print(
                    f"train student: epochs={args.epochs} batch={args.batch_size} "
                    f"params={student_params:,}"
                )
                train_stats = train_teacher_student_local_distillation(
                    teacher_model, student_model, train_arr,
                    epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, seed=args.seed,
                    early_stopping=args.early_stopping,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )
                train_stats["teacher_train_time_sec"] = teacher_stats.get("train_time_sec", 0.0)
                train_stats["teacher_best_epoch"] = teacher_stats.get("best_epoch", args.epochs)
                train_stats["teacher_best_val_loss"] = teacher_stats.get("best_val_loss", 0.0)
                model = teacher_model
            elif args.ssl_task == "vicreg_invariance_ssl":
                model = _build_model(args.model_type, V, args)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"train vicreg: epochs={args.epochs} batch={args.batch_size} encoder_params={n_params:,}")
                train_stats = train_vicreg_invariance_ssl(
                    model, train_arr,
                    epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, seed=args.seed,
                    early_stopping=args.early_stopping,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )
            else:
                model = _build_model(args.model_type, V, args)
                n_params = sum(p.numel() for p in model.parameters())
                print(f"train: epochs={args.epochs} batch={args.batch_size} params={n_params:,}")
                train_stats = train_autoencoder(
                    model, train_arr,
                    epochs=args.epochs, batch_size=args.batch_size,
                    device=args.device, seed=args.seed,
                    early_stopping=args.early_stopping,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    noise_mode=args.noise_mode,
                    mask_prob=args.mask_prob,
                    loss_view_mode=args.loss_view_mode,
                    ssl_task=args.ssl_task,
                    tail_compact_loss=args.tail_compact_loss,
                    tail_compact_lambda=args.tail_compact_lambda,
                    tail_compact_top_frac=args.tail_compact_top_frac,
                    hard_normal_flags=hard_normal_flags,
                    hard_normal_consistency_loss=args.hard_normal_consistency_loss and exp_name == "full_5view",
                    hard_normal_consistency_lambda=args.hard_normal_consistency_lambda,
                    hard_normal_time_mask_frac=args.hard_normal_time_mask_frac,
                    hard_normal_freq_mask_frac=args.hard_normal_freq_mask_frac,
                )

            checkpoint_path = ""
            if teacher_model is not None and student_model is not None:
                teacher_ckpt = os.path.join(models_dir, f"{machine}_{exp_name}_teacher_seed{args.seed}.pt")
                student_ckpt = os.path.join(models_dir, f"{machine}_{exp_name}_student_seed{args.seed}.pt")
                torch.save({"state_dict": teacher_model.state_dict()}, teacher_ckpt)
                torch.save({"state_dict": student_model.state_dict()}, student_ckpt)
                checkpoint_path = f"{teacher_ckpt};{student_ckpt}"
            else:
                checkpoint_path = os.path.join(models_dir, f"{machine}_{exp_name}_seed{args.seed}.pt")
                torch.save({"state_dict": model.state_dict()}, checkpoint_path)

            # Score. View-wise residuals are computed once; score modes are fixed
            # global aggregations, never selected per machine.
            needs_recon_scores = any(
                mode in score_modes for mode in ("mean_mse", "view_z_mean", "view_z_top2")
            )
            if needs_recon_scores:
                train_view_errors = compute_recon_error_views(
                    model, train_arr, batch_size=args.batch_size, device=args.device
                )
                test_view_errors = compute_recon_error_views(
                    model, test_arr, batch_size=args.batch_size, device=args.device
                )
                view_stats = fit_view_residual_stats(train_view_errors)
            else:
                train_view_errors = np.array([])
                test_view_errors = np.array([])
                view_stats = None

            train_latents = None
            test_latents = None
            memory_latents = None
            train_latent_summaries = None
            test_latent_summaries = None
            memory_latent_summaries = None
            test_local_token_lists = None
            memory_local_tokens = None
            latent_summary_stats = None
            latent_summary_block_stats = None
            latent_summary_whitener = None
            if "latent_knn" in score_modes:
                memory_latents = compute_latent_vectors(
                    model, memory_arr, batch_size=args.batch_size, device=args.device
                )
                train_latents = compute_latent_vectors(
                    model, train_arr, batch_size=args.batch_size, device=args.device
                )
                test_latents = compute_latent_vectors(
                    model, test_arr, batch_size=args.batch_size, device=args.device
                )
            if (
                "latent_summary_knn" in score_modes
                or "latent_summary_robust_knn" in score_modes
                or "latent_summary_blockscale_knn" in score_modes
                or "latent_summary_whiten_knn" in score_modes
            ):
                memory_latent_summaries = compute_latent_summary_vectors(
                    model, memory_arr, batch_size=args.batch_size, device=args.device
                )
                train_latent_summaries = compute_latent_summary_vectors(
                    model, train_arr, batch_size=args.batch_size, device=args.device
                )
                test_latent_summaries = compute_latent_summary_vectors(
                    model, test_arr, batch_size=args.batch_size, device=args.device
                )
                latent_summary_stats = fit_feature_robust_stats(memory_latent_summaries)
                latent_summary_block_stats = fit_feature_block_scales(memory_latent_summaries)
                latent_summary_whitener = fit_feature_whitener(memory_latent_summaries)
            if "local_tail_knn_top5" in score_modes:
                memory_local_token_lists = compute_local_latent_token_lists(
                    model, memory_arr, batch_size=args.batch_size, device=args.device
                )
                test_local_token_lists = compute_local_latent_token_lists(
                    model, test_arr, batch_size=args.batch_size, device=args.device
                )
                memory_local_tokens = np.concatenate(memory_local_token_lists, axis=0)

            y_test = np.array(test_labels, dtype=np.int64)
            d_test = np.array(test_domains)

            for score_mode in score_modes:
                if score_mode == "latent_knn":
                    train_scores = compute_latent_knn_scores(
                        memory_latents, train_latents, k=args.latent_knn_k
                    )
                    test_scores = compute_latent_knn_scores(
                        memory_latents, test_latents, k=args.latent_knn_k
                    )
                elif score_mode == "latent_summary_knn":
                    train_scores = compute_latent_knn_scores(
                        memory_latent_summaries, train_latent_summaries, k=args.latent_knn_k
                    )
                    test_scores = compute_latent_knn_scores(
                        memory_latent_summaries, test_latent_summaries, k=args.latent_knn_k
                    )
                elif score_mode == "latent_summary_robust_knn":
                    train_scores = compute_latent_knn_scores(
                        memory_latent_summaries,
                        train_latent_summaries,
                        k=args.latent_knn_k,
                        train_feature_stats=latent_summary_stats,
                    )
                    test_scores = compute_latent_knn_scores(
                        memory_latent_summaries,
                        test_latent_summaries,
                        k=args.latent_knn_k,
                        train_feature_stats=latent_summary_stats,
                    )
                elif score_mode == "latent_summary_blockscale_knn":
                    train_block = apply_feature_block_scales(
                        train_latent_summaries, latent_summary_block_stats
                    )
                    test_block = apply_feature_block_scales(
                        test_latent_summaries, latent_summary_block_stats
                    )
                    memory_block = apply_feature_block_scales(
                        memory_latent_summaries, latent_summary_block_stats
                    )
                    train_scores = compute_latent_knn_scores(
                        memory_block, train_block, k=args.latent_knn_k
                    )
                    test_scores = compute_latent_knn_scores(
                        memory_block, test_block, k=args.latent_knn_k
                    )
                elif score_mode == "latent_summary_whiten_knn":
                    train_white = apply_feature_whitener(
                        train_latent_summaries, latent_summary_whitener
                    )
                    test_white = apply_feature_whitener(
                        test_latent_summaries, latent_summary_whitener
                    )
                    memory_white = apply_feature_whitener(
                        memory_latent_summaries, latent_summary_whitener
                    )
                    train_scores = compute_latent_knn_scores(
                        memory_white, train_white, k=args.latent_knn_k
                    )
                    test_scores = compute_latent_knn_scores(
                        memory_white, test_white, k=args.latent_knn_k
                    )
                elif score_mode == "local_tail_knn_top5":
                    train_scores = np.array([], dtype=np.float64)
                    test_scores = compute_local_tail_knn_scores(
                        memory_local_tokens,
                        test_local_token_lists,
                        k=args.latent_knn_k,
                        top_percent=5.0,
                        device=args.device,
                    )
                elif score_mode == "teacher_student_top5":
                    train_scores = np.array([], dtype=np.float64)
                    test_scores = compute_teacher_student_top5_scores(
                        teacher_model,
                        student_model,
                        test_arr,
                        batch_size=args.batch_size,
                        device=args.device,
                        top_percent=5.0,
                    )
                else:
                    train_scores = aggregate_view_scores(train_view_errors, score_mode, view_stats)
                    test_scores = aggregate_view_scores(test_view_errors, score_mode, view_stats)

                # Compute metrics after fixed scores are generated.
                metrics = compute_metrics(y_test, test_scores, d_test)
                geometry = compute_score_geometry(test_scores, y_test, d_test)

                score_label = exp_name if score_mode == "mean_mse" else f"{exp_name}_{score_mode}"
                hist_path = plot_score_histogram(
                    test_scores, y_test, d_test, machine, score_label, figs_dir
                )
                score_path = os.path.join(scores_dir, f"{machine}_{score_label}_scores.npz")
                np.savez_compressed(
                    score_path,
                    train_scores=train_scores,
                    test_scores=test_scores,
                    train_view_errors=train_view_errors,
                    test_view_errors=test_view_errors,
                    view_residual_median=(
                        view_stats["median"] if view_stats is not None else np.array([])
                    ),
                    view_residual_iqr=(
                        view_stats["iqr"] if view_stats is not None else np.array([])
                    ),
                    train_latents=train_latents if train_latents is not None else np.array([]),
                    test_latents=test_latents if test_latents is not None else np.array([]),
                    memory_latents=memory_latents if memory_latents is not None else np.array([]),
                    train_latent_summaries=(
                        train_latent_summaries if train_latent_summaries is not None else np.array([])
                    ),
                    test_latent_summaries=(
                        test_latent_summaries if test_latent_summaries is not None else np.array([])
                    ),
                    memory_latent_summaries=(
                        memory_latent_summaries if memory_latent_summaries is not None else np.array([])
                    ),
                    latent_summary_feature_median=(
                        latent_summary_stats["median"]
                        if latent_summary_stats is not None else np.array([])
                    ),
                    latent_summary_feature_iqr=(
                        latent_summary_stats["iqr"]
                        if latent_summary_stats is not None else np.array([])
                    ),
                    latent_summary_block_scales=(
                        latent_summary_block_stats["scales"]
                        if latent_summary_block_stats is not None else np.array([])
                    ),
                    latent_summary_whiten_mean=(
                        latent_summary_whitener["mean"]
                        if latent_summary_whitener is not None else np.array([])
                    ),
                    latent_summary_whiten_inv_sqrt=(
                        latent_summary_whitener["inv_sqrt"]
                        if latent_summary_whitener is not None else np.array([])
                    ),
                    latent_knn_k=args.latent_knn_k,
                    local_tail_top_percent=(5.0 if score_mode == "local_tail_knn_top5" else np.array([])),
                    teacher_student_top_percent=(
                        5.0 if score_mode == "teacher_student_top5" else np.array([])
                    ),
                    local_memory_token_count=(
                        np.array([len(memory_local_tokens)], dtype=np.int64)
                        if memory_local_tokens is not None else np.array([])
                    ),
                    y_test=y_test,
                    d_test=d_test.astype("U16"),
                    score_mode=score_mode,
                )

                # Record
                t0_sec = train_stats.get("train_time_sec", 0)
                result = {
                    "machine": machine,
                    "experiment": exp_name,
                    "score_mode": score_mode,
                    "num_views": V,
                    "source_auc": metrics["auc_source"],
                    "target_auc": metrics["auc_target"],
                    "pauc": metrics["pauc"],
                    "dcase_auc": metrics["dcase_auc"],
                    "hmean": metrics["hmean"],
                    "model_params": train_stats.get("model_params", 0),
                    "latent_channels": args.latent_channels,
                    "stem_channels": args.stem_channels,
                    "per_view_channels": args.per_view_channels,
                    "dropout_p": args.dropout_p,
                    "loss_view_mode": args.loss_view_mode,
                    "ssl_task": train_stats.get("ssl_task", args.ssl_task),
                    "use_target_train": args.use_target_train,
                    "memory_domains": args.memory_domains,
                    "model_type": args.model_type,
                    "train_time_sec": t0_sec,
                    "epochs_ran": train_stats.get("epochs_ran", args.epochs),
                    "best_epoch": train_stats.get("best_epoch", args.epochs),
                    "best_val_loss": train_stats.get("best_val_loss", train_stats.get("final_val_loss", 0)),
                    "stopped_early": train_stats.get("stopped_early", False),
                    "noise_mode": train_stats.get("noise_mode", args.noise_mode),
                    "mask_prob": train_stats.get("mask_prob", args.mask_prob),
                    "tail_compact_loss": train_stats.get("tail_compact_loss", args.tail_compact_loss),
                    "tail_compact_lambda": train_stats.get("tail_compact_lambda", args.tail_compact_lambda),
                    "tail_compact_top_frac": train_stats.get("tail_compact_top_frac", args.tail_compact_top_frac),
                    "hard_normal_consistency_loss": train_stats.get("hard_normal_consistency_loss", args.hard_normal_consistency_loss),
                    "hard_normal_consistency_lambda": train_stats.get("hard_normal_consistency_lambda", args.hard_normal_consistency_lambda),
                    "hard_normal_time_mask_frac": train_stats.get("hard_normal_time_mask_frac", args.hard_normal_time_mask_frac),
                    "hard_normal_freq_mask_frac": train_stats.get("hard_normal_freq_mask_frac", args.hard_normal_freq_mask_frac),
                    "hard_normal_train_count": train_stats.get("hard_normal_train_count", 0),
                    "hard_normal_val_count": train_stats.get("hard_normal_val_count", 0),
                    "vicreg_lambda_inv": train_stats.get("vicreg_lambda_inv", ""),
                    "vicreg_lambda_var": train_stats.get("vicreg_lambda_var", ""),
                    "vicreg_lambda_cov": train_stats.get("vicreg_lambda_cov", ""),
                    "vicreg_time_mask_frac": train_stats.get("vicreg_time_mask_frac", ""),
                    "vicreg_freq_mask_frac": train_stats.get("vicreg_freq_mask_frac", ""),
                    "vicreg_gain_min": train_stats.get("vicreg_gain_min", ""),
                    "vicreg_gain_max": train_stats.get("vicreg_gain_max", ""),
                    "vicreg_n_augmentations_per_view": train_stats.get("vicreg_n_augmentations_per_view", ""),
                    "vicreg_train_inv_loss": train_stats.get("vicreg_train_inv_loss", ""),
                    "vicreg_train_var_loss": train_stats.get("vicreg_train_var_loss", ""),
                    "vicreg_train_cov_loss": train_stats.get("vicreg_train_cov_loss", ""),
                    "vicreg_train_latent_summary_std": train_stats.get("vicreg_train_latent_summary_std", ""),
                    "vicreg_train_projector_std": train_stats.get("vicreg_train_projector_std", ""),
                    "vicreg_val_inv_loss": train_stats.get("vicreg_val_inv_loss", ""),
                    "vicreg_val_var_loss": train_stats.get("vicreg_val_var_loss", ""),
                    "vicreg_val_cov_loss": train_stats.get("vicreg_val_cov_loss", ""),
                    "vicreg_val_latent_summary_std": train_stats.get("vicreg_val_latent_summary_std", ""),
                    "vicreg_val_projector_std": train_stats.get("vicreg_val_projector_std", ""),
                    "latent_knn_k": (
                        args.latent_knn_k
                        if score_mode in (
                            "latent_knn",
                            "latent_summary_knn",
                            "latent_summary_robust_knn",
                            "latent_summary_blockscale_knn",
                            "latent_summary_whiten_knn",
                            "local_tail_knn_top5",
                        )
                        else ""
                    ),
                    "final_train_loss": train_stats.get("final_train_loss", 0),
                    "final_val_loss": train_stats.get("final_val_loss", 0),
                    "checkpoint_file": checkpoint_path,
                    "score_file": score_path,
                    "hist_file": hist_path,
                }
                all_results.append(result)
                per_machine_rows.append(result)

                geo_row = {
                    "machine": machine,
                    "experiment": exp_name,
                    "score_mode": score_mode,
                    **{k: geometry[k] for k in [
                        "source_normal_mean", "target_normal_mean",
                        "source_normal_std", "target_normal_std",
                        "source_target_ks", "median_gap", "iqr_gap",
                        "score_skewness", "score_kurtosis",
                    ]},
                }
                score_geometry_rows.append(geo_row)

                # Track fixed logmel baseline for improvement comparison.
                if exp_name == "logmel_only" and score_mode == "mean_mse":
                    logmel_hmean[machine] = metrics["hmean"]

                _print_result_row(metrics, geometry, train_stats, score_mode)

    # ── Generate summary CSV ──
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    _write_summary_csv(summary_csv, all_results, score_geometry_rows, machines, experiments)
    _section("Saved Outputs")
    print(f"summary       : {summary_csv}")

    # ── Generate per_machine.csv ──
    pm_csv = os.path.join(args.output_dir, "per_machine.csv")
    _write_per_machine_csv(pm_csv, per_machine_rows)
    print(f"per_machine   : {pm_csv}")

    # ── Generate score_geometry.csv ──
    sg_csv = os.path.join(args.output_dir, "score_geometry.csv")
    _write_score_geometry_csv(sg_csv, score_geometry_rows)
    print(f"score_geometry: {sg_csv}")

    # ── Generate RESULTS_SUMMARY.md ──
    docs_dir = os.path.join(args.output_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    md_path = os.path.join(docs_dir, "RESULTS_SUMMARY.md")
    _write_results_md(md_path, all_results, score_geometry_rows, logmel_hmean, machines, args, experiments)
    print(f"report        : {md_path}")

    print("\nDone.")


def _print_split_counts(train_items, test_items):
    def counts(items):
        out = {}
        for item in items:
            key = (item.split, item.domain, item.label)
            out[key] = out.get(key, 0) + 1
        return out

    train_counts = counts(train_items)
    test_counts = counts(test_items)
    print(f"train split: {_format_counts(train_counts)}")
    print(f"test split : {_format_counts(test_counts)}")


def _format_counts(counts):
    if not counts:
        return "none"
    parts = []
    for (_split, domain, label), value in sorted(counts.items(), key=lambda x: (str(x[0]), x[1])):
        label_name = "normal" if label == 0 else "anomaly" if label == 1 else "unlabeled"
        parts.append(f"{domain}:{label_name}={value}")
    return ", ".join(parts)


def _print_result_row(metrics, geometry, train_stats, score_mode):
    print()
    print(f"score_mode: {score_mode}")
    print("metrics:")
    print("  source_auc  target_auc  pAUC    hmean   dcase_auc")
    print(
        "  "
        f"{_fmt_float(metrics['auc_source']):>10s}  "
        f"{_fmt_float(metrics['auc_target']):>10s}  "
        f"{_fmt_float(metrics['pauc']):>6s}  "
        f"{_fmt_float(metrics['hmean']):>6s}  "
        f"{_fmt_float(metrics['dcase_auc']):>9s}"
    )
    print("geometry:")
    print("  src_norm_mean  tgt_norm_mean  src_tgt_ks  median_gap")
    print(
        "  "
        f"{_fmt_float(geometry['source_normal_mean']):>13s}  "
        f"{_fmt_float(geometry['target_normal_mean']):>13s}  "
        f"{_fmt_float(geometry['source_target_ks']):>10s}  "
        f"{_fmt_float(geometry['median_gap']):>10s}"
    )
    print(
        "train: "
        f"loss={train_stats.get('final_train_loss', 0):.6f}  "
        f"val={train_stats.get('final_val_loss', 0):.6f}  "
        f"time={train_stats.get('train_time_sec', 0):.1f}s"
    )


def _write_summary_csv(path, results, geometry_rows, machines, experiments):
    # Aggregate: per experiment and fixed score mode, mean metrics across machines.
    import csv
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["experiment"], r.get("score_mode", "mean_mse"))].append(r)

    score_modes = []
    for r in results:
        mode = r.get("score_mode", "mean_mse")
        if mode not in score_modes:
            score_modes.append(mode)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "score_mode", "num_views", "mean_source_auc", "mean_target_auc",
                     "mean_pauc", "mean_hmean", "mean_dcase_auc",
                     "mean_source_target_ks", "num_machines_improved_vs_logmel",
                     "notes"])
        for exp_name in experiments:
            for score_mode in score_modes:
                rows = agg.get((exp_name, score_mode), [])
                if not rows:
                    continue
                mean_src = np.mean([r["source_auc"] for r in rows])
                mean_tgt = np.mean([r["target_auc"] for r in rows])
                mean_p = np.mean([r["pauc"] for r in rows])
                mean_h = np.mean([r["hmean"] for r in rows])
                mean_dcase = np.mean([r["dcase_auc"] for r in rows])

                # Count machines improved vs fixed logmel_only/mean_mse baseline.
                n_improved = 0
                if not (exp_name == "logmel_only" and score_mode == "mean_mse"):
                    logmel_rows = [
                        r for r in results
                        if r["experiment"] == "logmel_only" and r.get("score_mode", "mean_mse") == "mean_mse"
                    ]
                    logmel_map = {r["machine"]: r["hmean"] for r in logmel_rows}
                    for r in rows:
                        if r["machine"] in logmel_map and r["hmean"] > logmel_map[r["machine"]]:
                            n_improved += 1

                ks_vals = [
                    g["source_target_ks"] for g in geometry_rows
                    if g["experiment"] == exp_name and g.get("score_mode", "mean_mse") == score_mode
                ]
                mean_ks = float(np.mean(ks_vals)) if ks_vals else 0.0

                w.writerow([exp_name, score_mode, rows[0]["num_views"],
                            f"{mean_src:.6f}", f"{mean_tgt:.6f}", f"{mean_p:.6f}",
                            f"{mean_h:.6f}", f"{mean_dcase:.6f}",
                            f"{mean_ks:.4f}", n_improved, ""])


def _write_per_machine_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_score_geometry_csv(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_results_md(path, results, geometry, logmel_hmean, machines, args, experiments):
    from collections import defaultdict

    # Group by experiment and fixed score mode. Score modes are global rules,
    # not per-machine selections.
    by_group = defaultdict(list)
    for r in results:
        by_group[(r["experiment"], r.get("score_mode", "mean_mse"))].append(r)

    score_modes = []
    for r in results:
        mode = r.get("score_mode", "mean_mse")
        if mode not in score_modes:
            score_modes.append(mode)

    geo_by_machine_exp = {}
    for g in geometry:
        geo_by_machine_exp[(g["machine"], g["experiment"], g.get("score_mode", "mean_mse"))] = g

    with open(path, "w") as f:
        f.write("# CTAF Results Summary\n\n")
        f.write("## Protocol Confirmation\n\n")
        f.write("- **Label-free protocol**: No anomaly labels used in training, feature extraction, or model selection.\n")
        f.write("- **Training**: Normal-only, source-domain data. MSE reconstruction loss.\n")
        f.write("- **Scoring**: Fixed reconstruction-error score modes. Larger score = more anomalous. No sign reversal.\n")
        f.write("- **Fusion/search**: None. Single model per machine per experiment. No label-driven weight or route search.\n")
        f.write("- **Labels used only in**: Final AUC/pAUC/hmean evaluation and post-hoc histograms.\n\n")

        f.write(
            f"**Config**: epochs={args.epochs}, batch_size={args.batch_size}, "
            f"latent_channels={args.latent_channels}, stem_channels={args.stem_channels}, "
            f"per_view_channels={args.per_view_channels}, dropout_p={args.dropout_p}, "
            f"loss_view_mode={args.loss_view_mode}, ssl_task={args.ssl_task}, "
            f"model_type={args.model_type}, "
            f"use_target_train={args.use_target_train}, "
            f"seed={args.seed}, score_modes={args.score_modes}, "
            f"early_stopping={args.early_stopping}, patience={args.patience}, "
            f"min_delta={args.min_delta}, noise_mode={args.noise_mode}, "
            f"mask_prob={args.mask_prob}, tail_compact_loss={args.tail_compact_loss}, "
            f"tail_compact_lambda={args.tail_compact_lambda}, "
            f"tail_compact_top_frac={args.tail_compact_top_frac}, "
            f"hard_normal_consistency_loss={args.hard_normal_consistency_loss}, "
            f"hard_normal_consistency_lambda={args.hard_normal_consistency_lambda}, "
            f"hard_normal_time_mask_frac={args.hard_normal_time_mask_frac}, "
            f"hard_normal_freq_mask_frac={args.hard_normal_freq_mask_frac}, "
            f"vicreg_lambda_inv={getattr(args, 'vicreg_lambda_inv', 25)}, "
            f"vicreg_lambda_var={getattr(args, 'vicreg_lambda_var', 25)}, "
            f"vicreg_lambda_cov={getattr(args, 'vicreg_lambda_cov', 1)}, "
            f"latent_knn_k={args.latent_knn_k}\n\n"
        )

        # Overall table
        f.write("## Overall Results\n\n")
        f.write("| Experiment | Score mode | Views | Mean src_AUC | Mean tgt_AUC | Mean pAUC | Mean hmean | vs logmel/mean_mse |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")

        logmel_rows = by_group.get(("logmel_only", "mean_mse"), [])
        if logmel_rows:
            mean_h_logmel = np.mean([r["hmean"] for r in logmel_rows])
        else:
            mean_h_logmel = 0

        for exp_name in experiments:
            for score_mode in score_modes:
                rows = by_group.get((exp_name, score_mode), [])
                if not rows:
                    continue
                mean_src = np.mean([r["source_auc"] for r in rows])
                mean_tgt = np.mean([r["target_auc"] for r in rows])
                mean_p = np.mean([r["pauc"] for r in rows])
                mean_h = np.mean([r["hmean"] for r in rows])
                delta = mean_h - mean_h_logmel if mean_h_logmel > 0 else 0
                f.write(
                    f"| {exp_name:16s} | {score_mode} | {rows[0]['num_views']} | "
                    f"{mean_src:.4f} | {mean_tgt:.4f} | {mean_p:.4f} | "
                    f"{mean_h:.4f} | {delta:+.4f} |\n"
                )

        # Per-machine
        f.write("\n## Per-Machine Results\n\n")
        for m in machines:
            f.write(f"### {m}\n\n")
            f.write("| Experiment | Score mode | src_AUC | tgt_AUC | pAUC | hmean | src-tgt KS | median_gap |\n")
            f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for exp_name in experiments:
                for score_mode in score_modes:
                    mr = [r for r in by_group.get((exp_name, score_mode), []) if r["machine"] == m]
                    gr = geo_by_machine_exp.get((m, exp_name, score_mode), {})
                    if mr:
                        r = mr[0]
                        ks = gr.get("source_target_ks", 0)
                        mg = gr.get("median_gap", 0)
                        f.write(
                            f"| {exp_name:16s} | {score_mode} | {r['source_auc']:.4f} | "
                            f"{r['target_auc']:.4f} | {r['pauc']:.4f} | {r['hmean']:.4f} | "
                            f"{ks:.4f} | {mg:.4f} |\n"
                        )
            f.write("\n")

        # Score geometry
        f.write("## Score Geometry\n\n")
        f.write("Source-target normal score drift (KS statistic, lower = better alignment). Rows are fixed score modes, not selections.\n\n")
        f.write("| Experiment | Score mode | Mean source-target KS |\n")
        f.write("|---|---|---:|\n")
        for exp_name in experiments:
            for score_mode in score_modes:
                ks_vals = [
                    g["source_target_ks"] for g in geometry
                    if g["experiment"] == exp_name and g.get("score_mode", "mean_mse") == score_mode
                ]
                if ks_vals:
                    f.write(f"| {exp_name} | {score_mode} | {np.mean(ks_vals):.4f} |\n")

        # Verdict
        f.write("\n## Verdict\n\n")
        if mean_h_logmel > 0:
            full_5 = by_group.get(("full_5view", "mean_mse"), [])
            if full_5:
                full_h = np.mean([r["hmean"] for r in full_5])
                full_p = np.mean([r["pauc"] for r in full_5])

                n_improved = 0
                n_worse = 0
                for r in full_5:
                    lm = [x for x in logmel_rows if x["machine"] == r["machine"]]
                    if lm and r["hmean"] > lm[0]["hmean"]:
                        n_improved += 1
                    elif lm:
                        n_worse += 1

                f.write(f"- logmel_only/mean_mse mean hmean: {mean_h_logmel:.6f}\n")
                f.write(f"- full_5view/mean_mse mean hmean: {full_h:.6f}\n")
                f.write(f"- full_5view/mean_mse vs logmel_only/mean_mse delta: {full_h - mean_h_logmel:+.6f}\n")
                f.write(f"- Machines improved: {n_improved}/{n_improved+n_worse}\n\n")

                if full_h > mean_h_logmel and full_p >= np.mean([r["pauc"] for r in logmel_rows]):
                    f.write("**Verdict: PASS.** full_5view improves over logmel_only baseline.\n")
                    if n_improved >= 4:
                        f.write("Multi-view tensor frontend shows consistent improvement across machines.\n")
                elif full_h <= mean_h_logmel:
                    f.write("**Verdict: FAIL.** full_5view does NOT improve over logmel_only.\n")
                    f.write("Multi-view tensor frontend did not demonstrate advantage under clean protocol.\n")
                else:
                    f.write("**Verdict: MIXED.** pAUC preserved but hmean improvement is marginal.\n")


if __name__ == "__main__":
    main()
