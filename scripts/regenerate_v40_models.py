#!/usr/bin/env python
"""Regenerate V40 models on the exact same data/seed to enable V41 diagnostic.

Reproduces V40 training deterministically (seed=13711). Saves model
state_dict for each (machine, experiment). Verifies score reproduction
by comparing against original V40 NPZ scores.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--v40_dir", required=True,
                   help="V40 output dir (for score comparison and model saving)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--machines", default="gearbox,slider,toycar,toytrain,fan")
    p.add_argument("--experiments", default="logmel_only,full_5view,transient_3view")
    return p.parse_args()


def main():
    args = parse_args()

    from src.data.dataset import scan_dataset, filter_items, split_train_test
    from src.features.logmel import extract_logmel, extract_multires_logmels, load_audio
    from src.features.tensor_builder import build_tensor, TensorNormalizer
    from src.training.train_ae import train_autoencoder
    from src.evaluation.scorer import (
        compute_recon_error_views,
        compute_latent_summary_vectors,
        compute_latent_knn_scores,
        fit_feature_robust_stats,
        fit_view_residual_stats,
    )
    from src.utils.seed import set_seed

    from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder

    machines = [m.strip() for m in args.machines.split(",")]
    experiments = [e.strip() for e in args.experiments.split(",")]
    v40_dir = Path(args.v40_dir)
    models_dir = v40_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_items = scan_dataset(Path(args.data_root))
    set_seed(13711)

    for machine in machines:
        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)

        need_multires = "multires_7view" in experiments

        # Extract features (same order as V40)
        train_feats = {}
        for item in train_items:
            audio = load_audio(item.path)
            base = extract_logmel(audio)
            multires = extract_multires_logmels(audio) if need_multires else None
            train_feats[item.path] = (base, multires, audio, item.label, item.domain)

        test_feats = {}
        for item in test_items:
            audio = load_audio(item.path)
            base = extract_logmel(audio)
            multires = extract_multires_logmels(audio) if need_multires else None
            test_feats[item.path] = (base, multires, audio, item.label, item.domain)

        for exp_name in experiments:
            print(f"[{machine}/{exp_name}] building tensors...")

            train_tensors = []
            for item in train_items:
                base, multires, audio_t, _, _ = train_feats[item.path]
                feat = multires if exp_name == "multires_7view" else base
                t = build_tensor(exp_name, feat, audio=audio_t)
                train_tensors.append(t)

            test_tensors = []
            test_labels = []
            test_domains = []
            for item in test_items:
                base, multires, audio_t, label, domain = test_feats[item.path]
                feat = multires if exp_name == "multires_7view" else base
                t = build_tensor(exp_name, feat, audio=audio_t)
                test_tensors.append(t)
                test_labels.append(label)
                test_domains.append(domain)

            V = train_tensors[0].shape[0]

            normalizer = TensorNormalizer()
            train_norm = normalizer.fit_transform(train_tensors)
            test_norm = [normalizer.transform(t) for t in test_tensors]
            train_arr = np.stack(train_norm, axis=0)
            test_arr = np.stack(test_norm, axis=0)

            # Build model (same as V40)
            model = ViewwiseUnsharedLateFusionAutoencoder(
                in_channels=V,
                latent_channels=64,
                use_batch_norm=True,
                per_view_channels=16,
                stem_channels=8,
                dropout_p=0.0,
            )

            print(f"[{machine}/{exp_name}] training...")
            train_autoencoder(
                model, train_arr,
                epochs=50, batch_size=32,
                device=args.device, seed=13711,
                early_stopping=False,
                noise_mode="none", mask_prob=0.0,
                loss_view_mode="all_views",
                ssl_task="reconstruction",
                tail_compact_loss=False,
            )

            # Save model
            model_path = models_dir / f"{machine}_{exp_name}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"  saved model to {model_path}")

            # Verify score reproduction
            model.eval()
            with torch.no_grad():
                # Compute latent_summary_knn score (same as V40)
                memory_latent_summaries = compute_latent_summary_vectors(
                    model, train_arr, batch_size=32, device=args.device
                )
                test_latent_summaries = compute_latent_summary_vectors(
                    model, test_arr, batch_size=32, device=args.device
                )
                test_scores = compute_latent_knn_scores(
                    memory_latent_summaries, test_latent_summaries, k=5
                )

            # Compare with original V40 scores (check first 5)
            score_label = f"{exp_name}_latent_summary_knn"
            orig_npz = v40_dir / "scores" / f"{machine}_{score_label}_scores.npz"
            if orig_npz.exists():
                orig = np.load(orig_npz, allow_pickle=True)
                orig_scores = orig["test_scores"].astype(np.float64)
                max_diff = float(np.max(np.abs(test_scores - orig_scores)))
                print(f"  score_reprod: max_diff={max_diff:.6f} "
                      f"({'OK' if max_diff < 1e-4 else 'MISMATCH!'})")
            else:
                print(f"  (no original score file to compare)")

    print("\nDone. Models saved to:", models_dir)


if __name__ == "__main__":
    main()
