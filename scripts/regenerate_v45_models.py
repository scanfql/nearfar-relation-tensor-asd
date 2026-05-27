#!/usr/bin/env python
"""Regenerate V16 and V37 full_5view checkpoints for V45 segment transfer audit.

Usage:
  # V16: full_5view AE, seed 13711
  python scripts/regenerate_v45_models.py --data_root ... --out_dir ... --device cuda:0 \
    --experiment full_5view --ssl_task reconstruction --seeds 13711

  # V37: full_5view teacher-student, seeds 13711,13721,13731
  python scripts/regenerate_v45_models.py --data_root ... --out_dir ... --device cuda:1 \
    --experiment full_5view --ssl_task teacher_student_local_distillation \
    --seeds 13711,13721,13731
"""

import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import load_audio, extract_logmel
from src.features.tensor_builder import build_tensor, TensorNormalizer
from src.training.train_ae import train_autoencoder, train_teacher_student_local_distillation
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder
from src.utils.seed import set_seed

MACHINES = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--experiment", default="full_5view")
    p.add_argument("--ssl_task", default="reconstruction",
                   choices=["reconstruction", "teacher_student_local_distillation"])
    p.add_argument("--seeds", default="13711")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    all_items = scan_dataset(Path(args.data_root))

    for seed in seeds:
        set_seed(seed)
        for machine in MACHINES:
            model_path = models_dir / f"{machine}_full_5view_seed{seed}.pt"
            if model_path.exists():
                print(f"[seed {seed} / {machine}] SKIP: model exists")
                continue

            print(f"\n[seed {seed} / {machine}] extracting features...")
            items = filter_items(all_items, machine)
            train_items, test_items = split_train_test(items, use_target_train=False)

            train_tensors = []
            for item in train_items:
                audio = load_audio(item.path)
                base = extract_logmel(audio)
                t = build_tensor(args.experiment, base)
                train_tensors.append(t)

            V = train_tensors[0].shape[0]
            normalizer = TensorNormalizer()
            train_norm = normalizer.fit_transform(train_tensors)
            train_arr = np.stack(train_norm, axis=0)

            if args.ssl_task == "teacher_student_local_distillation":
                # Train teacher
                teacher = ViewwiseUnsharedLateFusionAutoencoder(
                    in_channels=V, latent_channels=64, use_batch_norm=True,
                    per_view_channels=16, stem_channels=8, dropout_p=0.0,
                )
                print(f"[seed {seed} / {machine}] training teacher AE...")
                train_autoencoder(teacher, train_arr, epochs=50, batch_size=32,
                                  device=args.device, seed=seed, early_stopping=False,
                                  noise_mode="none", mask_prob=0.0,
                                  loss_view_mode="all_views", ssl_task="reconstruction",
                                  tail_compact_loss=False)

                # Train student
                student = ViewwiseUnsharedLateFusionAutoencoder(
                    in_channels=V, latent_channels=64, use_batch_norm=True,
                    per_view_channels=16, stem_channels=8, dropout_p=0.0,
                )
                print(f"[seed {seed} / {machine}] training student (TS distillation)...")
                train_teacher_student_local_distillation(
                    teacher, student, train_arr,
                    epochs=50, batch_size=32,
                    device=args.device, seed=seed,
                    early_stopping=False,
                )
                # Save TEACHER (V37 uses teacher for scoring)
                torch.save(teacher.state_dict(), model_path)
                print(f"  saved: {model_path}")
            else:
                model = ViewwiseUnsharedLateFusionAutoencoder(
                    in_channels=V, latent_channels=64, use_batch_norm=True,
                    per_view_channels=16, stem_channels=8, dropout_p=0.0,
                )
                print(f"[seed {seed} / {machine}] training AE...")
                train_autoencoder(model, train_arr, epochs=50, batch_size=32,
                                  device=args.device, seed=seed, early_stopping=False,
                                  noise_mode="none", mask_prob=0.0,
                                  loss_view_mode="all_views", ssl_task="reconstruction",
                                  tail_compact_loss=False)
                torch.save(model.state_dict(), model_path)
                print(f"  saved: {model_path}")

    print(f"\nDone. Models saved to {models_dir}")


if __name__ == "__main__":
    main()
