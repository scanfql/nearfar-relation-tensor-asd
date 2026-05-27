#!/usr/bin/env python
"""Generate fixed hard-normal indices for V49 from V37 representations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import scan_dataset, filter_items, split_train_test
from src.features.logmel import extract_logmel, load_audio
from src.features.tensor_builder import TensorNormalizer, build_tensor
from src.models.conv_ae import ViewwiseUnsharedLateFusionAutoencoder
from src.evaluation.scorer import compute_latent_knn_scores, compute_latent_summary_vectors


DEFAULT_SEEDS = "13711,13721,13731"
DEFAULT_MACHINES = "bearing,fan,gearbox,slider,toycar,toytrain,valve"


def parse_args():
    p = argparse.ArgumentParser(description="Generate V49 hard-normal indices")
    p.add_argument("--data_root", required=True)
    p.add_argument("--score_seed_dirs", required=True,
                   help="One root containing seed_<SEED> dirs or one dir per seed")
    p.add_argument("--checkpoint_dirs", required=True,
                   help="One root containing seed_<SEED>/models dirs or one models dir per seed")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seeds", default=DEFAULT_SEEDS)
    p.add_argument("--machines", default=DEFAULT_MACHINES)
    p.add_argument("--experiment", default="full_5view")
    p.add_argument("--score_mode", default="latent_summary_knn")
    p.add_argument("--hard_normal_top_ratio", type=float, default=0.10)
    p.add_argument("--easy_normal_bottom_ratio", type=float, default=0.50)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def parse_seed_list(value: str):
    return [s.strip() for s in value.split(",") if s.strip()]


def resolve_seed_dirs(spec: str, seeds: list[str], expect_models: bool = False):
    parts = [Path(x.strip()) for x in spec.split(",") if x.strip()]
    if len(parts) == 1:
        if len(seeds) == 1:
            return parts
        root = parts[0]
        out = []
        for seed in seeds:
            cand = root / f"seed_{seed}"
            if expect_models:
                cand = cand / "models"
            out.append(cand)
        return out
    if len(parts) != len(seeds):
        raise ValueError("dir spec must be one root or one path per seed")
    return parts


def load_model(checkpoint_path: Path):
    model = ViewwiseUnsharedLateFusionAutoencoder(
        in_channels=5,
        latent_channels=64,
        use_batch_norm=True,
        per_view_channels=16,
        stem_channels=8,
        dropout_p=0.0,
    )
    blob = torch.load(checkpoint_path, map_location="cpu")
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(state)
    return model


def build_train_array(train_items):
    tensors = []
    for item in train_items:
        audio = load_audio(item.path)
        logmel = extract_logmel(audio)
        tensors.append(build_tensor("full_5view", logmel, audio=audio))
    normalizer = TensorNormalizer()
    train_norm = normalizer.fit_transform(tensors)
    return np.stack(train_norm, axis=0)


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    machines = parse_seed_list(args.machines)
    score_seed_dirs = resolve_seed_dirs(args.score_seed_dirs, seeds, expect_models=False)
    checkpoint_dirs = resolve_seed_dirs(args.checkpoint_dirs, seeds, expect_models=True)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    items = scan_dataset(Path(args.data_root))

    for seed, score_dir, ckpt_dir in zip(seeds, score_seed_dirs, checkpoint_dirs):
        seed_out = out_root / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        for machine in machines:
            score_npz = score_dir / "scores" / f"{machine}_{args.experiment}_{args.score_mode}_scores.npz"
            train_scores = None
            source = "score_file"
            if score_npz.exists():
                arr = np.load(score_npz, allow_pickle=True)
                train_scores = arr["train_scores"].astype(np.float64)
            else:
                ckpt = ckpt_dir / f"{machine}_{args.experiment}_seed{seed}.pt"
                if not ckpt.exists():
                    raise FileNotFoundError(f"Missing both score file and checkpoint for {machine} seed {seed}: {score_npz} ; {ckpt}")
                source = "checkpoint_fallback"
                m_items = filter_items(items, machine)
                train_items, _ = split_train_test(m_items, use_target_train=False)
                train_arr = build_train_array(train_items)
                model = load_model(ckpt)
                train_summaries = compute_latent_summary_vectors(model, train_arr, batch_size=32, device=args.device)
                train_scores = compute_latent_knn_scores(train_summaries, train_summaries, k=args.latent_knn_k)

            m_items = filter_items(items, machine)
            train_items, _ = split_train_test(m_items, use_target_train=False)
            if len(train_items) != len(train_scores):
                raise ValueError(f"Train item count mismatch for {machine} seed {seed}: {len(train_items)} vs {len(train_scores)}")
            order = np.argsort(train_scores)
            hard_n = max(1, int(np.ceil(len(train_scores) * args.hard_normal_top_ratio)))
            easy_n = max(1, int(np.floor(len(train_scores) * args.easy_normal_bottom_ratio)))
            hard_set = set(order[-hard_n:].tolist())
            easy_set = set(order[:easy_n].tolist())
            rows = []
            for idx, (item, score) in enumerate(zip(train_items, train_scores)):
                rows.append({
                    "seed": seed,
                    "machine": machine,
                    "train_index": idx,
                    "sample_path": str(item.path),
                    "score": f"{float(score):.6f}",
                    "hard_normal_flag": int(idx in hard_set),
                    "easy_normal_flag": int(idx in easy_set),
                    "source": source,
                })
            out_csv = seed_out / f"{machine}_{args.experiment}_{args.score_mode}_hard_normal.csv"
            with out_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
