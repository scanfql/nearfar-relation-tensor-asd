#!/usr/bin/env python
"""Stage25A: EAT segment pseudo-boundary head.

This is the first learned local-evidence probe after Stage24A failed raw segment
kNN scoring. It reuses clean EAT segment embeddings extracted by Stage24A and
trains a small per-machine binary head:

- clean segment embedding -> label 0
- synthetic pseudo segment embedding -> label 1

Pseudo segments are generated only from source train-normal segment embeddings
by fixed same-machine subspace replacement. Checkpoint selection uses validation
BCE on clean/pseudo segments, never test labels.

Inference:
- score every test segment with the pseudo-boundary head
- file score = mean top 5% segment pseudo-anomaly probabilities

No EAT training, no anomaly labels for training, no LDN tuning, no selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
SEGMENT_ROOT_DEFAULT = "results/stage24a_eat_segment_evidence"
OUTPUT_ROOT_DEFAULT = "results/stage25a_eat_segment_pseudo_boundary"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE25A_EAT_SEGMENT_PSEUDO_BOUNDARY.md"
CLIP_REFERENCE_DEFAULT = "results/stage18e_local_density_protocol_full7/per_machine.csv"

CANDIDATE = "eat_segment_pseudo_boundary_top5pct"
CLIP_REFERENCE_CANDIDATE = "ldn_diff_K1_scorek5"
CONFIRMED_STAGE22_HMEAN = 0.6166917954459288
CONFIRMED_CLEAN_LDN_HMEAN = 0.6159467288425517


def parse_args():
    p = argparse.ArgumentParser(description="Stage25A EAT segment pseudo-boundary head")
    p.add_argument("--mode", required=True, choices=["check", "train_score"])
    p.add_argument("--segment_root", default=SEGMENT_ROOT_DEFAULT)
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--clip_reference_per_machine", default=CLIP_REFERENCE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--feature_patch_frac", type=float, default=0.10)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--top_percent", type=float, default=5.0)
    p.add_argument("--hidden_dim", type=int, default=128)
    return p.parse_args()


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embedding_path(segment_root: Path, machine: str) -> Path:
    return segment_root / "embeddings" / f"{machine}_eat_segment_embeddings.npz"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_clip_reference(path: Path) -> dict[str, dict]:
    rows = read_csv(path)
    return {
        row["machine"]: row
        for row in rows
        if row.get("candidate") == CLIP_REFERENCE_CANDIDATE
    }


class BoundaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize(train: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sigma = train.std(axis=0, keepdims=True)
    sigma = np.maximum(sigma, 1e-6)
    return ((x - mu) / sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)


def apply_standardize(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((x - mu) / np.maximum(sigma, 1e-6)).astype(np.float32)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_n = max(1, int(n * val_ratio))
    val_n = min(val_n, n - 1)
    return idx[val_n:], idx[:val_n]


def make_pseudo_batch(clean_batch, bank, patch_frac: float, noise_std: float):
    bsz, dim = clean_batch.shape
    donor_idx = torch.randint(0, bank.shape[0], (bsz,), device=clean_batch.device)
    donors = bank[donor_idx]
    width = max(1, int(dim * patch_frac))
    starts = torch.randint(0, max(1, dim - width + 1), (bsz,), device=clean_batch.device)
    dim_ids = torch.arange(dim, device=clean_batch.device).unsqueeze(0)
    mask = (dim_ids >= starts.unsqueeze(1)) & (dim_ids < (starts + width).unsqueeze(1))
    pseudo = torch.where(mask, donors, clean_batch)
    if noise_std > 0:
        pseudo = pseudo + torch.randn_like(pseudo) * noise_std
    return pseudo


def train_one_machine(
    machine: str,
    train_segments: np.ndarray,
    test_segments: np.ndarray,
    test_offsets: list[dict],
    y_test: np.ndarray,
    d_test: np.ndarray,
    test_paths: np.ndarray,
    args,
    device,
) -> tuple[dict, dict, dict, np.ndarray, list[dict], list[dict]]:
    train_idx, val_idx = split_indices(train_segments.shape[0], args.val_ratio, args.seed)
    train_std, mu, sigma = standardize(train_segments[train_idx], train_segments)
    test_std = apply_standardize(test_segments, mu, sigma)
    train_bank = torch.tensor(train_std[train_idx], dtype=torch.float32, device=device)
    val_bank = torch.tensor(train_std[val_idx], dtype=torch.float32, device=device)

    model = BoundaryHead(train_segments.shape[1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_state = None
    log_rows = []
    rng = np.random.default_rng(args.seed)
    train_n = train_bank.shape[0]
    val_n = val_bank.shape[0]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(train_n)
        train_loss_sum = 0.0
        train_count = 0
        for start in range(0, train_n, args.batch_size):
            idx = torch.tensor(perm[start:start + args.batch_size], dtype=torch.long, device=device)
            clean = train_bank[idx]
            pseudo = make_pseudo_batch(clean, train_bank, args.feature_patch_frac, args.noise_std)
            x = torch.cat([clean, pseudo], dim=0)
            y = torch.cat([
                torch.zeros(clean.shape[0], device=device),
                torch.ones(pseudo.shape[0], device=device),
            ], dim=0)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            train_loss_sum += float(loss.item()) * x.shape[0]
            train_count += x.shape[0]

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for start in range(0, val_n, args.batch_size):
                clean = val_bank[start:start + args.batch_size]
                pseudo = make_pseudo_batch(clean, train_bank, args.feature_patch_frac, args.noise_std)
                x = torch.cat([clean, pseudo], dim=0)
                y = torch.cat([
                    torch.zeros(clean.shape[0], device=device),
                    torch.ones(pseudo.shape[0], device=device),
                ], dim=0)
                loss = criterion(model(x), y)
                val_loss_sum += float(loss.item()) * x.shape[0]
                val_count += x.shape[0]
        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        log_rows.append({
            "machine": machine,
            "epoch": epoch,
            "train_bce": train_loss,
            "val_bce": val_loss,
            "lr": args.lr,
        })
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_tensor = torch.tensor(test_std, dtype=torch.float32, device=device)
    segment_scores = []
    with torch.no_grad():
        for start in range(0, test_tensor.shape[0], args.batch_size):
            logits = model(test_tensor[start:start + args.batch_size])
            segment_scores.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64))
    segment_scores = np.concatenate(segment_scores, axis=0)
    file_scores = aggregate_top_percent(segment_scores, test_offsets, args.top_percent)

    summary, tail, geom = evaluate_scores(file_scores, y_test, d_test)
    model_blob = {
        "model_state_dict": model.state_dict(),
        "mu": mu,
        "sigma": sigma,
        "machine": machine,
        "config": {
            "hidden_dim": args.hidden_dim,
            "feature_patch_frac": args.feature_patch_frac,
            "noise_std": args.noise_std,
            "top_percent": args.top_percent,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_ratio": args.val_ratio,
            "best_val_bce": best_val,
        },
    }
    score_meta = {
        "test_scores": file_scores.astype(np.float64),
        "segment_scores": segment_scores.astype(np.float64),
        "y_test": y_test.astype(np.int64),
        "d_test": d_test.astype("U16"),
        "test_paths": np.asarray(test_paths, dtype=object),
        "candidate": CANDIDATE,
        "machine": machine,
        "top_percent": float(args.top_percent),
    }
    return summary, tail, geom, file_scores, log_rows, [{"machine": machine, "best_val_bce": best_val}], model_blob, score_meta


def aggregate_top_percent(segment_scores: np.ndarray, offsets: list[dict], top_percent: float) -> np.ndarray:
    file_scores = []
    for row in offsets:
        start = int(row["segment_start"])
        end = int(row["segment_end"])
        scores = segment_scores[start:end]
        top_n = max(1, int(np.ceil(scores.size * top_percent / 100.0)))
        top = np.partition(scores, kth=max(0, scores.size - top_n))[-top_n:]
        file_scores.append(float(np.mean(top)))
    return np.asarray(file_scores, dtype=np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray) -> dict:
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95)) if len(normals) else 0.0
    q99 = float(np.quantile(normals, 0.99)) if len(normals) else 0.0
    anomaly_median = float(np.quantile(anomalies, 0.50)) if len(anomalies) else 0.0
    anomaly_q75 = float(np.quantile(anomalies, 0.75)) if len(anomalies) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": anomaly_median,
        "anomaly_q75": anomaly_q75,
        "TPR@95n": float(np.mean(anomalies > q95)) if len(anomalies) else 0.0,
        "TPR@99n": float(np.mean(anomalies > q99)) if len(anomalies) else 0.0,
    }


def evaluate_scores(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    summary = {
        "hmean": float(met["hmean"]),
        "pAUC": float(met["pauc"]),
        "KS": float(geom["source_target_ks"]),
        "TPR@95n": float(tail["TPR@95n"]),
        "TPR@99n": float(tail["TPR@99n"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "anomaly_median": float(tail["anomaly_median"]),
        "anomaly_q75": float(tail["anomaly_q75"]),
    }
    return summary, tail, geom


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    segment_root = Path(args.segment_root)
    machines = parse_csv_list(args.machines)
    rows = []
    for machine in machines:
        path = embedding_path(segment_root, machine)
        exists = path.exists()
        info = {"machine": machine, "segment_embedding_exists": exists, "path": str(path)}
        if exists:
            arr = np.load(path, allow_pickle=True)
            info["train_segment_count"] = int(arr["train_segments"].shape[0])
            info["test_segment_count"] = int(arr["test_segments"].shape[0])
            info["embedding_dim"] = int(arr["train_segments"].shape[1])
        rows.append(info)
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "segment_root_exists": segment_root.exists(),
        "clip_reference_exists": Path(args.clip_reference_per_machine).exists(),
        "machines": machines,
        "candidate": CANDIDATE,
        "notes": [
            "No model training in check mode.",
            "Stage25A trains only small boundary heads on Stage24A clean EAT segment embeddings.",
            "No anomaly labels are used for training or checkpoint selection.",
        ],
    })


def run_train_score(args):
    set_seed(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    score_root = out_root / "scores" / CANDIDATE
    score_root.mkdir(parents=True, exist_ok=True)

    segment_root = Path(args.segment_root)
    machines = parse_csv_list(args.machines)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    train_logs = []
    model_rows = []
    comparison_rows = []
    clip_refs = get_clip_reference(Path(args.clip_reference_per_machine))

    for machine in machines:
        path = embedding_path(segment_root, machine)
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage24A segment embeddings: {path}")
        arr = np.load(path, allow_pickle=True)
        train_segments = arr["train_segments"].astype(np.float32)
        test_segments = arr["test_segments"].astype(np.float32)
        test_offsets = list(arr["test_offsets"])
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype(str)
        test_paths = arr["test_paths"].astype(str)

        summary, tail, geom, _, log_rows, model_rec, model_blob, score_meta = train_one_machine(
            machine,
            train_segments,
            test_segments,
            test_offsets,
            y_test,
            d_test,
            test_paths,
            args,
            device,
        )
        per_machine_rows.append({"candidate": CANDIDATE, "machine": machine, **summary})
        tail_rows.append({"candidate": CANDIDATE, "machine": machine, **tail})
        geom_rows.append({"candidate": CANDIDATE, "machine": machine, **geom})
        train_logs.extend(log_rows)
        model_rows.extend(model_rec)
        torch.save(model_blob, out_root / "models" / f"{machine}_boundary_head.pt")
        np.savez(score_root / f"{machine}_scores.npz", **score_meta)

        ref = clip_refs.get(machine)
        comparison_rows.append({
            "machine": machine,
            "stage25_hmean": summary["hmean"],
            "stage25_pAUC": summary["pAUC"],
            "stage25_TPR99": summary["TPR@99n"],
            "clip_ldn_hmean": float(ref["hmean"]) if ref else None,
            "clip_ldn_pAUC": float(ref["pAUC"]) if ref else None,
            "clip_ldn_TPR99": float(ref["TPR@99n"]) if ref else None,
            "delta_hmean_vs_clean_clip_ldn": summary["hmean"] - float(ref["hmean"]) if ref else None,
            "delta_pAUC_vs_clean_clip_ldn": summary["pAUC"] - float(ref["pAUC"]) if ref else None,
            "delta_TPR99_vs_clean_clip_ldn": summary["TPR@99n"] - float(ref["TPR@99n"]) if ref else None,
        })

    mean_h = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_p = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_ks = float(np.mean([r["KS"] for r in per_machine_rows]))
    mean_t95 = float(np.mean([r["TPR@95n"] for r in per_machine_rows]))
    mean_t99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))
    summary_rows = [{
        "candidate": CANDIDATE,
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_KS": mean_ks,
        "mean_TPR@95n": mean_t95,
        "mean_TPR@99n": mean_t99,
    }]
    improved_count = sum(
        1 for r in comparison_rows
        if r["delta_hmean_vs_clean_clip_ldn"] is not None and r["delta_hmean_vs_clean_clip_ldn"] > 0
    )
    bottleneck_t99 = sum(
        1 for r in comparison_rows
        if r["machine"] in {"fan", "gearbox", "slider", "toycar", "toytrain"}
        and r["delta_TPR99_vs_clean_clip_ldn"] is not None
        and r["delta_TPR99_vs_clean_clip_ldn"] > 0
    )
    strong_collapse = any(
        r["machine"] in {"bearing", "toytrain", "valve"}
        and r["delta_hmean_vs_clean_clip_ldn"] is not None
        and r["delta_hmean_vs_clean_clip_ldn"] < -0.02
        for r in comparison_rows
    )
    verdict = {
        "candidate": CANDIDATE,
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_TPR99": mean_t99,
        "delta_hmean_vs_clean_ldn_best": mean_h - CONFIRMED_CLEAN_LDN_HMEAN,
        "delta_hmean_vs_stage22_hmean_best": mean_h - CONFIRMED_STAGE22_HMEAN,
        "improved_machine_count_vs_clean_clip_ldn": improved_count,
        "bottleneck_TPR99_improved_count_vs_clean_clip_ldn": bottleneck_t99,
        "strong_machine_severe_collapse": strong_collapse,
        "pseudo_boundary_positive": bool(
            mean_h > CONFIRMED_CLEAN_LDN_HMEAN
            or bottleneck_t99 >= 2
            or improved_count >= 4
        ),
        "promotable": bool(mean_h > CONFIRMED_STAGE22_HMEAN and improved_count >= 4 and not strong_collapse),
        "uses_anomaly_labels_for_training": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "notes": [
            "Trains only a small boundary head on clean EAT segment embeddings.",
            "Pseudo labels are synthetic source-normal subspace replacement labels.",
            "No EAT fine-tuning, LDN tuning, selector, or fusion.",
        ],
    }

    write_csv(out_root / "summary.csv", summary_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "train_log.csv", train_logs)
    write_csv(out_root / "model_summary.csv", model_rows)
    write_csv(out_root / "comparison_vs_clean_clip_ldn.csv", comparison_rows)
    write_json(out_root / "stage25a_pseudo_boundary_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage25A EAT Segment Pseudo-Boundary",
        "",
        f"- candidate: `{CANDIDATE}`",
        f"- mean_hmean: `{mean_h:.12f}`",
        f"- mean_pAUC: `{mean_p:.12f}`",
        f"- mean_TPR@99n: `{mean_t99:.12f}`",
        f"- delta hmean vs clean LDN best: `{verdict['delta_hmean_vs_clean_ldn_best']:.12f}`",
        f"- delta hmean vs Stage22 hmean best: `{verdict['delta_hmean_vs_stage22_hmean_best']:.12f}`",
        f"- improved machines vs clean clip LDN: `{improved_count}`",
        f"- bottleneck TPR@99n improved count: `{bottleneck_t99}`",
        f"- strong-machine severe collapse: `{strong_collapse}`",
        f"- pseudo-boundary positive: `{verdict['pseudo_boundary_positive']}`",
        f"- promotable: `{verdict['promotable']}`",
        "",
        "No anomaly labels are used for training or checkpoint selection. No EAT fine-tuning, LDN tuning, selector, or fusion is introduced.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "train_score":
        run_train_score(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
