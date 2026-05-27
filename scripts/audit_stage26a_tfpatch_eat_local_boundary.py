#!/usr/bin/env python
"""Stage26A: real TF-patch pseudo-boundary before frozen EAT.

This is the first post-Stage25 local-evidence training probe that generates
pseudo anomalies before EAT, in mel/TF space rather than embedding space.

Protocol:
- source train-normal only
- frozen clean EAT
- same-machine mel TF patch replacement
- train a small local boundary head on EAT frame chunks
- clean chunks label 0
- pseudo chunks overlapping the replaced TF time patch label 1
- checkpoint selection uses clean/pseudo validation BCE only
- inference file score = top 5% chunk pseudo-anomaly probability

No anomaly labels for training/checkpoint selection, no EAT fine-tuning, no LDN
tuning, no selector, no fusion.
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
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry

from audit_stage21e_adapted_eval import (
    extract_frame_output,
    load_clean_eat_local,
    load_waveform_16k,
    min_clamp,
    waveform_to_eat_tensor,
)


MACHINES_DEFAULT = "fan,valve"
MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
EAT_CKPT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage26a_tfpatch_eat_local_boundary"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE26A_TFPATCH_EAT_LOCAL_BOUNDARY.md"
CLIP_REFERENCE_DEFAULT = "results/stage18e_local_density_protocol_full7/per_machine.csv"

CANDIDATE = "tfpatch_eat_local_boundary_top5pct"
CLIP_REFERENCE_CANDIDATE = "ldn_diff_K1_scorek5"
CONFIRMED_STAGE22_HMEAN = 0.6166917954459288
CONFIRMED_CLEAN_LDN_HMEAN = 0.6159467288425517


def parse_args():
    p = argparse.ArgumentParser(description="Stage26A TF-patch EAT local boundary")
    p.add_argument("--mode", required=True, choices=["check", "train_score"])
    p.add_argument("--eat_model_path", default=EAT_CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--clip_reference_per_machine", default=CLIP_REFERENCE_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--segment_len", type=int, default=4)
    p.add_argument("--segment_stride", type=int, default=2)
    p.add_argument("--patch_time_frac", type=float, default=0.08)
    p.add_argument("--patch_freq_frac", type=float, default=0.10)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--top_percent", type=float, default=5.0)
    p.add_argument("--positive_loss_weight", type=float, default=2.0)
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


def import_status():
    status = {}
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


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


class LocalBoundaryHead(nn.Module):
    def __init__(self, in_dim: int = 768, hidden_dim: int = 128):
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


def split_train_val(rows: list, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rows))
    val_n = max(1, int(len(rows) * val_ratio))
    val_n = min(val_n, len(rows) - 1)
    val_idx = set(int(i) for i in idx[:val_n])
    train_rows = [r for i, r in enumerate(rows) if i not in val_idx]
    val_rows = [r for i, r in enumerate(rows) if i in val_idx]
    return train_rows, val_rows


def load_mel_for_item(item, device):
    wav = load_waveform_16k(Path(item.path))
    return waveform_to_eat_tensor(wav).to(device)


def make_pseudo_mel_batch(clean_mel, rng: np.random.Generator, patch_time_frac: float, patch_freq_frac: float):
    bsz, t, f = clean_mel.shape
    donor_idx = np.arange(bsz)
    rng.shuffle(donor_idx)
    if bsz > 1:
        for i in range(bsz):
            if donor_idx[i] == i:
                donor_idx[i] = (i + 1) % bsz
    pseudo = clean_mel.clone()
    t_len = max(1, int(t * patch_time_frac))
    f_len = max(1, int(f * patch_freq_frac))
    patch_rows = []
    for i in range(bsz):
        t0 = int(rng.integers(0, t - t_len + 1))
        f0 = int(rng.integers(0, f - f_len + 1))
        pseudo[i, t0:t0 + t_len, f0:f0 + f_len] = clean_mel[int(donor_idx[i]), t0:t0 + t_len, f0:f0 + f_len]
        patch_rows.append({"t0": t0, "t1": t0 + t_len, "f0": f0, "f1": f0 + f_len})
    return pseudo, patch_rows


def chunk_frames_with_labels(frames, patch_rows, mel_t: int, segment_len: int, stride: int):
    chunks = []
    labels = []
    bsz, out_t, dim = frames.shape
    for b in range(bsz):
        patch = patch_rows[b]
        out_p0 = int(np.floor(patch["t0"] / mel_t * out_t))
        out_p1 = int(np.ceil(patch["t1"] / mel_t * out_t))
        out_p1 = max(out_p0 + 1, min(out_t, out_p1))
        starts = list(range(0, max(1, out_t - segment_len + 1), stride))
        if not starts or starts[-1] != max(0, out_t - segment_len):
            starts.append(max(0, out_t - segment_len))
        for start in starts:
            end = min(out_t, start + segment_len)
            chunk = frames[b, start:end].mean(dim=0)
            overlap = max(0, min(end, out_p1) - max(start, out_p0))
            label = 1.0 if overlap > 0 else 0.0
            chunks.append(chunk)
            labels.append(label)
    return torch.stack(chunks, dim=0), torch.tensor(labels, dtype=torch.float32, device=frames.device)


def chunk_frames_no_labels(frames, segment_len: int, stride: int):
    chunks = []
    offsets = []
    bsz, out_t, _ = frames.shape
    for b in range(bsz):
        start_idx = len(chunks)
        starts = list(range(0, max(1, out_t - segment_len + 1), stride))
        if not starts or starts[-1] != max(0, out_t - segment_len):
            starts.append(max(0, out_t - segment_len))
        for start in starts:
            end = min(out_t, start + segment_len)
            chunks.append(frames[b, start:end].mean(dim=0))
        offsets.append((start_idx, len(chunks)))
    return torch.stack(chunks, dim=0), offsets


def extract_clean_chunks(model, mel_batch, device, segment_len: int, stride: int):
    with torch.no_grad():
        frames = extract_frame_output(model, mel_batch.to(device), device)
        frames = torch.where(torch.abs(frames) < 0.1, torch.zeros_like(frames), frames)
        chunks, offsets = chunk_frames_no_labels(frames, segment_len, stride)
    return chunks.detach(), offsets


def batch_items_to_mel(items, device):
    return torch.stack([load_mel_for_item(item, device) for item in items], dim=0)


def compute_train_loss(model, head, mel_batch, rng, args, device):
    clean_mel = mel_batch.to(device)
    pseudo_mel, patch_rows = make_pseudo_mel_batch(clean_mel, rng, args.patch_time_frac, args.patch_freq_frac)
    with torch.no_grad():
        clean_frames = extract_frame_output(model, clean_mel, device)
        pseudo_frames = extract_frame_output(model, pseudo_mel, device)
        clean_frames = torch.where(torch.abs(clean_frames) < 0.1, torch.zeros_like(clean_frames), clean_frames)
        pseudo_frames = torch.where(torch.abs(pseudo_frames) < 0.1, torch.zeros_like(pseudo_frames), pseudo_frames)
        clean_chunks, _ = chunk_frames_no_labels(clean_frames, args.segment_len, args.segment_stride)
        clean_labels = torch.zeros(clean_chunks.shape[0], dtype=torch.float32, device=device)
        pseudo_chunks, pseudo_labels = chunk_frames_with_labels(
            pseudo_frames,
            patch_rows,
            mel_t=clean_mel.shape[1],
            segment_len=args.segment_len,
            stride=args.segment_stride,
        )
        x = torch.cat([clean_chunks, pseudo_chunks], dim=0).detach()
        y = torch.cat([clean_labels, pseudo_labels], dim=0)
    logits = head(x)
    pos_weight = torch.tensor(args.positive_loss_weight, dtype=torch.float32, device=device)
    return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight), {
        "positive_chunk_fraction": float(y.mean().detach().cpu().item()),
        "chunk_count": int(y.numel()),
    }


def aggregate_top_percent(segment_scores: np.ndarray, offsets: list[tuple[int, int]], top_percent: float):
    file_scores = []
    for start, end in offsets:
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


def score_test_items(model, head, test_items, args, device):
    head.eval()
    y_test = []
    d_test = []
    test_paths = []
    all_segment_scores = []
    all_offsets = []
    with torch.no_grad():
        for start in range(0, len(test_items), args.batch_size):
            batch = test_items[start:start + args.batch_size]
            mel = batch_items_to_mel(batch, device)
            chunks, offsets = extract_clean_chunks(model, mel, device, args.segment_len, args.segment_stride)
            logits = head(chunks.to(device))
            scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
            base = len(all_segment_scores)
            all_segment_scores.extend(scores.tolist())
            all_offsets.extend([(base + s, base + e) for s, e in offsets])
            for item in batch:
                y_test.append(int(item.label))
                d_test.append(item.domain)
                test_paths.append(str(item.path))
    segment_scores = np.asarray(all_segment_scores, dtype=np.float64)
    file_scores = aggregate_top_percent(segment_scores, all_offsets, args.top_percent)
    return file_scores, segment_scores, np.asarray(y_test, dtype=np.int64), np.asarray(d_test, dtype="U16"), np.asarray(test_paths, dtype=object)


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    machines = parse_csv_list(args.machines)
    items = scan_dataset(data_root) if data_root.exists() else []
    rows = []
    for machine in machines:
        machine_items = filter_items(items, machine) if items else []
        train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
        rows.append({
            "machine": machine,
            "source_train_normal_count": len(train_items),
            "test_count": len(test_items),
        })
    write_csv(out_root / "check_machine_inventory.csv", rows)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "eat_ckpt_exists": Path(args.eat_model_path).exists(),
        "data_root_exists": data_root.exists(),
        "clip_reference_exists": Path(args.clip_reference_per_machine).exists(),
        "machines": machines,
        "imports": import_status(),
        "notes": [
            "No model load, no forward, no training.",
            "Stage26A trains only a small head on frozen EAT local chunks.",
        ],
    })


def run_train_score(args):
    set_seed(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    score_root = out_root / "scores" / CANDIDATE
    score_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    machines = parse_csv_list(args.machines)
    if not all(m in MACHINES_ALL for m in machines):
        raise ValueError(f"Unknown machine in {machines}; known={MACHINES_ALL}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    eat_model, device = load_clean_eat_local(Path(args.eat_model_path), str(device))
    for param in eat_model.parameters():
        param.requires_grad_(False)
    eat_model.eval()

    all_items = scan_dataset(data_root)
    clip_refs = get_clip_reference(Path(args.clip_reference_per_machine))
    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    train_logs = []
    comparison_rows = []

    for machine in machines:
        machine_items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(machine_items, use_target_train=False)
        train_rows, val_rows = split_train_val(train_items, args.val_ratio, args.seed)
        if not train_rows or not val_rows:
            raise RuntimeError(f"Invalid split for {machine}: train={len(train_rows)} val={len(val_rows)}")

        head = LocalBoundaryHead(768, args.hidden_dim).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float("inf")
        best_state = None
        rng = np.random.default_rng(args.seed)

        for epoch in range(1, args.epochs + 1):
            head.train()
            order = rng.permutation(len(train_rows))
            train_loss_sum = 0.0
            train_chunks = 0
            train_pos_sum = 0.0
            for start in range(0, len(order), args.batch_size):
                batch = [train_rows[int(i)] for i in order[start:start + args.batch_size]]
                mel = batch_items_to_mel(batch, device)
                opt.zero_grad(set_to_none=True)
                loss, meta = compute_train_loss(eat_model, head, mel, rng, args, device)
                loss.backward()
                opt.step()
                train_loss_sum += float(loss.item()) * meta["chunk_count"]
                train_chunks += meta["chunk_count"]
                train_pos_sum += meta["positive_chunk_fraction"] * meta["chunk_count"]

            head.eval()
            val_loss_sum = 0.0
            val_chunks = 0
            val_pos_sum = 0.0
            with torch.no_grad():
                for start in range(0, len(val_rows), args.batch_size):
                    batch = val_rows[start:start + args.batch_size]
                    mel = batch_items_to_mel(batch, device)
                    loss, meta = compute_train_loss(eat_model, head, mel, rng, args, device)
                    val_loss_sum += float(loss.item()) * meta["chunk_count"]
                    val_chunks += meta["chunk_count"]
                    val_pos_sum += meta["positive_chunk_fraction"] * meta["chunk_count"]

            train_loss = train_loss_sum / max(train_chunks, 1)
            val_loss = val_loss_sum / max(val_chunks, 1)
            train_logs.append({
                "machine": machine,
                "epoch": epoch,
                "train_bce": train_loss,
                "val_bce": val_loss,
                "train_positive_chunk_fraction": train_pos_sum / max(train_chunks, 1),
                "val_positive_chunk_fraction": val_pos_sum / max(val_chunks, 1),
            })
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

        if best_state is not None:
            head.load_state_dict(best_state)

        file_scores, segment_scores, y_test, d_test, test_paths = score_test_items(eat_model, head, test_items, args, device)
        summary, tail, geom = evaluate_scores(file_scores, y_test, d_test)
        per_machine_rows.append({"candidate": CANDIDATE, "machine": machine, **summary})
        tail_rows.append({"candidate": CANDIDATE, "machine": machine, **tail})
        geom_rows.append({"candidate": CANDIDATE, "machine": machine, **geom})
        torch.save({
            "head_state_dict": head.state_dict(),
            "machine": machine,
            "best_val_bce": best_val,
            "config": vars(args),
        }, out_root / "models" / f"{machine}_tfpatch_local_head.pt")
        np.savez(
            score_root / f"{machine}_scores.npz",
            test_scores=file_scores.astype(np.float64),
            segment_scores=segment_scores.astype(np.float64),
            y_test=y_test,
            d_test=d_test,
            test_paths=test_paths,
            candidate=CANDIDATE,
            machine=machine,
        )

        ref = clip_refs.get(machine)
        comparison_rows.append({
            "machine": machine,
            "stage26_hmean": summary["hmean"],
            "stage26_pAUC": summary["pAUC"],
            "stage26_TPR99": summary["TPR@99n"],
            "clip_ldn_hmean": float(ref["hmean"]) if ref else None,
            "clip_ldn_pAUC": float(ref["pAUC"]) if ref else None,
            "clip_ldn_TPR99": float(ref["TPR@99n"]) if ref else None,
            "delta_hmean_vs_clean_clip_ldn": summary["hmean"] - float(ref["hmean"]) if ref else None,
            "delta_pAUC_vs_clean_clip_ldn": summary["pAUC"] - float(ref["pAUC"]) if ref else None,
            "delta_TPR99_vs_clean_clip_ldn": summary["TPR@99n"] - float(ref["TPR@99n"]) if ref else None,
            "best_val_bce": best_val,
        })

    mean_h = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_p = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_t99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))
    mean_ks = float(np.mean([r["KS"] for r in per_machine_rows]))
    improved_count = sum(
        1 for r in comparison_rows
        if r["delta_hmean_vs_clean_clip_ldn"] is not None and r["delta_hmean_vs_clean_clip_ldn"] > 0
    )
    gate_pass = False
    if set(machines) == {"fan", "valve"}:
        fan = next(r for r in comparison_rows if r["machine"] == "fan")
        valve = next(r for r in comparison_rows if r["machine"] == "valve")
        gate_pass = bool(fan["delta_hmean_vs_clean_clip_ldn"] > 0.01 and valve["delta_hmean_vs_clean_clip_ldn"] > -0.02)

    verdict = {
        "candidate": CANDIDATE,
        "machines": machines,
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_TPR99": mean_t99,
        "delta_hmean_vs_clean_ldn_best": mean_h - CONFIRMED_CLEAN_LDN_HMEAN if len(machines) == 7 else None,
        "delta_hmean_vs_stage22_hmean_best": mean_h - CONFIRMED_STAGE22_HMEAN if len(machines) == 7 else None,
        "improved_machine_count_vs_clean_clip_ldn": improved_count,
        "fan_valve_gate_pass": gate_pass,
        "uses_anomaly_labels_for_training": False,
        "uses_test_labels_for_checkpoint_selection": False,
        "notes": [
            "Frozen clean EAT.",
            "Small local boundary head only.",
            "TF patch pseudo labels are generated from source train-normal mel tensors.",
            "No LDN tuning, selector, or fusion.",
        ],
    }

    write_csv(out_root / "summary.csv", [{
        "candidate": CANDIDATE,
        "mean_hmean": mean_h,
        "mean_pAUC": mean_p,
        "mean_KS": mean_ks,
        "mean_TPR@99n": mean_t99,
    }])
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "train_log.csv", train_logs)
    write_csv(out_root / "comparison_vs_clean_clip_ldn.csv", comparison_rows)
    write_json(out_root / "stage26a_tfpatch_local_boundary_verdict.json", verdict)
    write_markdown(Path(args.doc_path), [
        "# CTAF Stage26A TF-Patch EAT Local Boundary",
        "",
        f"- candidate: `{CANDIDATE}`",
        f"- machines: `{','.join(machines)}`",
        f"- mean_hmean: `{mean_h:.12f}`",
        f"- mean_pAUC: `{mean_p:.12f}`",
        f"- mean_TPR@99n: `{mean_t99:.12f}`",
        f"- improved machines vs clean clip LDN: `{improved_count}`",
        f"- fan/valve gate pass: `{gate_pass}`",
        "",
        "No anomaly labels, EAT fine-tuning, LDN tuning, selector, or fusion were used.",
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
