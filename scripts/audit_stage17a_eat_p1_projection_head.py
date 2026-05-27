#!/usr/bin/env python
"""Stage 17A: EAT P1_mean_std projection-head adaptation.

This script trains a lightweight projection head on existing clean EAT
P1_mean_std embeddings. It does NOT load EAT or run EAT forward.

Protocol boundary
-----------------
Allowed:
- load existing Stage16B P1_mean_std embedding caches
- train a small projection head using source train-normal embeddings only
- use machine-type labels derived from the machine loop for training
- evaluate projected embeddings with per-machine source train-normal kNN
- fixed score-level fusion with A1 / V63 scores

Blocked:
- no EAT backbone fine-tuning
- no EAT forward
- no embedding extraction
- no anomaly labels for training
- no test labels for training
- no alpha sweep
- no per-machine routing / oracle selection
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]


@dataclass
class TrainConfig:
    input_dim: int = 1536
    hidden_dim: int = 512
    proj_dim: int = 256
    dropout_p: float = 0.10
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 13711
    loss_mode: str = "ce"  # ce or ce_supcon
    supcon_weight: float = 0.05
    temperature: float = 0.10
    val_ratio: float = 0.10
    early_stop_patience: int = 15


def parse_args():
    p = argparse.ArgumentParser(description="Stage17A EAT P1 projection-head adaptation")
    p.add_argument("--mode", required=True, choices=["check", "train_projector", "eval_projector"])
    p.add_argument("--pooling_root", default="results/stage16b_clean_eat_pooling_fusion")
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--stage14a_root", default="results/stage14a_frozen_beats_v63_fusion")
    p.add_argument("--machines", default=",".join(MACHINES_ALL))
    p.add_argument("--seed", type=int, default=13711, help="projection-head training seed and A1/V63 seed for eval")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--latent_knn_k", type=int, default=5)

    # Model/training hyperparameters. Keep these fixed in a preregistered run.
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--dropout_p", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss_mode", default="ce", choices=["ce", "ce_supcon"])
    p.add_argument("--supcon_weight", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.10)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--early_stop_patience", type=int, default=15)
    p.add_argument("--ckpt_path", default="", help="Optional explicit checkpoint path for eval_projector")
    return p.parse_args()


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def import_status() -> Dict[str, bool]:
    status = {}
    for name in ["torch", "numpy", "scipy"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def p1_embedding_path(pooling_root: Path, machine: str) -> Path:
    return pooling_root / "embeddings" / "P1_mean_std" / f"{machine}_embeddings.npz"


def p1_score_path(pooling_root: Path, machine: str) -> Path:
    return pooling_root / "scores" / "P1_mean_std" / f"{machine}_scores.npz"


def a1_score_path(stage14a_root: Path, seed: int, machine: str) -> Path:
    return (
        stage14a_root
        / "a1_v63_beats_fusion"
        / f"seed_{seed}"
        / "scores"
        / f"{machine}_v63_beats_rankfusion_scores.npz"
    )


def v63_score_path(v63_root: Path, seed: int, machine: str) -> Path:
    return v63_root / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"


def load_embedding_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    arr = np.load(path, allow_pickle=True)
    required = ["train_embeddings", "test_embeddings", "y_test", "d_test", "test_paths"]
    missing = [k for k in required if k not in arr.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}; keys={arr.files}")
    return {
        "train_embeddings": arr["train_embeddings"].astype(np.float32),
        "test_embeddings": arr["test_embeddings"].astype(np.float32),
        "y_test": arr["y_test"].astype(np.int64),
        "d_test": arr["d_test"].astype(str),
        "test_paths": arr["test_paths"].astype(str),
    }


def load_score_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing score file: {path}")
    arr = np.load(path, allow_pickle=True)
    if "test_scores" in arr.files:
        scores = arr["test_scores"]
    elif "scores" in arr.files:
        scores = arr["scores"]
    else:
        raise KeyError(f"{path} has no test_scores/scores; keys={arr.files}")
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "scores": scores.astype(np.float64),
        "y": arr["y_test"].astype(np.int64),
        "d": arr[domains_key].astype(str),
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        "path": str(path),
    }


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5) -> np.ndarray:
    train_n = l2_normalize(train_emb.astype(np.float64), axis=1)
    test_n = l2_normalize(test_emb.astype(np.float64), axis=1)
    d2 = (
        np.sum(test_n * test_n, axis=1, keepdims=True)
        + np.sum(train_n * train_n, axis=1)[None, :]
        - 2.0 * (test_n @ train_n.T)
    )
    d2 = np.maximum(d2, 0.0)
    k = min(k, train_n.shape[0])
    nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    return np.sqrt(nearest).mean(axis=1).astype(np.float64)


def tail_metrics(scores: np.ndarray, y_test: np.ndarray) -> dict:
    normals = scores[y_test == 0]
    anomalies = scores[y_test == 1]
    q95 = float(np.quantile(normals, 0.95))
    q99 = float(np.quantile(normals, 0.99))
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "TPR@95n": float(np.mean(anomalies > q95)),
        "TPR@99n": float(np.mean(anomalies > q99)),
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> dict:
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    return {
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


def make_device(device_arg: str):
    import torch

    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    return dev


def choose_split_indices(y: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio))) if len(idx) > 1 else 0
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.concatenate(train_parts, axis=0)
    val_idx = np.concatenate(val_parts, axis=0) if val_parts else np.asarray([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def load_all_embeddings(pooling_root: Path, machines: List[str]) -> Tuple[dict, np.ndarray, np.ndarray]:
    bundles = {}
    x_train_all = []
    y_train_all = []
    for midx, machine in enumerate(machines):
        bundle = load_embedding_bundle(p1_embedding_path(pooling_root, machine))
        bundles[machine] = bundle
        x_train_all.append(bundle["train_embeddings"])
        y_train_all.append(np.full((bundle["train_embeddings"].shape[0],), midx, dtype=np.int64))
    return bundles, np.concatenate(x_train_all, axis=0), np.concatenate(y_train_all, axis=0)


def build_model(input_dim: int, hidden_dim: int, proj_dim: int, num_classes: int, dropout_p: float):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProjectionHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, proj_dim),
            )
            self.classifier = nn.Linear(proj_dim, num_classes)

        def forward(self, x):
            z = self.net(x)
            z = F.normalize(z, dim=-1)
            logits = self.classifier(z)
            return z, logits

    return ProjectionHead()


def supervised_contrastive_loss(z, labels, temperature: float = 0.10):
    import torch
    import torch.nn.functional as F

    z = F.normalize(z, dim=1)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    logits = (z @ z.T) / temperature

    # Remove self-contrast.
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=z.device)
    mask = mask * logits_mask

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

    positive_count = mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return torch.zeros([], device=z.device)
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / positive_count.clamp_min(1.0)
    return -mean_log_prob_pos[valid].mean()


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = parse_csv_list(args.machines)
    pooling_root = Path(args.pooling_root)
    stage14a_root = Path(args.stage14a_root)
    v63_root = Path(args.v63_root)

    rows = []
    missing = []
    for machine in machines:
        paths = {
            "P1_embedding": p1_embedding_path(pooling_root, machine),
            "P1_score": p1_score_path(pooling_root, machine),
            "A1_score": a1_score_path(stage14a_root, args.seed, machine),
            "V63_score": v63_score_path(v63_root, args.seed, machine),
        }
        row = {"machine": machine}
        try:
            emb = load_embedding_bundle(paths["P1_embedding"])
            row.update({
                "train_count": int(emb["train_embeddings"].shape[0]),
                "test_count": int(emb["test_embeddings"].shape[0]),
                "input_dim": int(emb["train_embeddings"].shape[1]),
            })
        except Exception as exc:
            row.update({"train_count": 0, "test_count": 0, "input_dim": -1, "embedding_error": str(exc)})
        for key, path in paths.items():
            exists = path.exists()
            row[f"{key}_exists"] = exists
            row[f"{key}_path"] = str(path)
            if not exists:
                missing.append({"machine": machine, "item": key, "path": str(path)})
        rows.append(row)

    write_csv(out_root / "check_files.csv", rows)
    write_csv(out_root / "missing_files.csv", missing)
    write_json(out_root / "check_status.json", {
        "mode": "check",
        "machines": machines,
        "seed": args.seed,
        "pooling_root_exists": pooling_root.exists(),
        "stage14a_root_exists": stage14a_root.exists(),
        "v63_root_exists": v63_root.exists(),
        "imports": import_status(),
        "num_missing_files": len(missing),
        "all_required_files_present": len(missing) == 0,
        "notes": "No training, no scoring, no model forward, no EAT loading.",
    })


def run_train_projector(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(args.seed)
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    machines = parse_csv_list(args.machines)
    bundles, x_all, y_all = load_all_embeddings(Path(args.pooling_root), machines)
    input_dim = int(x_all.shape[1])
    cfg = TrainConfig(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        proj_dim=args.proj_dim,
        dropout_p=args.dropout_p,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        loss_mode=args.loss_mode,
        supcon_weight=args.supcon_weight,
        temperature=args.temperature,
        val_ratio=args.val_ratio,
        early_stop_patience=args.early_stop_patience,
    )

    train_idx, val_idx = choose_split_indices(y_all, cfg.val_ratio, cfg.seed)
    device = make_device(args.device)

    model = build_model(cfg.input_dim, cfg.hidden_dim, cfg.proj_dim, len(machines), cfg.dropout_p).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_train = torch.tensor(x_all[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_all[train_idx], dtype=torch.long)
    x_val = torch.tensor(x_all[val_idx], dtype=torch.float32) if len(val_idx) else torch.empty(0, cfg.input_dim)
    y_val = torch.tensor(y_all[val_idx], dtype=torch.long) if len(val_idx) else torch.empty(0, dtype=torch.long)

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 0
    log_rows = []
    best_path = ckpt_dir / f"stage17a_projector_{cfg.loss_mode}_seed{cfg.seed}.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_sc = 0.0
        total_correct = 0
        total_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            z, logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            sc = supervised_contrastive_loss(z, yb, cfg.temperature) if cfg.loss_mode == "ce_supcon" else torch.zeros([], device=device)
            loss = ce + cfg.supcon_weight * sc
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            total_loss += float(loss.item()) * xb.shape[0]
            total_ce += float(ce.item()) * xb.shape[0]
            total_sc += float(sc.item()) * xb.shape[0]
            total_correct += int((logits.argmax(dim=1) == yb).sum().item())
            total_count += int(xb.shape[0])

        model.eval()
        with torch.no_grad():
            if len(val_idx):
                zv, logits_v = model(x_val.to(device))
                val_ce = F.cross_entropy(logits_v, y_val.to(device))
                val_acc = float((logits_v.argmax(dim=1).cpu() == y_val).float().mean().item())
                val_loss = float(val_ce.item())
            else:
                val_loss = total_loss / max(total_count, 1)
                val_acc = 0.0

        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_count, 1),
            "train_ce": total_ce / max(total_count, 1),
            "train_supcon": total_sc / max(total_count, 1),
            "train_acc": total_correct / max(total_count, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": opt.param_groups[0]["lr"],
        }
        log_rows.append(row)

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": asdict(cfg),
                "machines": machines,
                "machine_to_label": {m: i for i, m in enumerate(machines)},
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "protocol": "Stage17A projection head on frozen clean EAT P1_mean_std embeddings; no EAT fine-tuning.",
            }, best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "train_summary.json", {
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "config": asdict(cfg),
        "machines": machines,
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "notes": "Projection head trained on train-normal P1_mean_std embeddings only. No anomaly labels or test labels used for training.",
    })


def load_projector_for_eval(ckpt_path: Path, device):
    import torch

    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg = ckpt["config"]
    machines = ckpt["machines"]
    model = build_model(
        int(cfg["input_dim"]),
        int(cfg["hidden_dim"]),
        int(cfg["proj_dim"]),
        len(machines),
        float(cfg["dropout_p"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def project_numpy(model, x: np.ndarray, device, batch_size: int = 1024) -> np.ndarray:
    import torch

    outs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            z, _ = model(xb)
            outs.append(z.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outs, axis=0)


def check_alignment(machine: str, candidate: str, ref_y: np.ndarray, ref_d: np.ndarray, ref_paths: np.ndarray, bundle: dict) -> dict:
    y_match = np.array_equal(ref_y, bundle["y"])
    d_match = np.array_equal(ref_d, bundle["d"])
    if bundle["paths"] is not None:
        p_match = np.array_equal(ref_paths, bundle["paths"])
    else:
        p_match = bool(y_match and d_match)
    return {
        "machine": machine,
        "candidate": candidate,
        "score_path": bundle.get("path", ""),
        "y_test_match": y_match,
        "domains_match": d_match,
        "sample_order_match": p_match,
        "alignment_ok": bool(y_match and d_match and p_match),
    }


def summarize(rows: List[dict], candidate_key: str = "candidate") -> List[dict]:
    out = []
    for cand in sorted({r[candidate_key] for r in rows}):
        rr = [r for r in rows if r[candidate_key] == cand]
        out.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rr])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rr])),
            "mean_KS": float(np.mean([r["KS"] for r in rr])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rr])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rr])),
        })
    return out


def run_eval_projector(args):
    import torch

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    machines = parse_csv_list(args.machines)
    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else out_root / "checkpoints" / f"stage17a_projector_{args.loss_mode}_seed{args.seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing projector checkpoint: {ckpt_path}")

    device = make_device(args.device)
    model, ckpt = load_projector_for_eval(ckpt_path, device)

    per_machine_rows = []
    tail_rows = []
    align_rows = []

    for machine in machines:
        emb = load_embedding_bundle(p1_embedding_path(Path(args.pooling_root), machine))
        train_proj = project_numpy(model, emb["train_embeddings"], device)
        test_proj = project_numpy(model, emb["test_embeddings"], device)
        y = emb["y_test"]
        d = emb["d_test"]
        paths = emb["test_paths"]

        proj_scores = knn_cosine_scores(train_proj, test_proj, k=args.latent_knn_k)

        p1 = load_score_bundle(p1_score_path(Path(args.pooling_root), machine))
        a1 = load_score_bundle(a1_score_path(Path(args.stage14a_root), args.seed, machine))
        v63 = load_score_bundle(v63_score_path(Path(args.v63_root), args.seed, machine))

        for cand_name, bundle in {"P1_mean_std": p1, "A1": a1, "V63": v63}.items():
            row = check_alignment(machine, cand_name, y, d, paths, bundle)
            align_rows.append(row)
            if not row["alignment_ok"]:
                raise RuntimeError(f"Alignment failed for {machine}/{cand_name}: {row}")

        candidates = {
            "P1_mean_std": p1["scores"],
            "A1": a1["scores"],
            "V63": v63["scores"],
            "P1_proj_CE" if ckpt["config"]["loss_mode"] == "ce" else "P1_proj_CE_SupCon": proj_scores,
        }
        proj_name = "P1_proj_CE" if ckpt["config"]["loss_mode"] == "ce" else "P1_proj_CE_SupCon"
        candidates[f"F_{proj_name}_A1_065"] = 0.65 * rank_norm(proj_scores) + 0.35 * rank_norm(a1["scores"])
        candidates[f"F_{proj_name}_A1_050"] = 0.5 * rank_norm(proj_scores) + 0.5 * rank_norm(a1["scores"])
        candidates[f"F_{proj_name}_V63_050"] = 0.5 * rank_norm(proj_scores) + 0.5 * rank_norm(v63["scores"])
        candidates["F_P1_A1_065_reference"] = 0.65 * rank_norm(p1["scores"]) + 0.35 * rank_norm(a1["scores"])

        projected_dir = out_root / "projected_embeddings"
        projected_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            projected_dir / f"{machine}_{proj_name}_embeddings.npz",
            train_embeddings=train_proj.astype(np.float64),
            test_embeddings=test_proj.astype(np.float64),
            y_test=y.astype(np.int64),
            d_test=d.astype("U16"),
            test_paths=np.asarray(paths, dtype=object),
            ckpt_path=str(ckpt_path),
            score_mode=proj_name,
        )

        for cand, scores in candidates.items():
            ev = evaluate(scores, y, d)
            per_machine_rows.append({
                "machine": machine,
                "candidate": cand,
                "hmean": ev["hmean"],
                "pAUC": ev["pAUC"],
                "KS": ev["KS"],
                "TPR@95n": ev["TPR@95n"],
                "TPR@99n": ev["TPR@99n"],
            })
            tail_rows.append({
                "machine": machine,
                "candidate": cand,
                "q95": ev["q95"],
                "q99": ev["q99"],
                "anomaly_median": ev["anomaly_median"],
                "anomaly_q75": ev["anomaly_q75"],
                "TPR@95n": ev["TPR@95n"],
                "TPR@99n": ev["TPR@99n"],
            })
            score_dir = scores_root / cand
            score_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                score_dir / f"{machine}_scores.npz",
                test_scores=scores.astype(np.float64),
                y_test=y.astype(np.int64),
                d_test=d.astype("U16"),
                test_paths=np.asarray(paths, dtype=object),
                score_mode=cand,
                seed=args.seed,
                machine=machine,
            )

    write_csv(out_root / "alignment_summary.csv", align_rows)
    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    summary = summarize(per_machine_rows)
    write_csv(out_root / "summary.csv", summary)

    best = max(summary, key=lambda r: r["mean_hmean"])
    ref = next((r for r in summary if r["candidate"] == "F_P1_A1_065_reference"), None)
    gain_vs_ref = None if ref is None else float(best["mean_hmean"] - ref["mean_hmean"])
    write_json(out_root / "eval_verdict.json", {
        "best_candidate": best["candidate"],
        "best_mean_hmean": best["mean_hmean"],
        "best_mean_pAUC": best["mean_pAUC"],
        "best_mean_TPR@99n": best["mean_TPR@99n"],
        "reference_candidate": "F_P1_A1_065_reference",
        "reference_mean_hmean": None if ref is None else ref["mean_hmean"],
        "hmean_gain_vs_reference": gain_vs_ref,
        "stage17a_promising": bool(gain_vs_ref is not None and gain_vs_ref >= 0.010),
        "notes": "Single-seed projection-head adaptation audit. Do not freeze without confirmation.",
    })

    doc_lines = [
        "# CTAF Stage17A EAT P1 Projection Head Audit",
        "",
        "## 1. Protocol",
        "",
        "- Frozen clean EAT P1_mean_std embeddings are used as input.",
        "- A lightweight projection head is trained on source train-normal embeddings only.",
        "- EAT backbone is not loaded or fine-tuned.",
        "- Test labels are used only for reporting metrics after scoring.",
        "",
        "## 2. Training",
        "",
        f"- checkpoint: `{ckpt_path}`",
        f"- loss_mode: `{ckpt['config']['loss_mode']}`",
        f"- best_epoch: `{ckpt.get('best_epoch')}`",
        f"- best_val_loss: `{ckpt.get('best_val_loss')}`",
        "",
        "## 3. Evaluation Summary",
        "",
        f"- best candidate: `{best['candidate']}`",
        f"- best mean hmean: `{best['mean_hmean']:.6f}`",
        f"- best mean pAUC: `{best['mean_pAUC']:.6f}`",
        f"- best mean TPR@99n: `{best['mean_TPR@99n']:.6f}`",
        "",
        "## 4. Confirmation",
        "",
        "- no EAT forward",
        "- no EAT fine-tuning",
        "- no LoRA",
        "- no adapter",
        "- no anomaly-label training",
        "- no alpha sweep",
        "- no per-machine oracle route",
    ]
    Path(args.doc_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.doc_path).write_text("\n".join(doc_lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "train_projector":
        run_train_projector(args)
        return
    if args.mode == "eval_projector":
        run_eval_projector(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
