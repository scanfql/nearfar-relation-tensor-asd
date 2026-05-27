#!/usr/bin/env python
"""Stage 16B-5A: clean EAT fixed pooling + fixed fusion audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.features.logmel import load_audio


MACHINES_ALL = ["bearing", "fan", "gearbox", "slider", "toycar", "toytrain", "valve"]
POOLINGS = ["P0_mean", "P1_mean_std", "P2_mean_max", "P3_mean_std_max"]


def parse_args():
    p = argparse.ArgumentParser(description="Stage 16B clean EAT pooling/fusion audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_pooling_embeddings", "score_pooling_knn", "fusion_pooling"])
    p.add_argument("--eat_model_path", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--machines", default="bearing,fan,gearbox,slider,toycar,toytrain,valve")
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", required=True)
    p.add_argument("--doc_path", required=True)
    p.add_argument("--v63_root", default="results/v63_fixed_v37_v61_rankfusion")
    p.add_argument("--stage14a_root", default="results/stage14a_frozen_beats_v63_fusion")
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def import_status():
    status = {}
    for name in ["torch", "torchaudio", "soundfile", "librosa", "transformers", "timm", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_clean_eat(model_path: Path, device: str):
    import torch

    package_parent = model_path.parent.resolve()
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception:
        from checkpoints.configuration_eat import EATConfig
        from checkpoints.modeling_eat import EATModel
        try:
            model = EATModel.from_pretrained(str(model_path), trust_remote_code=False)
        except Exception:
            from safetensors.torch import load_file
            cfg = EATConfig.from_pretrained(str(model_path))
            model = EATModel(cfg)
            state = load_file(str(model_path / "model.safetensors"))
            model.load_state_dict(state, strict=True)

    model.eval()
    model.to(requested)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def eat_frames(model, waveform: np.ndarray, orig_sr: int, device):
    import torch
    import torchaudio

    target_sr = 16000
    wav = torch.tensor(waveform, dtype=torch.float32)
    if orig_sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_sr, target_sr)
    final_sr = target_sr
    with torch.no_grad():
        mel = torchaudio.compliance.kaldi.fbank(
            wav.unsqueeze(0),
            htk_compat=True,
            sample_frequency=final_sr,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        target_length = 1024
        if mel.shape[0] < target_length:
            mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
        else:
            mel = mel[:target_length, :]
        norm_mean = -4.268
        norm_std = 4.569
        mel = (mel - norm_mean) / (norm_std * 2)
        x = mel.unsqueeze(0).unsqueeze(0).to(device)
        output = model(x)
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            feat = output.last_hidden_state
        elif isinstance(output, torch.Tensor):
            feat = output
        elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            feat = output[0]
        else:
            raise RuntimeError(f"Unsupported EAT output type: {type(output)}")
        if feat.ndim != 3:
            raise RuntimeError(f"Unexpected EAT output shape: {tuple(feat.shape)}")
    return feat, final_sr


def pool_frames(frames: np.ndarray):
    mean = frames.mean(axis=0)
    std = frames.std(axis=0)
    maxv = frames.max(axis=0)
    return {
        "P0_mean": mean.astype(np.float64),
        "P1_mean_std": np.concatenate([mean, std]).astype(np.float64),
        "P2_mean_max": np.concatenate([mean, maxv]).astype(np.float64),
        "P3_mean_std_max": np.concatenate([mean, std, maxv]).astype(np.float64),
    }


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
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)),
        "anomaly_q75": float(np.quantile(anomalies, 0.75)),
        "TPR@95n": float(np.mean(anomalies > q95)),
        "TPR@99n": float(np.mean(anomalies > q99)),
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray):
    met = compute_metrics(y_test, scores, domains)
    geom = compute_score_geometry(scores, y_test, domains)
    tail = tail_metrics(scores, y_test)
    return {
        "hmean": met["hmean"],
        "pAUC": met["pauc"],
        "KS": geom["source_target_ks"],
        "TPR@95n": tail["TPR@95n"],
        "TPR@99n": tail["TPR@99n"],
        "q95": tail["q95"],
        "q99": tail["q99"],
        "anomaly_median": tail["anomaly_median"],
        "anomaly_q75": tail["anomaly_q75"],
    }


def rank_norm(scores: np.ndarray) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def load_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    domains_key = "d_test" if "d_test" in arr.files else "domains"
    return {
        "test_scores": arr["test_scores"].astype(np.float64),
        "y_test": arr["y_test"].astype(np.int64),
        "domains": arr[domains_key].astype(str),
        "test_paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def expected_test_reference(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    all_items = scan_dataset(data_root) if data_root.exists() else []
    rows = []
    for machine in machines:
        items = filter_items(all_items, machine) if all_items else []
        train_items, test_items = split_train_test(items, use_target_train=False) if items else ([], [])
        rows.append({
            "machine": machine,
            "exists_in_data_root": len(items) > 0,
            "source_train_normal_count": len(train_items),
            "test_count": len(test_items),
        })
    write_csv(out_root / "machine_check.csv", rows)
    status = {
        "mode": "check",
        "eat_model_path_exists": model_path.exists(),
        "data_root_exists": data_root.exists(),
        "v63_root_exists": Path(args.v63_root).exists(),
        "stage14a_root_exists": Path(args.stage14a_root).exists(),
        "imports": import_status(),
    }
    (out_root / "check_status.json").write_text(json.dumps(status, indent=2) + "\n")


def run_extract_pooling_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model, device = load_clean_eat(Path(args.eat_model_path), args.device)
    import soundfile as sf

    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    all_items = scan_dataset(Path(args.data_root))
    emb_rows = []
    for machine in machines:
        items, train_items, test_items = expected_test_reference(Path(args.data_root), machine)
        if not train_items or not test_items:
            raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")
        train_store = {p: [] for p in POOLINGS}
        test_store = {p: [] for p in POOLINGS}
        y_test, d_test, test_paths = [], [], []
        frame_shape = None
        sample_rate = None
        for item in train_items:
            wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
            wav = wav.mean(axis=1)
            feat, final_sr = eat_frames(model, wav, int(sr), device)
            frame_shape = list(feat.shape)
            sample_rate = final_sr
            pooled = pool_frames(feat.squeeze(0).cpu().numpy())
            for k, v in pooled.items():
                train_store[k].append(v)
        for item in test_items:
            wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
            wav = wav.mean(axis=1)
            feat, final_sr = eat_frames(model, wav, int(sr), device)
            frame_shape = list(feat.shape)
            sample_rate = final_sr
            pooled = pool_frames(feat.squeeze(0).cpu().numpy())
            for k, v in pooled.items():
                test_store[k].append(v)
            y_test.append(item.label)
            d_test.append(item.domain)
            test_paths.append(str(item.path))
        y_test = np.asarray(y_test, dtype=np.int64)
        d_test = np.asarray(d_test)
        for pooling in POOLINGS:
            pool_dir = out_root / "embeddings" / pooling
            pool_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pool_dir / f"{machine}_embeddings.npz",
                train_embeddings=np.stack(train_store[pooling], axis=0).astype(np.float64),
                test_embeddings=np.stack(test_store[pooling], axis=0).astype(np.float64),
                y_test=y_test,
                d_test=d_test.astype("U16"),
                test_paths=np.asarray(test_paths, dtype=object),
                frame_output_shape=frame_shape,
                eat_model_path=str(args.eat_model_path),
            )
        emb_rows.append({
            "machine": machine,
            "frame_output_shape": frame_shape,
            "train_count": len(train_items),
            "test_count": len(test_items),
            "sample_rate": sample_rate,
            "P0_dim": 768,
            "P1_dim": 1536,
            "P2_dim": 1536,
            "P3_dim": 2304,
        })
    write_csv(out_root / "embedding_summary.csv", emb_rows)


def run_score_pooling_knn(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    pool_summary = []
    pool_pm = []
    pool_tail = []
    geom_rows = []
    for pooling in POOLINGS:
        for machine in machines:
            emb_path = out_root / "embeddings" / pooling / f"{machine}_embeddings.npz"
            if not emb_path.exists():
                raise FileNotFoundError(f"Missing embeddings: {emb_path}")
            arr = np.load(emb_path, allow_pickle=True)
            train_emb = arr["train_embeddings"].astype(np.float64)
            test_emb = arr["test_embeddings"].astype(np.float64)
            y_test = arr["y_test"].astype(np.int64)
            domains = arr["d_test"].astype(str)
            test_paths = arr["test_paths"].astype(str)
            scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
            ev = evaluate(scores, y_test, domains)
            cand = pooling
            score_dir = scores_root / cand
            score_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                score_dir / f"{machine}_scores.npz",
                test_scores=scores,
                y_test=y_test,
                d_test=domains.astype("U16"),
                test_paths=test_paths,
                score_mode=cand,
            )
            pool_pm.append({
                "machine": machine,
                "candidate": cand,
                "hmean": ev["hmean"],
                "pAUC": ev["pAUC"],
                "KS": ev["KS"],
                "TPR@95n": ev["TPR@95n"],
                "TPR@99n": ev["TPR@99n"],
            })
            pool_tail.append({
                "machine": machine,
                "candidate": cand,
                "q95": ev["q95"],
                "q99": ev["q99"],
                "anomaly_median": ev["anomaly_median"],
                "anomaly_q75": ev["anomaly_q75"],
                "TPR@95n": ev["TPR@95n"],
                "TPR@99n": ev["TPR@99n"],
            })
            geom_rows.append({"machine": machine, "candidate": cand, "KS": ev["KS"]})
    for pooling in POOLINGS:
        rows = [r for r in pool_pm if r["candidate"] == pooling]
        pool_summary.append({
            "candidate": pooling,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "pooling_summary.csv", pool_summary)
    write_csv(out_root / "pooling_per_machine.csv", pool_pm)
    write_csv(out_root / "pooling_tail_table.csv", pool_tail)
    write_csv(out_root / "score_geometry.csv", geom_rows)


def run_fusion_pooling(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    seed = args.seed
    align_rows = []
    fusion_rows = []
    fusion_pm = []
    fusion_tail = []
    comp_rows = []

    def load_existing(npz_path: Path):
        arr = np.load(npz_path, allow_pickle=True)
        dk = "d_test" if "d_test" in arr.files else "domains"
        return {
            "scores": arr["test_scores"].astype(np.float64),
            "y": arr["y_test"].astype(np.int64),
            "d": arr[dk].astype(str),
            "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
        }

    for machine in machines:
        expected_items = filter_items(scan_dataset(Path(args.data_root)), machine)
        _, test_items = split_train_test(expected_items, use_target_train=False)
        expected_paths = np.asarray([str(x.path) for x in test_items], dtype=str)
        baseline = {
            "V63": load_existing(Path(args.v63_root) / f"seed_{seed}" / "scores" / f"{machine}_full_5view_rankfusion_scores.npz"),
            "A0_raw_BEATs": load_existing(Path(args.stage14a_root) / "a0_frozen_beats" / "scores" / f"{machine}_frozen_beats_knn_scores.npz"),
            "A1": load_existing(Path(args.stage14a_root) / "a1_v63_beats_fusion" / f"seed_{seed}" / "scores" / f"{machine}_v63_beats_rankfusion_scores.npz"),
        }
        for cand, bundle in baseline.items():
            y_match = np.array_equal(bundle["y"], baseline["V63"]["y"])
            d_match = np.array_equal(bundle["d"], baseline["V63"]["d"])
            p_match = np.array_equal(bundle["paths"], expected_paths) if bundle["paths"] is not None else (y_match and d_match)
            align_rows.append({
                "machine": machine,
                "candidate": cand,
                "y_test_match": y_match,
                "domains_match": d_match,
                "sample_order_match": p_match,
                "alignment_ok": bool(y_match and d_match and p_match),
            })
            if not (y_match and d_match and p_match):
                raise RuntimeError(f"Alignment failed for {machine}/{cand}")

        for pooling in POOLINGS:
            p_npz = out_root / "scores" / pooling / f"{machine}_scores.npz"
            p_bundle = load_existing(p_npz)
            y_match = np.array_equal(p_bundle["y"], baseline["V63"]["y"])
            d_match = np.array_equal(p_bundle["d"], baseline["V63"]["d"])
            p_match = np.array_equal(p_bundle["paths"], expected_paths) if p_bundle["paths"] is not None else (y_match and d_match)
            align_rows.append({
                "machine": machine,
                "candidate": pooling,
                "y_test_match": y_match,
                "domains_match": d_match,
                "sample_order_match": p_match,
                "alignment_ok": bool(y_match and d_match and p_match),
            })
            if not (y_match and d_match and p_match):
                raise RuntimeError(f"Alignment failed for {machine}/{pooling}")
            p_scores = p_bundle["scores"]
            v63 = baseline["V63"]["scores"]
            a0 = baseline["A0_raw_BEATs"]["scores"]
            a1 = baseline["A1"]["scores"]
            y = baseline["V63"]["y"]
            d = baseline["V63"]["d"]
            candidates = {
                pooling: p_scores,
                f"F_{pooling}_A1_050": 0.5 * rank_norm(p_scores) + 0.5 * rank_norm(a1),
                f"F_{pooling}_A1_065": 0.65 * rank_norm(p_scores) + 0.35 * rank_norm(a1),
                f"F_{pooling}_V63_050": 0.5 * rank_norm(p_scores) + 0.5 * rank_norm(v63),
                f"F_{pooling}_TRI": (rank_norm(p_scores) + rank_norm(v63) + rank_norm(a0)) / 3.0,
            }
            sp_v63 = spearmanr(p_scores, v63).correlation
            sp_a0 = spearmanr(p_scores, a0).correlation
            q99_p = tail_metrics(p_scores, y)["q99"]
            q99_v63 = tail_metrics(v63, y)["q99"]
            normal_mask = y == 0
            anomaly_mask = y == 1
            top_p = set(np.where(normal_mask & (p_scores >= q99_p))[0].tolist())
            top_v63 = set(np.where(normal_mask & (v63 >= q99_v63))[0].tolist())
            missed_p = set(np.where(anomaly_mask & (p_scores <= q99_p))[0].tolist())
            missed_v63 = set(np.where(anomaly_mask & (v63 <= q99_v63))[0].tolist())
            top_union = len(top_p | top_v63)
            miss_union = len(missed_p | missed_v63)
            comp_rows.append({
                "machine": machine,
                "pooling": pooling,
                "spearman_vs_V63": float(sp_v63) if sp_v63 == sp_v63 else 0.0,
                "spearman_vs_A0_raw_BEATs": float(sp_a0) if sp_a0 == sp_a0 else 0.0,
                "top_normal_tail_jaccard": float(len(top_p & top_v63) / top_union) if top_union else 1.0,
                "missed_anomaly_jaccard": float(len(missed_p & missed_v63) / miss_union) if miss_union else 1.0,
            })
            for cand, scores in candidates.items():
                ev = evaluate(scores, y, d)
                fusion_pm.append({
                    "machine": machine,
                    "candidate": cand,
                    "hmean": ev["hmean"],
                    "pAUC": ev["pAUC"],
                    "KS": ev["KS"],
                    "TPR@95n": ev["TPR@95n"],
                    "TPR@99n": ev["TPR@99n"],
                })
                fusion_tail.append({
                    "machine": machine,
                    "candidate": cand,
                    "q95": ev["q95"],
                    "q99": ev["q99"],
                    "anomaly_median": ev["anomaly_median"],
                    "anomaly_q75": ev["anomaly_q75"],
                    "TPR@95n": ev["TPR@95n"],
                    "TPR@99n": ev["TPR@99n"],
                })
                score_dir = out_root / "scores" / cand
                score_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    score_dir / f"{machine}_scores.npz",
                    test_scores=scores,
                    y_test=y,
                    d_test=d.astype("U16"),
                    test_paths=expected_paths,
                    score_mode=cand,
                )

    # summaries
    pooling_baseline_rows = [r for r in fusion_pm if r["candidate"] in POOLINGS]
    fusion_only_rows = [r for r in fusion_pm if r["candidate"] not in POOLINGS]
    pooling_summary = []
    for cand in POOLINGS:
        rows = [r for r in pooling_baseline_rows if r["candidate"] == cand]
        pooling_summary.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    fusion_summary = []
    for cand in sorted({r["candidate"] for r in fusion_only_rows}):
        rows = [r for r in fusion_only_rows if r["candidate"] == cand]
        fusion_summary.append({
            "candidate": cand,
            "mean_hmean": float(np.mean([r["hmean"] for r in rows])),
            "mean_pAUC": float(np.mean([r["pAUC"] for r in rows])),
            "mean_KS": float(np.mean([r["KS"] for r in rows])),
            "mean_TPR@95n": float(np.mean([r["TPR@95n"] for r in rows])),
            "mean_TPR@99n": float(np.mean([r["TPR@99n"] for r in rows])),
        })
    write_csv(out_root / "pooling_summary.csv", pooling_summary)
    write_csv(out_root / "pooling_per_machine.csv", pooling_baseline_rows)
    write_csv(out_root / "pooling_tail_table.csv", [r for r in fusion_tail if r["candidate"] in POOLINGS])
    write_csv(out_root / "fusion_summary.csv", fusion_summary)
    write_csv(out_root / "fusion_per_machine.csv", fusion_only_rows)
    write_csv(out_root / "fusion_tail_table.csv", [r for r in fusion_tail if r["candidate"] not in POOLINGS])
    write_csv(out_root / "alignment_summary.csv", align_rows)
    write_csv(out_root / "complementarity_summary.csv", comp_rows)

    # baseline recap in doc
    def lookup_mean(rows, cand, key):
        match = next(r for r in rows if r["candidate"] == cand)
        return match[key]

    p0_h = lookup_mean(pooling_summary, "P0_mean", "mean_hmean")
    a1_h = 0.5813
    best_pool = max(pooling_summary, key=lambda r: r["mean_hmean"])
    best_fusion = max(fusion_summary, key=lambda r: r["mean_hmean"])

    lines = [
        "# CTAF Stage16B Clean EAT Pooling Fusion Audit",
        "",
        "## 1. Baseline Recap",
        "",
        f"- clean_EAT mean pooling hmean = `{p0_h:.6f}`",
        f"- A1 hmean ≈ `{a1_h:.4f}`",
        "- best previous fixed fusion was close to 0.6 but not stable",
        "",
        "## 2. Pooling Results",
        "",
        f"- highest-hmean pooling candidate: `{best_pool['candidate']}` with `{best_pool['mean_hmean']:.6f}`",
        "",
        "## 3. Fusion Results",
        "",
        f"- highest-hmean fusion candidate: `{best_fusion['candidate']}` with `{best_fusion['mean_hmean']:.6f}`",
        "",
        "## 4. Confirmation",
        "",
        "- no training",
        "- no fine-tuning",
        "- no LoRA",
        "- no adapter",
        "- no per-machine selection",
        "- no oracle route",
        "- no Stage16C opened automatically",
    ]
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_pooling_embeddings":
        run_extract_pooling_embeddings(args)
        return
    if args.mode == "score_pooling_knn":
        run_score_pooling_knn(args)
        return
    if args.mode == "fusion_pooling":
        run_fusion_pooling(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
