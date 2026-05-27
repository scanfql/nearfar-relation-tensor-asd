#!/usr/bin/env python
"""Stage 16B-2: clean EAT one-machine frozen kNN sanity."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import filter_items, scan_dataset, split_train_test
from src.evaluation.metrics import compute_metrics
from src.evaluation.score_geometry import compute_score_geometry
from src.features.logmel import load_audio


def parse_args():
    p = argparse.ArgumentParser(description="Stage 16B clean EAT one-machine kNN sanity")
    p.add_argument("--mode", required=True, choices=["check", "one_machine_knn"])
    p.add_argument("--eat_model_path", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--machine", default="gearbox")
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

    # Try local-path AutoModel as requested, then fallback to local module import.
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


def eat_forward(model, waveform: np.ndarray, orig_sr: int, device):
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
        emb = feat.mean(dim=1)  # [B, D]
    return feat, emb, final_sr


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


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
    items = scan_dataset(data_root) if data_root.exists() else []
    machine_items = filter_items(items, args.machine) if items else []
    train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
    result = {
        "mode": "check",
        "eat_model_path_exists": model_path.exists(),
        "data_root_exists": data_root.exists(),
        "requested_machine": args.machine,
        "machine_exists": len(machine_items) > 0,
        "source_train_normal_count": len(train_items),
        "test_count": len(test_items),
        "imports": import_status(),
    }
    (out_root / "check_status.json").write_text(json.dumps(result, indent=2) + "\n")


def run_one_machine_knn(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_dir = out_root / "embeddings"
    scores_dir = out_root / "scores"
    emb_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    model, device = load_clean_eat(Path(args.eat_model_path), args.device)
    items = filter_items(scan_dataset(Path(args.data_root)), args.machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    if not train_items or not test_items:
        raise RuntimeError(f"Machine {args.machine}: train={len(train_items)}, test={len(test_items)}")

    train_embs = []
    test_embs = []
    sample_feat_shape = None
    sample_final_sr = None
    for item in train_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        feat, emb, final_sr = eat_forward(model, wav, int(sr), device)
        train_embs.append(emb.squeeze(0).cpu().numpy())
        sample_feat_shape = list(feat.shape)
        sample_final_sr = final_sr
    y_test = []
    d_test = []
    test_paths = []
    for item in test_items:
        wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)
        feat, emb, final_sr = eat_forward(model, wav, int(sr), device)
        test_embs.append(emb.squeeze(0).cpu().numpy())
        y_test.append(item.label)
        d_test.append(item.domain)
        test_paths.append(str(item.path))
        sample_feat_shape = list(feat.shape)
        sample_final_sr = final_sr

    train_emb = np.stack(train_embs, axis=0).astype(np.float64)
    test_emb = np.stack(test_embs, axis=0).astype(np.float64)
    y_test = np.asarray(y_test, dtype=np.int64)
    d_test = np.asarray(d_test)

    scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
    metrics = compute_metrics(y_test, scores, d_test)
    geometry = compute_score_geometry(scores, y_test, d_test)
    tail = tail_metrics(scores, y_test)

    np.savez_compressed(
        emb_dir / f"{args.machine}_clean_eat_embeddings.npz",
        train_embeddings=train_emb,
        test_embeddings=test_emb,
        y_test=y_test,
        d_test=d_test.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        eat_model_path=str(args.eat_model_path),
    )
    np.savez(
        scores_dir / f"{args.machine}_clean_eat_knn_scores.npz",
        test_scores=scores,
        y_test=y_test,
        d_test=d_test.astype("U16"),
        test_paths=np.asarray(test_paths, dtype=object),
        score_mode="clean_eat_knn",
    )

    # optional baseline comparison if available
    baseline_rows = []
    v63_npz = Path(args.v63_root) / f"seed_{args.seed}" / "scores" / f"{args.machine}_full_5view_rankfusion_scores.npz"
    a0_npz = Path(args.stage14a_root) / "a0_frozen_beats" / "scores" / f"{args.machine}_frozen_beats_knn_scores.npz"
    a1_npz = Path(args.stage14a_root) / "a1_v63_beats_fusion" / f"seed_{args.seed}" / "scores" / f"{args.machine}_v63_beats_rankfusion_scores.npz"
    for name, npz_path in [("V63", v63_npz), ("A0_raw_BEATs", a0_npz), ("A1", a1_npz)]:
        if npz_path.exists():
            arr = np.load(npz_path, allow_pickle=True)
            dom_key = "d_test" if "d_test" in arr.files else "domains"
            base_scores = arr["test_scores"].astype(np.float64)
            base_y = arr["y_test"].astype(np.int64)
            base_d = arr[dom_key].astype(str)
            base_m = compute_metrics(base_y, base_scores, base_d)
            base_g = compute_score_geometry(base_scores, base_y, base_d)
            base_t = tail_metrics(base_scores, base_y)
            baseline_rows.append({
                "baseline": name,
                "hmean": base_m["hmean"],
                "pAUC": base_m["pauc"],
                "KS": base_g["source_target_ks"],
                "TPR@95n": base_t["TPR@95n"],
                "TPR@99n": base_t["TPR@99n"],
                "q95": base_t["q95"],
                "q99": base_t["q99"],
            })

    write_csv(out_root / "embedding_summary.csv", [{
        "machine": args.machine,
        "train_count": len(train_items),
        "test_count": len(test_items),
        "frame_output_shape": sample_feat_shape,
        "utterance_embedding_dim": int(train_emb.shape[1]),
        "sample_rate": sample_final_sr,
        "device": str(device),
    }])
    write_csv(out_root / "summary.csv", [{
        "machine": args.machine,
        "candidate": "clean_EAT_knn",
        "hmean": metrics["hmean"],
        "pAUC": metrics["pauc"],
        "KS": geometry["source_target_ks"],
        "TPR@95n": tail["TPR@95n"],
        "TPR@99n": tail["TPR@99n"],
    }])
    write_csv(out_root / "tail_table.csv", [{
        "machine": args.machine,
        **tail,
    }])
    write_csv(out_root / "score_geometry.csv", [{
        "machine": args.machine,
        **geometry,
    }])
    if baseline_rows:
        write_csv(out_root / "baseline_comparison.csv", baseline_rows)

    lines = [
        "# CTAF Stage16B Clean EAT One-Machine KNN Audit",
        "",
        f"- machine: `{args.machine}`",
        f"- seed reference: `{args.seed}`",
        f"- train count: `{len(train_items)}`",
        f"- test count: `{len(test_items)}`",
        f"- frame output shape: `{sample_feat_shape}`",
        f"- utterance embedding dim: `{train_emb.shape[1]}`",
        f"- hmean: `{metrics['hmean']:.6f}`",
        f"- pAUC: `{metrics['pauc']:.6f}`",
        f"- KS: `{geometry['source_target_ks']:.6f}`",
        f"- TPR@95n: `{tail['TPR@95n']:.6f}`",
        f"- TPR@99n: `{tail['TPR@99n']:.6f}`",
        f"- q95: `{tail['q95']:.6f}`",
        f"- q99: `{tail['q99']:.6f}`",
        "",
        "No training occurred.",
    ]
    Path(args.doc_path).write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "one_machine_knn":
        run_one_machine_knn(args)
        return
    raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
