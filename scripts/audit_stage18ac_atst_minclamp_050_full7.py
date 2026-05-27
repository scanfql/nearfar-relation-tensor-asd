#!/usr/bin/env python
"""Stage 18A-C: ATST full7 + fixed F_ATST_minclamp_050 audit."""

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


CHECKPOINT_DEFAULT = "/data1/test/pretrained_models/atst_clean/atst_as2M.ckpt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
MINCLAMP_ROOT_DEFAULT = "results/stage16d_minclamp_mean_full7"
OUTPUT_ROOT_DEFAULT = "results/stage18ac_atst_minclamp_050_full7"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18AC_ATST_MINCLAMP_050_FULL7_AUDIT.md"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"

CURRENT_BEST_REGISTERED = {
    "candidate": "F_P1_mean_std_A1_065_REGISTERED",
    "mean_hmean": 0.607175,
    "std_hmean": 0.001630,
    "mean_pAUC": 0.558321,
    "mean_TPR@99n": 0.098095,
}

REGISTERED_ATST_GEARBOX = {
    "hmean": 0.565605,
    "pAUC": 0.533684,
    "TPR@99n": 0.030000,
}

REGISTERED_MINCLAMP_GEARBOX = {
    "hmean": 0.577034,
    "pAUC": 0.504211,
    "TPR@99n": 0.030000,
}

CANDIDATE_FUSION = "F_ATST_minclamp_050"
ATST_BASE = "ATST_knn"
MINCLAMP_BASE = "min_clamp__mean"


def parse_args():
    p = argparse.ArgumentParser(description="Stage18A-C ATST full7 + F_ATST_minclamp_050 audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_atst_embeddings", "score_atst", "fixed_fusion"])
    p.add_argument("--atst_ckpt_path", default=CHECKPOINT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--minclamp_root", default=MINCLAMP_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    return p.parse_args()


def parse_machines(text: str) -> list[str]:
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
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def import_status():
    status = {}
    for name in ["torch", "torchaudio", "soundfile", "librosa", "timm", "safetensors", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def load_checkpoint_header(ckpt_path: Path):
    blob = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else {}
    return blob, state_dict


def build_atst_model(device: str):
    import torch.nn as nn
    from timm.models.vision_transformer import Block

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Linear(256, 768)

        def forward(self, x):
            return self.patch_embed(x)

    class ATSTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.mask_embed = nn.Parameter(torch.zeros(1, 1, 768))
            self.pos_embed = nn.Parameter(torch.zeros(1, 251, 768))
            self.patch_embed = PatchEmbed()
            self.blocks = nn.ModuleList(
                [Block(dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=False, norm_layer=nn.LayerNorm) for _ in range(12)]
            )
            self.norm_frame = nn.LayerNorm(768)

        def forward_tokens(self, x):
            if x.ndim != 3 or x.shape[1] != 64 or x.shape[2] != 1004:
                raise RuntimeError(f"ATST expected input [B,64,1004], got {tuple(x.shape)}")
            x = x.unfold(2, 4, 4)
            x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], 251, 256)
            x = self.patch_embed(x)
            x = x + self.pos_embed[:, : x.shape[1], :]
            for blk in self.blocks:
                x = blk(x)
            x = self.norm_frame(x)
            return x

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    model = ATSTBackbone().to(requested)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def load_atst_checkpoint(ckpt_path: Path, device: str):
    blob = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = blob["state_dict"] if isinstance(blob, dict) else blob
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(state_dict)}")

    model, requested = build_atst_model(device)
    stripped = {}
    prefix = "encoder.encoder."
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = value
    if not stripped:
        raise RuntimeError("No encoder.encoder.* weights found in ATST checkpoint.")

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading ATST backbone: {unexpected}")
    required_missing = [k for k in missing if not k.startswith("mask_embed")]
    if required_missing:
        raise RuntimeError(f"Missing ATST backbone keys: {required_missing}")
    model.eval()
    model.to(requested)
    return model, requested, blob


def waveform_to_atst_input(wav_16k: np.ndarray):
    import torchaudio

    wav = torch.tensor(wav_16k, dtype=torch.float32)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=16000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=64,
        dither=0.0,
        frame_shift=10,
    )
    target_length = 1004
    if mel.shape[0] < target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
    else:
        mel = mel[:target_length, :]
    mel = mel.transpose(0, 1).contiguous()
    return mel.to(dtype=torch.float32)


def load_waveform_16k(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != 16000:
        tensor = torchaudio.functional.resample(tensor, int(sr), 16000)
    return tensor.detach().cpu().numpy().astype(np.float32), int(sr)


def atst_embed(model, waveform_16k: np.ndarray, device):
    x = waveform_to_atst_input(waveform_16k).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = model.forward_tokens(x)
        emb = tokens.mean(dim=1)
    return tokens, emb


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


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


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray) -> tuple[dict, dict, dict]:
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


def atst_score_path(output_root: Path, machine: str) -> Path:
    return output_root / "scores" / f"{machine}_atst_knn_scores.npz"


def atst_embedding_path(output_root: Path, machine: str) -> Path:
    return output_root / "embeddings" / f"{machine}_atst_embeddings.npz"


def minclamp_score_path(minclamp_root: Path, machine: str) -> Path:
    return minclamp_root / "scores" / f"{machine}_minclamp_mean_scores.npz"


def load_score_bundle(npz_path: Path) -> dict:
    arr = np.load(npz_path, allow_pickle=True)
    if "test_scores" in arr.files:
        scores = arr["test_scores"].astype(np.float64)
    elif "scores" in arr.files:
        scores = arr["scores"].astype(np.float64)
    else:
        raise KeyError(f"{npz_path} has no test_scores/scores key; keys={arr.files}")
    if "d_test" in arr.files:
        domains = arr["d_test"].astype(str)
    elif "domains" in arr.files:
        domains = arr["domains"].astype(str)
    else:
        raise KeyError(f"{npz_path} has no d_test/domains key; keys={arr.files}")
    return {
        "path": str(npz_path),
        "scores": scores,
        "y": arr["y_test"].astype(np.int64),
        "d": domains,
        "paths": arr["test_paths"].astype(str) if "test_paths" in arr.files else None,
    }


def rank_norm(scores: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    if scores.size <= 1:
        return np.zeros_like(scores, dtype=np.float64)
    ranks = rankdata(scores, method="average")
    return ((ranks - 1.0) / (scores.size - 1.0)).astype(np.float64)


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage18A-C ATST Full7 + F_ATST_minclamp_050 Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no model load",
        "- no forward",
        "- no extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- checkpoint path exists: `{status['ckpt_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- minclamp root exists: `{status['minclamp_root_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_extract_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "# CTAF Stage18A-C ATST Full7 + F_ATST_minclamp_050 Audit",
        "",
        "## Mode",
        "",
        "- `extract_atst_embeddings`",
        "- frozen ATST only",
        "- no scoring",
        "",
        "## Embedding Summary",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['machine']}` dim=`{row['embedding_dim']}` train=`{row['train_count']}` test=`{row['test_count']}`")
    return lines


def build_score_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage18A-C ATST Full7 + F_ATST_minclamp_050 Audit",
        "",
        "## Candidate",
        "",
        f"- candidate: `{CANDIDATE_FUSION}`",
        f"- atst mean hmean: `{verdict['atst_mean_hmean']:.6f}`",
        f"- fusion mean hmean: `{verdict['fusion_mean_hmean']:.6f}`",
        f"- fusion mean pAUC: `{verdict['fusion_mean_pAUC']:.6f}`",
        f"- fusion mean TPR@99n: `{verdict['fusion_mean_TPR@99n']:.6f}`",
        f"- improved_vs_atst: `{verdict['improved_vs_atst']}`",
        f"- improved_vs_minclamp: `{verdict['improved_vs_minclamp']}`",
        f"- improved_vs_current_best: `{verdict['improved_vs_current_best']}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)
    ckpt_path = Path(args.atst_ckpt_path)
    minclamp_root = Path(args.minclamp_root)
    machines = parse_machines(args.machines)
    items = scan_dataset(data_root) if data_root.exists() else []
    machine_rows = []
    for machine in machines:
        machine_items = filter_items(items, machine) if items else []
        train_items, test_items = split_train_test(machine_items, use_target_train=False) if machine_items else ([], [])
        machine_rows.append({
            "machine": machine,
            "exists_in_data_root": len(machine_items) > 0,
            "source_train_normal_count": len(train_items),
            "test_count": len(test_items),
            "minclamp_score_exists": minclamp_score_path(minclamp_root, machine).exists(),
        })
    write_csv(out_root / "machine_check.csv", machine_rows)
    status = {
        "mode": "check",
        "ckpt_exists": ckpt_path.exists(),
        "data_root_exists": data_root.exists(),
        "minclamp_root_exists": minclamp_root.exists(),
        "machines": machines,
        "imports": import_status(),
    }
    write_json(out_root / "check_status.json", status)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_extract_atst_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.atst_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ATST checkpoint path does not exist: {ckpt_path}")
    model, device, blob = load_atst_checkpoint(ckpt_path, args.device)
    machines = parse_machines(args.machines)
    all_items = scan_dataset(Path(args.data_root))
    rows = []

    for machine in machines:
        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)
        if not train_items or not test_items:
            raise RuntimeError(f"Machine {machine}: train={len(train_items)}, test={len(test_items)}")

        train_embs = []
        test_embs = []
        y_test = []
        d_test = []
        test_paths = []
        sample_token_shape = None
        for item in train_items:
            wav16k, _ = load_waveform_16k(Path(item.path))
            tokens, emb = atst_embed(model, wav16k, device)
            train_embs.append(emb.squeeze(0).detach().cpu().numpy())
            sample_token_shape = list(tokens.shape)
        for item in test_items:
            wav16k, _ = load_waveform_16k(Path(item.path))
            tokens, emb = atst_embed(model, wav16k, device)
            test_embs.append(emb.squeeze(0).detach().cpu().numpy())
            y_test.append(int(item.label))
            d_test.append(item.domain)
            test_paths.append(str(item.path))
            sample_token_shape = list(tokens.shape)

        train_emb = np.stack(train_embs, axis=0).astype(np.float64)
        test_emb = np.stack(test_embs, axis=0).astype(np.float64)
        np.savez_compressed(
            atst_embedding_path(out_root, machine),
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            y_test=np.asarray(y_test, dtype=np.int64),
            d_test=np.asarray(d_test, dtype="U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            atst_ckpt_path=str(ckpt_path),
        )
        rows.append({
            "machine": machine,
            "train_count": len(train_items),
            "test_count": len(test_items),
            "embedding_dim": int(train_emb.shape[1]),
            "token_shape": sample_token_shape,
            "embedding_path": str(atst_embedding_path(out_root, machine)),
        })

    write_csv(out_root / "embedding_summary.csv", rows)
    write_markdown(Path(args.doc_path), build_extract_markdown(rows))


def run_score_atst(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    machines = parse_machines(args.machines)

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    for machine in machines:
        emb_path = atst_embedding_path(out_root, machine)
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing ATST embedding file: {emb_path}")
        arr = np.load(emb_path, allow_pickle=True)
        train_emb = arr["train_embeddings"].astype(np.float64)
        test_emb = arr["test_embeddings"].astype(np.float64)
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype(str)
        test_paths = arr["test_paths"].astype(str)

        scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
        summary, tail, geom = evaluate(scores, y_test, d_test)

        np.savez(
            atst_score_path(out_root, machine),
            test_scores=scores.astype(np.float64),
            y_test=y_test.astype(np.int64),
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            score_mode=ATST_BASE,
            machine=machine,
            latent_knn_k=int(args.latent_knn_k),
        )
        per_machine_rows.append({"machine": machine, "candidate": ATST_BASE, **summary})
        tail_rows.append({"machine": machine, "candidate": ATST_BASE, **tail})
        geom_rows.append({"machine": machine, "candidate": ATST_BASE, **geom})

    mean_hmean = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_pAUC = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_KS = float(np.mean([r["KS"] for r in per_machine_rows]))
    mean_TPR95 = float(np.mean([r["TPR@95n"] for r in per_machine_rows]))
    mean_TPR99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))
    write_csv(out_root / "atst_summary.csv", [{
        "candidate": ATST_BASE,
        "mean_hmean": mean_hmean,
        "mean_pAUC": mean_pAUC,
        "mean_KS": mean_KS,
        "mean_TPR@95n": mean_TPR95,
        "mean_TPR@99n": mean_TPR99,
    }])
    write_csv(out_root / "atst_per_machine.csv", per_machine_rows)
    write_csv(out_root / "atst_tail_table.csv", tail_rows)
    write_csv(out_root / "atst_score_geometry.csv", geom_rows)


def run_fixed_fusion(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    scores_root = out_root / "fusion_scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    minclamp_root = Path(args.minclamp_root)
    machines = parse_machines(args.machines)

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    comparison_rows = []
    for machine in machines:
        atst_bundle = load_score_bundle(atst_score_path(out_root, machine))
        minc_bundle = load_score_bundle(minclamp_score_path(minclamp_root, machine))

        y_match = np.array_equal(atst_bundle["y"], minc_bundle["y"])
        d_match = np.array_equal(atst_bundle["d"], minc_bundle["d"])
        if atst_bundle["paths"] is not None and minc_bundle["paths"] is not None:
            p_match = np.array_equal(atst_bundle["paths"], minc_bundle["paths"])
        else:
            p_match = bool(y_match and d_match)
        if not (y_match and d_match and p_match):
            raise RuntimeError(f"ATST/min_clamp alignment failed on machine={machine}: y={y_match} d={d_match} paths={p_match}")

        atst_scores = atst_bundle["scores"]
        minc_scores = minc_bundle["scores"]
        fusion_scores = 0.5 * rank_norm(atst_scores) + 0.5 * rank_norm(minc_scores)
        y = atst_bundle["y"]
        d = atst_bundle["d"]
        paths = atst_bundle["paths"]
        summary, tail, geom = evaluate(fusion_scores, y, d)

        score_dir = scores_root / machine
        score_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            score_dir / f"{machine}_scores.npz",
            test_scores=fusion_scores.astype(np.float64),
            y_test=y.astype(np.int64),
            d_test=d.astype("U16"),
            test_paths=np.asarray(paths, dtype=object) if paths is not None else None,
            score_mode=CANDIDATE_FUSION,
            machine=machine,
        )

        per_machine_rows.append({"machine": machine, "candidate": CANDIDATE_FUSION, **summary})
        tail_rows.append({"machine": machine, "candidate": CANDIDATE_FUSION, **tail})
        geom_rows.append({"machine": machine, "candidate": CANDIDATE_FUSION, **geom})
        comparison_rows.append({
            "machine": machine,
            "atst_hmean": REGISTERED_ATST_GEARBOX["hmean"] if machine == "gearbox" else "",
            "atst_pAUC": REGISTERED_ATST_GEARBOX["pAUC"] if machine == "gearbox" else "",
            "atst_TPR@99n": REGISTERED_ATST_GEARBOX["TPR@99n"] if machine == "gearbox" else "",
            "minclamp_hmean": REGISTERED_MINCLAMP_GEARBOX["hmean"] if machine == "gearbox" else "",
            "minclamp_pAUC": REGISTERED_MINCLAMP_GEARBOX["pAUC"] if machine == "gearbox" else "",
            "minclamp_TPR@99n": REGISTERED_MINCLAMP_GEARBOX["TPR@99n"] if machine == "gearbox" else "",
            "fusion_hmean": summary["hmean"],
            "fusion_pAUC": summary["pAUC"],
            "fusion_TPR@99n": summary["TPR@99n"],
        })

    mean_hmean = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_pAUC = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_KS = float(np.mean([r["KS"] for r in per_machine_rows]))
    mean_TPR95 = float(np.mean([r["TPR@95n"] for r in per_machine_rows]))
    mean_TPR99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))

    atst_summary_path = out_root / "atst_summary.csv"
    atst_mean_hmean = None
    if atst_summary_path.exists():
        rows = list(csv.DictReader(atst_summary_path.open()))
        if rows:
            atst_mean_hmean = float(rows[0]["mean_hmean"])

    minclamp_summary_path = Path(args.minclamp_root) / "summary.csv"
    minclamp_mean_hmean = None
    if minclamp_summary_path.exists():
        rows = list(csv.DictReader(minclamp_summary_path.open()))
        if rows:
            minclamp_mean_hmean = float(rows[0]["mean_hmean"])

    improved_vs_atst = bool(atst_mean_hmean is not None and mean_hmean > atst_mean_hmean)
    improved_vs_minclamp = bool(minclamp_mean_hmean is not None and mean_hmean > minclamp_mean_hmean)
    improved_vs_current_best = bool(mean_hmean > CURRENT_BEST_REGISTERED["mean_hmean"])
    promising = bool(improved_vs_minclamp and mean_pAUC >= 0.54)
    strong_positive = bool(improved_vs_current_best and mean_pAUC >= CURRENT_BEST_REGISTERED["mean_pAUC"] - 0.005)
    negative = bool((not improved_vs_atst) and (not improved_vs_minclamp))

    write_csv(out_root / "fusion_summary.csv", [{
        "candidate": CANDIDATE_FUSION,
        "mean_hmean": mean_hmean,
        "mean_pAUC": mean_pAUC,
        "mean_KS": mean_KS,
        "mean_TPR@95n": mean_TPR95,
        "mean_TPR@99n": mean_TPR99,
        "improved_vs_atst": improved_vs_atst,
        "improved_vs_minclamp": improved_vs_minclamp,
        "improved_vs_current_best": improved_vs_current_best,
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
    }])
    write_csv(out_root / "fusion_per_machine.csv", per_machine_rows)
    write_csv(out_root / "fusion_tail_table.csv", tail_rows)
    write_csv(out_root / "fusion_score_geometry.csv", geom_rows)
    write_csv(out_root / "fusion_comparison.csv", comparison_rows)

    verdict = {
        "candidate": CANDIDATE_FUSION,
        "atst_mean_hmean": atst_mean_hmean,
        "minclamp_mean_hmean": minclamp_mean_hmean,
        "fusion_mean_hmean": mean_hmean,
        "fusion_mean_pAUC": mean_pAUC,
        "fusion_mean_TPR@99n": mean_TPR99,
        "improved_vs_atst": improved_vs_atst,
        "improved_vs_minclamp": improved_vs_minclamp,
        "improved_vs_current_best": improved_vs_current_best,
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "notes": [
            "Only fixed candidate F_ATST_minclamp_050.",
            "No alpha sweep, selector, or oracle.",
            "Human review required before any promotion.",
        ],
    }
    write_json(out_root / "stage18ac_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_score_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_atst_embeddings":
        run_extract_atst_embeddings(args)
        return
    if args.mode == "score_atst":
        run_score_atst(args)
        return
    if args.mode == "fixed_fusion":
        run_fixed_fusion(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
