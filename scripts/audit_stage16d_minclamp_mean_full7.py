#!/usr/bin/env python
"""Stage 16D full-7 audit for fixed clean EAT min_clamp__mean."""

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


MODEL_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage16d_minclamp_mean_full7"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE16D_MINCLAMP_MEAN_FULL7_AUDIT.md"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"

P1_PER_MACHINE_DEFAULT = "results/stage16b_clean_eat_full_frozen_knn/per_machine.csv"
STAGE16D_ONE_MACHINE_TEMPLATE = "results/stage16d_clean_eat_temporal_pooling_revisit_one_machine_{machine}/summary.csv"

CANDIDATE_NAME = "min_clamp__mean"
PREPROCESS_NAME = "min_clamp"
POOLING_NAME = "mean"


def parse_args():
    p = argparse.ArgumentParser(description="Stage16D min_clamp__mean full7 audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_minclamp_mean", "score_minclamp_mean"])
    p.add_argument("--eat_model_path", default=MODEL_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--stage16b_p1_per_machine_path", default=P1_PER_MACHINE_DEFAULT)
    return p.parse_args()


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
    for name in [
        "torch",
        "torchaudio",
        "soundfile",
        "librosa",
        "transformers",
        "timm",
        "safetensors",
        "scipy",
        "sklearn",
    ]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def parse_machines(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def expected_items(data_root: Path, machine: str):
    items = filter_items(scan_dataset(data_root), machine)
    train_items, test_items = split_train_test(items, use_target_train=False)
    return items, train_items, test_items


def load_clean_eat(model_path: Path, device: str):
    import torch

    package_parent = model_path.parent.resolve()
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))

    from checkpoints.configuration_eat import EATConfig
    from checkpoints.modeling_eat import EATModel

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

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
    for param in model.parameters():
        param.requires_grad_(False)
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
    return feat.squeeze(0).detach().cpu().numpy().astype(np.float32), final_sr


def min_clamp(frames: np.ndarray) -> np.ndarray:
    return np.where(np.abs(frames) < 0.1, 0.0, frames).astype(np.float32)


def mean_pool(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=0, dtype=np.float64).astype(np.float64)


def knn_cosine_scores(train_emb: np.ndarray, test_emb: np.ndarray, k: int = 5) -> np.ndarray:
    train_n = train_emb / np.maximum(np.linalg.norm(train_emb, axis=1, keepdims=True), 1e-12)
    test_n = test_emb / np.maximum(np.linalg.norm(test_emb, axis=1, keepdims=True), 1e-12)
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


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_p1_per_machine(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        machine = row.get("machine", "")
        out[machine] = row
    return out


def load_stage16d_one_machine_row(machine: str) -> dict | None:
    path = ROOT / STAGE16D_ONE_MACHINE_TEMPLATE.format(machine=machine)
    if not path.exists():
        return None
    rows = load_csv_rows(path)
    match = next((r for r in rows if r.get("candidate") == CANDIDATE_NAME), None)
    return match


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage16D MinClamp Mean Full7 Audit",
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
        f"- candidate: `{CANDIDATE_NAME}`",
        f"- checkpoint path exists: `{status['eat_model_path_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_extract_markdown(rows: list[dict]) -> list[str]:
    lines = [
        "# CTAF Stage16D MinClamp Mean Full7 Audit",
        "",
        "## Mode",
        "",
        "- `extract_minclamp_mean`",
        "- frozen clean EAT only",
        "- no scoring",
        "",
        "## Embedding Summary",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['machine']}` dim=`{row['embedding_dim']}` train=`{row['train_count']}` test=`{row['test_count']}`"
        )
    return lines


def build_score_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage16D MinClamp Mean Full7 Audit",
        "",
        "## Candidate",
        "",
        f"- candidate: `{CANDIDATE_NAME}`",
        f"- mean hmean: `{verdict['mean_hmean']:.6f}`",
        f"- mean pAUC: `{verdict['mean_pAUC']:.6f}`",
        f"- mean TPR@99n: `{verdict['mean_TPR@99n']:.6f}`",
        f"- p1 comparison available: `{verdict['p1_comparison_available']}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Guardrails",
        "",
        "- no LoRA",
        "- no adapter",
        "- no Conformer",
        "- no fusion",
        "- no selector",
        "- no oracle",
        "- human review required before any promotion",
    ]


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.eat_model_path)
    data_root = Path(args.data_root)
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
        })
    write_csv(out_root / "machine_check.csv", machine_rows)
    status = {
        "mode": "check",
        "candidate": CANDIDATE_NAME,
        "eat_model_path": str(model_path),
        "eat_model_path_exists": model_path.exists(),
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "machines": machines,
        "imports": import_status(),
        "notes": "No model load, no forward, no extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", status)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_extract_minclamp_mean(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    model, device = load_clean_eat(Path(args.eat_model_path), args.device)
    machines = parse_machines(args.machines)
    all_items = scan_dataset(Path(args.data_root))
    emb_rows = []

    for machine in machines:
        items = filter_items(all_items, machine)
        train_items, test_items = split_train_test(items, use_target_train=False)
        if not train_items or not test_items:
            raise RuntimeError(f"Machine {machine}: train={len(train_items)} test={len(test_items)}")

        train_embs = []
        test_embs = []
        y_test = []
        d_test = []
        test_paths = []
        sample_final_sr = None
        sample_feat_shape = None

        for item in train_items:
            wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
            wav = wav.mean(axis=1)
            feat, final_sr = eat_frames(model, wav, int(sr), device)
            feat = min_clamp(feat)
            emb = mean_pool(feat)
            train_embs.append(emb)
            sample_final_sr = final_sr
            sample_feat_shape = list(feat.shape)

        for item in test_items:
            wav, sr = sf.read(str(item.path), always_2d=True, dtype="float32")
            wav = wav.mean(axis=1)
            feat, final_sr = eat_frames(model, wav, int(sr), device)
            feat = min_clamp(feat)
            emb = mean_pool(feat)
            test_embs.append(emb)
            y_test.append(int(item.label))
            d_test.append(item.domain)
            test_paths.append(str(item.path))
            sample_final_sr = final_sr
            sample_feat_shape = list(feat.shape)

        train_emb = np.stack(train_embs, axis=0).astype(np.float64)
        test_emb = np.stack(test_embs, axis=0).astype(np.float64)
        y_test_arr = np.asarray(y_test, dtype=np.int64)
        d_test_arr = np.asarray(d_test, dtype="U16")
        test_paths_arr = np.asarray(test_paths, dtype=object)

        np.savez_compressed(
            emb_root / f"{machine}_minclamp_mean_embeddings.npz",
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            y_test=y_test_arr,
            d_test=d_test_arr,
            test_paths=test_paths_arr,
            eat_model_path=str(args.eat_model_path),
            preprocess=PREPROCESS_NAME,
            pooling=POOLING_NAME,
            candidate=CANDIDATE_NAME,
            sample_rate=sample_final_sr,
            frame_output_shape=sample_feat_shape,
        )
        emb_rows.append({
            "machine": machine,
            "candidate": CANDIDATE_NAME,
            "train_count": len(train_items),
            "test_count": len(test_items),
            "embedding_dim": int(train_emb.shape[1]),
            "sample_rate": sample_final_sr,
            "frame_output_shape": sample_feat_shape,
            "embedding_path": str(emb_root / f"{machine}_minclamp_mean_embeddings.npz"),
        })

    write_csv(out_root / "embedding_summary.csv", emb_rows)
    write_markdown(Path(args.doc_path), build_extract_markdown(emb_rows))


def run_score_minclamp_mean(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)
    machines = parse_machines(args.machines)

    p1_path = Path(args.stage16b_p1_per_machine_path)
    p1_per_machine = load_p1_per_machine(p1_path)
    missing_rows = []
    if p1_per_machine is None:
        missing_rows.append({
            "kind": "stage16b_p1_per_machine",
            "path": str(p1_path),
            "exists": False,
            "used": False,
            "notes": "Missing Stage16B P1 full7 comparison file.",
        })
    else:
        missing_rows.append({
            "kind": "stage16b_p1_per_machine",
            "path": str(p1_path),
            "exists": True,
            "used": True,
            "notes": "Loaded Stage16B P1 full7 comparison file.",
        })

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    comparison_rows = []
    collapse_rows = []

    for machine in machines:
        one_machine_path = ROOT / STAGE16D_ONE_MACHINE_TEMPLATE.format(machine=machine)
        one_machine_row = load_stage16d_one_machine_row(machine)
        missing_rows.append({
            "kind": f"stage16d_one_machine_{machine}",
            "path": str(one_machine_path),
            "exists": one_machine_path.exists(),
            "used": one_machine_row is not None,
            "notes": "Optional Stage16D one-machine comparison.",
        })

        emb_path = emb_root / f"{machine}_minclamp_mean_embeddings.npz"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embedding file: {emb_path}")
        arr = np.load(emb_path, allow_pickle=True)
        train_emb = arr["train_embeddings"].astype(np.float64)
        test_emb = arr["test_embeddings"].astype(np.float64)
        y_test = arr["y_test"].astype(np.int64)
        d_test = arr["d_test"].astype(str)
        test_paths = arr["test_paths"].astype(str)

        scores = knn_cosine_scores(train_emb, test_emb, k=args.latent_knn_k)
        summary, tail, geom = evaluate(scores, y_test, d_test)

        np.savez(
            scores_root / f"{machine}_minclamp_mean_scores.npz",
            test_scores=scores.astype(np.float64),
            y_test=y_test.astype(np.int64),
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            score_mode=CANDIDATE_NAME,
            machine=machine,
            latent_knn_k=int(args.latent_knn_k),
        )

        p1_row = p1_per_machine.get(machine) if p1_per_machine is not None else None
        p1_h = float(p1_row["hmean"]) if p1_row is not None else None
        p1_p = float(p1_row["pAUC"]) if p1_row is not None else None
        p1_t99 = float(p1_row["TPR@99n"]) if p1_row is not None else None
        stage16d_1m_h = float(one_machine_row["hmean"]) if one_machine_row is not None else None
        stage16d_1m_p = float(one_machine_row["pAUC"]) if one_machine_row is not None else None
        stage16d_1m_t99 = float(one_machine_row["TPR@99n"]) if one_machine_row is not None else None

        per_machine_rows.append({
            "machine": machine,
            "candidate": CANDIDATE_NAME,
            "hmean": summary["hmean"],
            "pAUC": summary["pAUC"],
            "KS": summary["KS"],
            "TPR@95n": summary["TPR@95n"],
            "TPR@99n": summary["TPR@99n"],
        })
        tail_rows.append({
            "machine": machine,
            "candidate": CANDIDATE_NAME,
            **tail,
        })
        geom_rows.append({
            "machine": machine,
            "candidate": CANDIDATE_NAME,
            **geom,
        })
        comparison_rows.append({
            "machine": machine,
            "candidate": CANDIDATE_NAME,
            "full7_hmean": summary["hmean"],
            "full7_pAUC": summary["pAUC"],
            "full7_TPR@99n": summary["TPR@99n"],
            "p1_hmean": p1_h if p1_h is not None else "",
            "p1_pAUC": p1_p if p1_p is not None else "",
            "p1_TPR@99n": p1_t99 if p1_t99 is not None else "",
            "delta_hmean_vs_p1": (summary["hmean"] - p1_h) if p1_h is not None else "",
            "delta_pAUC_vs_p1": (summary["pAUC"] - p1_p) if p1_p is not None else "",
            "delta_TPR99_vs_p1": (summary["TPR@99n"] - p1_t99) if p1_t99 is not None else "",
            "stage16d_one_machine_hmean": stage16d_1m_h if stage16d_1m_h is not None else "",
            "stage16d_one_machine_pAUC": stage16d_1m_p if stage16d_1m_p is not None else "",
            "stage16d_one_machine_TPR@99n": stage16d_1m_t99 if stage16d_1m_t99 is not None else "",
            "delta_hmean_vs_stage16d_one_machine": (summary["hmean"] - stage16d_1m_h) if stage16d_1m_h is not None else "",
        })
        collapse_rows.append({
            "machine": machine,
            "is_strong_machine_for_check": machine in {"bearing", "valve", "toytrain"},
            "is_expected_positive_machine": machine in {"gearbox", "slider"},
            "full7_hmean": summary["hmean"],
            "p1_hmean": p1_h if p1_h is not None else "",
            "delta_hmean_vs_p1": (summary["hmean"] - p1_h) if p1_h is not None else "",
            "improved_vs_p1": (summary["hmean"] > p1_h) if p1_h is not None else "",
            "severe_collapse_gt_0p02": ((p1_h - summary["hmean"]) > 0.02) if p1_h is not None else "",
        })

    mean_hmean = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_pAUC = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_KS = float(np.mean([r["KS"] for r in per_machine_rows]))
    mean_TPR95 = float(np.mean([r["TPR@95n"] for r in per_machine_rows]))
    mean_TPR99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))

    p1_available = p1_per_machine is not None and len(p1_per_machine) >= len(machines)
    p1_mean_hmean = float(np.mean([float(p1_per_machine[m]["hmean"]) for m in machines])) if p1_available else None
    improved_count = int(np.sum([r["hmean"] > float(p1_per_machine[r["machine"]]["hmean"]) for r in per_machine_rows])) if p1_available else 0
    gearbox_positive = bool(next(r for r in comparison_rows if r["machine"] == "gearbox")["delta_hmean_vs_p1"] > 0) if p1_available else False
    slider_positive = bool(next(r for r in comparison_rows if r["machine"] == "slider")["delta_hmean_vs_p1"] > 0) if p1_available else False
    strong_machine_collapse = False
    if p1_available:
        for row in collapse_rows:
            if row["machine"] in {"bearing", "valve", "toytrain"} and bool(row["severe_collapse_gt_0p02"]):
                strong_machine_collapse = True
                break

    gains_only_one_machine = bool(improved_count <= 1) if p1_available else False
    promising = bool(
        p1_available
        and mean_hmean >= p1_mean_hmean
        and gearbox_positive
        and slider_positive
        and not strong_machine_collapse
    )
    strong_positive = bool(
        p1_available
        and mean_hmean >= p1_mean_hmean + 0.005
        and improved_count >= 4
        and not strong_machine_collapse
    )
    negative = bool(
        p1_available
        and (
            mean_hmean < p1_mean_hmean
            or strong_machine_collapse
            or gains_only_one_machine
        )
    )

    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "comparison_vs_p1.csv", comparison_rows)
    write_csv(out_root / "collapse_check.csv", collapse_rows)
    write_csv(out_root / "missing_comparison_files.csv", missing_rows)
    write_csv(out_root / "summary.csv", [{
        "candidate": CANDIDATE_NAME,
        "mean_hmean": mean_hmean,
        "mean_pAUC": mean_pAUC,
        "mean_KS": mean_KS,
        "mean_TPR@95n": mean_TPR95,
        "mean_TPR@99n": mean_TPR99,
        "p1_comparison_available": p1_available,
        "p1_mean_hmean": p1_mean_hmean if p1_mean_hmean is not None else "",
        "improved_machine_count_vs_p1": improved_count if p1_available else "",
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
    }])

    verdict = {
        "candidate": CANDIDATE_NAME,
        "machines": machines,
        "mean_hmean": mean_hmean,
        "mean_pAUC": mean_pAUC,
        "mean_KS": mean_KS,
        "mean_TPR@95n": mean_TPR95,
        "mean_TPR@99n": mean_TPR99,
        "p1_comparison_available": p1_available,
        "p1_mean_hmean": p1_mean_hmean,
        "gearbox_hmean_positive_vs_p1": gearbox_positive if p1_available else None,
        "slider_hmean_positive_vs_p1": slider_positive if p1_available else None,
        "improved_machine_count_vs_p1": improved_count if p1_available else None,
        "strong_machine_severe_collapse": strong_machine_collapse if p1_available else None,
        "gains_only_one_machine": gains_only_one_machine if p1_available else None,
        "promising": promising,
        "strong_positive": strong_positive,
        "negative": negative,
        "notes": [
            "Fixed candidate only: min_clamp__mean.",
            "No LoRA, adapter, Conformer, fusion, selector, or oracle.",
            "Human review required before any promotion.",
        ],
    }
    write_json(out_root / "stage16d_full7_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_score_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_minclamp_mean":
        run_extract_minclamp_mean(args)
        return
    if args.mode == "score_minclamp_mean":
        run_score_minclamp_mean(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
