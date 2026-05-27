#!/usr/bin/env python
"""Stage 18D-B: PaSST full-7 frozen kNN audit.

Protocol:
- frozen PaSST checkpoint only
- 32 kHz waveform handling
- 128-bin Kaldi fbank
- 16x16 patch projection, stride 10x10
- remove cls/dist tokens
- mean-pool patch tokens
- source train-normal memory only
- cosine kNN, k=5
- no training / no fine-tuning / no fusion / no selector / no oracle
"""

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


CKPT_DEFAULT = "/data1/test/pretrained_models/passt_clean/passt-s-f128-p16-s10-ap.476-swa.pt"
DATA_ROOT_DEFAULT = "/data1/test/LAES_2025/data/dcase"
OUTPUT_ROOT_DEFAULT = "results/stage18d_passt_full7_knn"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE18D_PASST_FULL7_KNN_AUDIT.md"
MACHINES_DEFAULT = "bearing,fan,gearbox,slider,toycar,toytrain,valve"

MINCLAMP_PER_MACHINE_DEFAULT = "results/stage16d_minclamp_mean_full7/per_machine.csv"
STAGE18D_ONE_MACHINE_DEFAULT = "results/stage18d_passt_one_machine_knn/summary.csv"

CANDIDATE_NAME = "PaSST_knn"
CURRENT_BEST_ROUTE = "F_P1_mean_std_A1_065"
CURRENT_BEST_MEAN_HMEAN = 0.607175
EAT_MINCLAMP_MEAN_HMEAN = 0.604713


def parse_args():
    p = argparse.ArgumentParser(description="Stage18D-B PaSST full7 frozen kNN audit")
    p.add_argument("--mode", required=True, choices=["check", "extract_passt_embeddings", "score_passt"])
    p.add_argument("--passt_ckpt_path", default=CKPT_DEFAULT)
    p.add_argument("--data_root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--machines", default=MACHINES_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--latent_knn_k", type=int, default=5)
    p.add_argument("--minclamp_per_machine_path", default=MINCLAMP_PER_MACHINE_DEFAULT)
    p.add_argument("--stage18d_one_machine_path", default=STAGE18D_ONE_MACHINE_DEFAULT)
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
    for name in ["torch", "torchaudio", "soundfile", "librosa", "timm", "scipy", "sklearn"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def build_passt_model(device: str):
    import torch.nn as nn
    from timm.models.vision_transformer import Block

    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(10, 10))

        def forward(self, x):
            return self.proj(x)

    class PaSSTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.dist_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.new_pos_embed = nn.Parameter(torch.zeros(1, 2, 768))
            self.freq_new_pos_embed = nn.Parameter(torch.zeros(1, 768, 12, 1))
            self.time_new_pos_embed = nn.Parameter(torch.zeros(1, 768, 1, 99))
            self.patch_embed = PatchEmbed()
            self.blocks = nn.ModuleList(
                [
                    Block(
                        dim=768,
                        num_heads=12,
                        mlp_ratio=4.0,
                        qkv_bias=True,
                        norm_layer=nn.LayerNorm,
                    )
                    for _ in range(12)
                ]
            )
            self.norm = nn.LayerNorm(768)

        def forward_tokens(self, x):
            # x: [B, 1, 128, 998]
            x = self.patch_embed(x)  # [B, 768, 12, 99]
            x = x + self.freq_new_pos_embed[:, :, : x.shape[2], :]
            x = x + self.time_new_pos_embed[:, :, :, : x.shape[3]]
            x = x.flatten(2).transpose(1, 2)
            cls = self.cls_token + self.new_pos_embed[:, 0:1, :]
            dist = self.dist_token + self.new_pos_embed[:, 1:2, :]
            cls = cls.expand(x.shape[0], -1, -1)
            dist = dist.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, dist, x], dim=1)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")

    model = PaSSTBackbone().to(requested)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, requested


def load_passt_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt if isinstance(ckpt, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported PaSST checkpoint format: {type(ckpt)}")

    model, requested = build_passt_model(device)

    # PaSST checkpoints may contain classification / distillation heads.
    # Stage18D-B only uses frozen backbone patch-token embeddings, so these heads
    # are ignored while backbone weight checks remain strict.
    ignored_prefixes = ("head.", "head_dist.")
    ignored_keys: list[str] = []
    filtered = {}
    for k, v in state_dict.items():
        if k.startswith(ignored_prefixes):
            ignored_keys.append(k)
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    unexpected_filtered = [k for k in unexpected if not k.startswith(ignored_prefixes)]
    if unexpected_filtered:
        raise RuntimeError(
            "Unexpected PaSST checkpoint keys outside ignored classifier heads: "
            f"{unexpected_filtered[:20]}"
        )

    missing_filtered = [k for k in missing if not k.startswith(ignored_prefixes)]
    if missing_filtered:
        raise RuntimeError(f"Missing PaSST backbone keys: {missing_filtered[:20]}")

    model.eval()
    model.to(requested)
    return model, requested, ckpt, ignored_keys


def load_waveform_32k(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != 32000:
        tensor = torchaudio.functional.resample(tensor, int(sr), 32000)
    return tensor.detach().cpu().numpy().astype(np.float32), int(sr)


def waveform_to_passt_input(wav_32k: np.ndarray):
    import torchaudio

    wav = torch.tensor(wav_32k, dtype=torch.float32)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=32000,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )  # [T, 128]
    target_length = 998
    if mel.shape[0] < target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, target_length - mel.shape[0]))
    else:
        mel = mel[:target_length, :]
    mel = mel.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 128, 998]
    return mel.to(dtype=torch.float32)


def passt_embed(model, waveform_32k: np.ndarray, device):
    x = waveform_to_passt_input(waveform_32k).unsqueeze(0).to(device)  # [1, 1, 128, 998]
    with torch.no_grad():
        tokens = model.forward_tokens(x)
        patch_tokens = tokens[:, 2:, :]
        emb = patch_tokens.mean(dim=1)
    return tokens, emb


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
    q95 = float(np.quantile(normals, 0.95)) if len(normals) else 0.0
    q99 = float(np.quantile(normals, 0.99)) if len(normals) else 0.0
    return {
        "q95": q95,
        "q99": q99,
        "anomaly_median": float(np.quantile(anomalies, 0.50)) if len(anomalies) else 0.0,
        "anomaly_q75": float(np.quantile(anomalies, 0.75)) if len(anomalies) else 0.0,
        "TPR@95n": float(np.mean(anomalies > q95)) if len(anomalies) else 0.0,
        "TPR@99n": float(np.mean(anomalies > q99)) if len(anomalies) else 0.0,
    }


def evaluate(scores: np.ndarray, y_test: np.ndarray, domains: np.ndarray):
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


def load_per_machine_csv(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    rows = load_csv_rows(path)
    out = {}
    for row in rows:
        machine = row.get("machine", "")
        if machine:
            out[machine] = row
    return out


def load_stage18d_one_machine(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = load_csv_rows(path)
    return next((r for r in rows if r.get("candidate") == CANDIDATE_NAME), None)


def build_check_markdown(status: dict) -> list[str]:
    failed = [k for k, ok in status["imports"].items() if not ok]
    return [
        "# CTAF Stage18D-B PaSST Full7 KNN Audit",
        "",
        "## Mode",
        "",
        "- `check`",
        "- no checkpoint load",
        "- no forward",
        "- no embedding extraction",
        "- no scoring",
        "",
        "## Status",
        "",
        f"- candidate: `{CANDIDATE_NAME}`",
        f"- checkpoint path exists: `{status['ckpt_exists']}`",
        f"- data root exists: `{status['data_root_exists']}`",
        f"- failed imports: `{', '.join(failed) if failed else 'none'}`",
    ]


def build_extract_markdown(rows: list[dict], ignored_keys: list[str]) -> list[str]:
    lines = [
        "# CTAF Stage18D-B PaSST Full7 KNN Audit",
        "",
        "## Mode",
        "",
        "- `extract_passt_embeddings`",
        "- frozen PaSST only",
        "- no scoring",
        "",
        "## Ignored Checkpoint Heads",
        "",
        "- ignored prefixes: `head.`, `head_dist.`",
        f"- ignored key count: `{len(ignored_keys)}`",
        "",
        "## Embedding Summary",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['machine']}` dim=`{row['embedding_dim']}` train=`{row['train_count']}` test=`{row['test_count']}` tokens=`{row['token_shape']}`"
        )
    return lines


def build_score_markdown(verdict: dict) -> list[str]:
    return [
        "# CTAF Stage18D-B PaSST Full7 KNN Audit",
        "",
        "## Candidate",
        "",
        f"- candidate: `{CANDIDATE_NAME}`",
        f"- mean hmean: `{verdict['mean_hmean']:.6f}`",
        f"- mean pAUC: `{verdict['mean_pAUC']:.6f}`",
        f"- mean TPR@99n: `{verdict['mean_TPR@99n']:.6f}`",
        f"- minclamp comparison available: `{verdict['minclamp_comparison_available']}`",
        f"- promising: `{verdict['promising']}`",
        f"- strong_positive: `{verdict['strong_positive']}`",
        f"- very_strong: `{verdict['very_strong']}`",
        f"- negative: `{verdict['negative']}`",
        "",
        "## Registered References",
        "",
        f"- EAT min_clamp__mean full7 mean_hmean: `{EAT_MINCLAMP_MEAN_HMEAN:.6f}`",
        f"- current best `{CURRENT_BEST_ROUTE}` mean_hmean: `{CURRENT_BEST_MEAN_HMEAN:.6f}`",
        "",
        "## Guardrails",
        "",
        "- no training",
        "- no fine-tuning",
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

    ckpt_path = Path(args.passt_ckpt_path)
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
        "passt_ckpt_path": str(ckpt_path),
        "ckpt_exists": ckpt_path.exists(),
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "machines": machines,
        "imports": import_status(),
        "notes": "No checkpoint load, no forward, no extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", status)
    write_markdown(Path(args.doc_path), build_check_markdown(status))


def run_extract_passt_embeddings(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    emb_root.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.passt_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PaSST checkpoint path does not exist: {ckpt_path}")

    model, device, ckpt, ignored_keys = load_passt_checkpoint(ckpt_path, args.device)
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
        sample_token_shape = None
        sample_orig_sr = None

        for item in train_items:
            wav32k, orig_sr = load_waveform_32k(Path(item.path))
            tokens, emb = passt_embed(model, wav32k, device)
            train_embs.append(emb.squeeze(0).detach().cpu().numpy())
            sample_token_shape = list(tokens.shape)
            sample_orig_sr = orig_sr

        for item in test_items:
            wav32k, orig_sr = load_waveform_32k(Path(item.path))
            tokens, emb = passt_embed(model, wav32k, device)
            test_embs.append(emb.squeeze(0).detach().cpu().numpy())
            y_test.append(int(item.label))
            d_test.append(item.domain)
            test_paths.append(str(item.path))
            sample_token_shape = list(tokens.shape)
            sample_orig_sr = orig_sr

        train_emb = np.stack(train_embs, axis=0).astype(np.float64)
        test_emb = np.stack(test_embs, axis=0).astype(np.float64)
        y_test_arr = np.asarray(y_test, dtype=np.int64)
        d_test_arr = np.asarray(d_test, dtype="U16")
        test_paths_arr = np.asarray(test_paths, dtype=object)

        emb_path = emb_root / f"{machine}_passt_embeddings.npz"
        np.savez_compressed(
            emb_path,
            train_embeddings=train_emb,
            test_embeddings=test_emb,
            y_test=y_test_arr,
            d_test=d_test_arr,
            test_paths=test_paths_arr,
            passt_ckpt_path=str(ckpt_path),
            candidate=CANDIDATE_NAME,
            original_sample_rate=sample_orig_sr,
            resampled_sample_rate=32000,
            token_shape=sample_token_shape,
            ignored_checkpoint_keys=np.asarray(ignored_keys, dtype=object),
        )

        emb_rows.append({
            "machine": machine,
            "candidate": CANDIDATE_NAME,
            "train_count": len(train_items),
            "test_count": len(test_items),
            "embedding_dim": int(train_emb.shape[1]),
            "original_sample_rate_example": sample_orig_sr,
            "resampled_sample_rate": 32000,
            "token_shape": sample_token_shape,
            "embedding_path": str(emb_path),
            "ignored_checkpoint_key_count": len(ignored_keys),
        })

    write_csv(out_root / "embedding_summary.csv", emb_rows)
    write_markdown(Path(args.doc_path), build_extract_markdown(emb_rows, ignored_keys))


def run_score_passt(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    emb_root = out_root / "embeddings"
    scores_root = out_root / "scores"
    scores_root.mkdir(parents=True, exist_ok=True)

    machines = parse_machines(args.machines)
    minclamp_path = Path(args.minclamp_per_machine_path)
    minclamp_per_machine = load_per_machine_csv(minclamp_path)
    one_machine_path = Path(args.stage18d_one_machine_path)
    one_machine_row = load_stage18d_one_machine(one_machine_path)

    missing_rows = []
    missing_rows.append({
        "kind": "stage16d_minclamp_per_machine",
        "path": str(minclamp_path),
        "exists": minclamp_path.exists(),
        "used": minclamp_per_machine is not None,
        "notes": "Optional EAT min_clamp__mean full7 comparison.",
    })
    missing_rows.append({
        "kind": "stage18d_passt_one_machine",
        "path": str(one_machine_path),
        "exists": one_machine_path.exists(),
        "used": one_machine_row is not None,
        "notes": "Optional PaSST gearbox one-machine comparison.",
    })

    per_machine_rows = []
    tail_rows = []
    geom_rows = []
    comparison_rows = []
    collapse_rows = []

    for machine in machines:
        emb_path = emb_root / f"{machine}_passt_embeddings.npz"
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

        score_path = scores_root / f"{machine}_passt_knn_scores.npz"
        np.savez(
            score_path,
            test_scores=scores.astype(np.float64),
            y_test=y_test.astype(np.int64),
            d_test=d_test.astype("U16"),
            test_paths=np.asarray(test_paths, dtype=object),
            score_mode=CANDIDATE_NAME,
            machine=machine,
            latent_knn_k=int(args.latent_knn_k),
        )

        min_row = minclamp_per_machine.get(machine) if minclamp_per_machine is not None else None
        min_h = float(min_row["hmean"]) if min_row is not None else None
        min_p = float(min_row["pAUC"]) if min_row is not None else None
        min_t99 = float(min_row["TPR@99n"]) if min_row is not None else None

        one_machine_h = ""
        one_machine_p = ""
        one_machine_t99 = ""
        if machine == "gearbox" and one_machine_row is not None:
            one_machine_h = float(one_machine_row["hmean"])
            one_machine_p = float(one_machine_row["pAUC"])
            one_machine_t99 = float(one_machine_row["TPR@99n"])

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
            "passt_hmean": summary["hmean"],
            "passt_pAUC": summary["pAUC"],
            "passt_TPR@99n": summary["TPR@99n"],
            "minclamp_hmean": min_h if min_h is not None else "",
            "minclamp_pAUC": min_p if min_p is not None else "",
            "minclamp_TPR@99n": min_t99 if min_t99 is not None else "",
            "delta_hmean_vs_minclamp": (summary["hmean"] - min_h) if min_h is not None else "",
            "delta_pAUC_vs_minclamp": (summary["pAUC"] - min_p) if min_p is not None else "",
            "delta_TPR99_vs_minclamp": (summary["TPR@99n"] - min_t99) if min_t99 is not None else "",
            "stage18d_one_machine_hmean": one_machine_h,
            "stage18d_one_machine_pAUC": one_machine_p,
            "stage18d_one_machine_TPR@99n": one_machine_t99,
            "delta_hmean_vs_stage18d_one_machine": (summary["hmean"] - one_machine_h) if isinstance(one_machine_h, float) else "",
        })
        collapse_rows.append({
            "machine": machine,
            "is_strong_machine_for_check": machine in {"bearing", "valve", "toytrain"},
            "is_expected_positive_machine": machine == "gearbox",
            "passt_hmean": summary["hmean"],
            "minclamp_hmean": min_h if min_h is not None else "",
            "delta_hmean_vs_minclamp": (summary["hmean"] - min_h) if min_h is not None else "",
            "improved_vs_minclamp": (summary["hmean"] > min_h) if min_h is not None else "",
            "severe_collapse_gt_0p02_vs_minclamp": ((min_h - summary["hmean"]) > 0.02) if min_h is not None else "",
        })

    mean_hmean = float(np.mean([r["hmean"] for r in per_machine_rows]))
    mean_pAUC = float(np.mean([r["pAUC"] for r in per_machine_rows]))
    mean_KS = float(np.mean([r["KS"] for r in per_machine_rows]))
    mean_TPR95 = float(np.mean([r["TPR@95n"] for r in per_machine_rows]))
    mean_TPR99 = float(np.mean([r["TPR@99n"] for r in per_machine_rows]))

    minclamp_available = minclamp_per_machine is not None and all(m in minclamp_per_machine for m in machines)
    minclamp_mean_hmean = (
        float(np.mean([float(minclamp_per_machine[m]["hmean"]) for m in machines]))
        if minclamp_available
        else None
    )
    improved_count = (
        int(np.sum([r["hmean"] > float(minclamp_per_machine[r["machine"]]["hmean"]) for r in per_machine_rows]))
        if minclamp_available
        else 0
    )
    gearbox_positive = (
        bool(next(r for r in comparison_rows if r["machine"] == "gearbox")["delta_hmean_vs_minclamp"] > 0)
        if minclamp_available and "gearbox" in machines
        else False
    )
    strong_machine_collapse = False
    if minclamp_available:
        for row in collapse_rows:
            if row["machine"] in {"bearing", "valve", "toytrain"} and bool(row["severe_collapse_gt_0p02_vs_minclamp"]):
                strong_machine_collapse = True
                break

    gains_only_one_machine = bool(improved_count <= 1) if minclamp_available else False

    promising = bool(mean_hmean >= EAT_MINCLAMP_MEAN_HMEAN and not strong_machine_collapse)
    strong_positive = bool(mean_hmean > CURRENT_BEST_MEAN_HMEAN and not strong_machine_collapse)
    very_strong = bool(mean_hmean >= 0.610000 and improved_count >= 4 and not strong_machine_collapse) if minclamp_available else bool(mean_hmean >= 0.610000)
    negative = bool(mean_hmean < 0.590000 or strong_machine_collapse or (minclamp_available and gains_only_one_machine))

    write_csv(out_root / "per_machine.csv", per_machine_rows)
    write_csv(out_root / "tail_table.csv", tail_rows)
    write_csv(out_root / "score_geometry.csv", geom_rows)
    write_csv(out_root / "comparison_vs_minclamp.csv", comparison_rows)
    write_csv(out_root / "collapse_check.csv", collapse_rows)
    write_csv(out_root / "missing_comparison_files.csv", missing_rows)
    write_csv(out_root / "summary.csv", [{
        "candidate": CANDIDATE_NAME,
        "mean_hmean": mean_hmean,
        "mean_pAUC": mean_pAUC,
        "mean_KS": mean_KS,
        "mean_TPR@95n": mean_TPR95,
        "mean_TPR@99n": mean_TPR99,
        "minclamp_comparison_available": minclamp_available,
        "minclamp_mean_hmean": minclamp_mean_hmean if minclamp_mean_hmean is not None else "",
        "registered_eat_minclamp_mean_hmean": EAT_MINCLAMP_MEAN_HMEAN,
        "registered_current_best_route": CURRENT_BEST_ROUTE,
        "registered_current_best_mean_hmean": CURRENT_BEST_MEAN_HMEAN,
        "improved_machine_count_vs_minclamp": improved_count if minclamp_available else "",
        "promising": promising,
        "strong_positive": strong_positive,
        "very_strong": very_strong,
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
        "minclamp_comparison_available": minclamp_available,
        "minclamp_mean_hmean": minclamp_mean_hmean,
        "registered_eat_minclamp_mean_hmean": EAT_MINCLAMP_MEAN_HMEAN,
        "registered_current_best_route": CURRENT_BEST_ROUTE,
        "registered_current_best_mean_hmean": CURRENT_BEST_MEAN_HMEAN,
        "gearbox_hmean_positive_vs_minclamp": gearbox_positive if minclamp_available else None,
        "improved_machine_count_vs_minclamp": improved_count if minclamp_available else None,
        "strong_machine_severe_collapse_vs_minclamp": strong_machine_collapse if minclamp_available else None,
        "gains_only_one_machine_vs_minclamp": gains_only_one_machine if minclamp_available else None,
        "promising": promising,
        "strong_positive": strong_positive,
        "very_strong": very_strong,
        "negative": negative,
        "notes": [
            "Frozen PaSST only.",
            "Ignored checkpoint classifier/distillation heads: head.*, head_dist.*.",
            "Patch tokens only: cls/dist tokens removed before mean pooling.",
            "No training, fine-tuning, LoRA, adapter, Conformer, fusion, selector, or oracle.",
            "Human review required before any promotion.",
        ],
    }
    write_json(out_root / "stage18d_passt_full7_verdict.json", verdict)
    write_markdown(Path(args.doc_path), build_score_markdown(verdict))


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "extract_passt_embeddings":
        run_extract_passt_embeddings(args)
        return
    if args.mode == "score_passt":
        run_score_passt(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
