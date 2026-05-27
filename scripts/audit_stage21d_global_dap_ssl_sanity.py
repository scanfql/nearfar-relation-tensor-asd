#!/usr/bin/env python
"""Stage21D: global normal-only DAP SSL sanity on Stage21C allowlist."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CHECKPOINT_DEFAULT = "/data1/test/pretrained_models/eat_clean/checkpoints"
ALLOWLIST_DEFAULT = "results/stage21c_dcase2022_additional_integrity_audit/allowed_train_normal_filelist.csv"
STAGE21B_ROOT_DEFAULT = "results/stage21b_prior_year_dcase_dap_design"
STAGE21C_ROOT_DEFAULT = "results/stage21c_dcase2022_additional_integrity_audit"
OUTPUT_ROOT_DEFAULT = "results/stage21d_global_dap_ssl_sanity"
AUDIT_MD_DEFAULT = "docs/CTAF_STAGE21D_GLOBAL_DAP_SSL_SANITY_AUDIT.md"


@dataclass
class Stage21DTrainConfig:
    trainable_last_blocks: int = 1
    mask_ratio: float = 0.30
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 2
    val_ratio: float = 0.10
    anchor_loss_weight: float = 0.0
    norm_mean: float = -4.268
    norm_std: float = 4.569
    target_length: int = 1024
    num_mel_bins: int = 128
    sample_rate: int = 16000
    frame_shift_ms: int = 10


def parse_args():
    p = argparse.ArgumentParser(description="Stage21D global DAP SSL sanity")
    p.add_argument("--mode", required=True, choices=["check", "inspect_trainable", "train_global_ssl"])
    p.add_argument("--eat_model_path", default=CHECKPOINT_DEFAULT)
    p.add_argument("--allowlist_csv", default=ALLOWLIST_DEFAULT)
    p.add_argument("--seed", type=int, default=13711)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output_root", default=OUTPUT_ROOT_DEFAULT)
    p.add_argument("--doc_path", default=AUDIT_MD_DEFAULT)
    p.add_argument("--stage21b_root", default=STAGE21B_ROOT_DEFAULT)
    p.add_argument("--stage21c_root", default=STAGE21C_ROOT_DEFAULT)
    p.add_argument("--trainable_last_blocks", type=int, default=1)
    p.add_argument("--mask_ratio", type=float, default=0.30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--anchor_loss_weight", type=float, default=0.0)
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
    for name in ["torch", "torchaudio", "soundfile", "numpy", "scipy", "sklearn", "safetensors"]:
        try:
            __import__(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def load_allowlist_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    clean = []
    for row in rows:
        row = dict(row)
        row["path"] = str(row["path"])
        row["machine"] = str(row["machine"]).lower()
        row["split_group"] = str(row["split_group"]).lower()
        row["label_inferred"] = str(row["label_inferred"]).lower()
        row["domain_inferred"] = str(row["domain_inferred"]).lower()
        row["section"] = str(row["section"])
        row["allowed_train_normal"] = str(row["allowed_train_normal"]).lower() == "true"
        clean.append(row)
    return clean


def validate_allowlist_rows(rows: list[dict]):
    if not rows:
        raise RuntimeError("Allowlist is empty.")
    bad = []
    for row in rows:
        if not row["allowed_train_normal"]:
            bad.append((row["path"], "allowed_train_normal_false"))
        if row["split_group"] != "train":
            bad.append((row["path"], "non_train_split"))
        if row["label_inferred"] != "normal":
            bad.append((row["path"], "non_normal_label"))
    if bad:
        raise RuntimeError(f"Allowlist contains invalid rows, first few: {bad[:10]}")


def summarize_allowlist(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keyset = sorted({(r["machine"], r["domain_inferred"], r["section"]) for r in rows})
    by_group = []
    by_machine = []
    for machine, domain, section in keyset:
        rr = [r for r in rows if (r["machine"], r["domain_inferred"], r["section"]) == (machine, domain, section)]
        by_group.append({
            "machine": machine,
            "domain_inferred": domain,
            "section": section,
            "count": len(rr),
        })
    for machine in sorted({r["machine"] for r in rows}):
        rr = [r for r in rows if r["machine"] == machine]
        by_machine.append({
            "machine": machine,
            "count": len(rr),
            "source_count": sum(r["domain_inferred"] == "source" for r in rr),
            "target_count": sum(r["domain_inferred"] == "target" for r in rr),
        })
    return by_group, by_machine


def split_train_val(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    train_rows = []
    val_rows = []
    keyset = sorted({(r["machine"], r["domain_inferred"], r["section"]) for r in rows})
    for key in keyset:
        rr = [r for r in rows if (r["machine"], r["domain_inferred"], r["section"]) == key]
        total = len(rr)
        if total == 1:
            train_rows.extend(rr)
            continue
        val_count = max(1, int(total * val_ratio))
        val_count = min(val_count, total - 1)
        perm = rng.permutation(total)
        val_idx = set(int(x) for x in perm[:val_count])
        for idx, row in enumerate(rr):
            (val_rows if idx in val_idx else train_rows).append(row)
    return train_rows, val_rows


def load_clean_eat_local(model_path: Path, device: str):
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

    model.to(requested)
    model.eval()
    return model, requested


def freeze_all_parameters(model):
    for param in model.parameters():
        param.requires_grad_(False)


def enable_last_n_blocks(model, n_last: int) -> list[str]:
    block_names = []
    if n_last <= 0:
        return block_names
    if not hasattr(model, "model") or not hasattr(model.model, "blocks"):
        raise RuntimeError("Clean EAT model does not expose model.blocks for last-block training.")
    blocks = model.model.blocks
    total_blocks = len(blocks)
    start = max(0, total_blocks - n_last)
    for idx in range(start, total_blocks):
        block = blocks[idx]
        for param in block.parameters():
            param.requires_grad_(True)
        block_names.append(f"eat.model.blocks.{idx}")
    return block_names


def build_reconstruction_head(device):
    import torch.nn as nn

    head = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, 128),
    )
    head.to(device)
    return head


def parameter_audit_rows(model, recon_head) -> tuple[list[dict], list[dict]]:
    trainable_rows = []
    frozen_rows = []
    for scope, module in [("eat", model), ("recon_head", recon_head)]:
        for name, param in module.named_parameters():
            full_name = f"{scope}.{name}"
            row = {
                "parameter_name": full_name,
                "numel": int(param.numel()),
                "requires_grad": bool(param.requires_grad),
            }
            if param.requires_grad:
                trainable_rows.append(row)
            else:
                frozen_rows.append(row)
    return trainable_rows, frozen_rows


def summarize_trainable(trainable_rows: list[dict], frozen_rows: list[dict], allowed_block_names: list[str]) -> dict:
    trainable_total = sum(r["numel"] for r in trainable_rows)
    frozen_total = sum(r["numel"] for r in frozen_rows)
    total = trainable_total + frozen_total
    trainable_names = [r["parameter_name"] for r in trainable_rows]
    unexpected = []
    for name in trainable_names:
        if name.startswith("recon_head."):
            continue
        if not any(name.startswith(block_name) for block_name in allowed_block_names):
            unexpected.append(name)

    scope_counts = {}
    for name in trainable_names:
        prefix = ".".join(name.split(".")[:4]) if name.startswith("eat.model.blocks.") else ("recon_head" if name.startswith("recon_head.") else "other")
        scope_counts[prefix] = scope_counts.get(prefix, 0) + 1

    return {
        "total_parameter_count": int(total),
        "frozen_parameter_count": int(frozen_total),
        "trainable_parameter_count": int(trainable_total),
        "trainable_ratio": float(trainable_total / total) if total > 0 else 0.0,
        "allowed_block_names": allowed_block_names,
        "trainable_parameter_names_summary": trainable_names[:200],
        "trainable_scope_counts": scope_counts,
        "no_unexpected_earlier_layer_trainable_params": len(unexpected) == 0,
        "unexpected_trainable_parameter_names": unexpected,
    }


def load_waveform_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    import torch
    import torchaudio

    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    wav = wav.mean(axis=1)
    tensor = torch.tensor(wav, dtype=torch.float32)
    if int(sr) != 16000:
        tensor = torchaudio.functional.resample(tensor, int(sr), 16000)
    return tensor.detach().cpu().numpy().astype(np.float32)


def waveform_to_eat_tensor(wav_16k: np.ndarray, cfg: Stage21DTrainConfig):
    import torch
    import torchaudio

    wav = torch.tensor(wav_16k, dtype=torch.float32)
    mel = torchaudio.compliance.kaldi.fbank(
        wav.unsqueeze(0),
        htk_compat=True,
        sample_frequency=cfg.sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=cfg.num_mel_bins,
        dither=0.0,
        frame_shift=cfg.frame_shift_ms,
    )
    if mel.shape[0] < cfg.target_length:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, cfg.target_length - mel.shape[0]))
    else:
        mel = mel[:cfg.target_length, :]
    mel = (mel - cfg.norm_mean) / (cfg.norm_std * 2)
    return mel.to(dtype=torch.float32)


def build_batch(paths: list[Path], cfg: Stage21DTrainConfig, device):
    import torch

    feats = [waveform_to_eat_tensor(load_waveform_16k(path), cfg) for path in paths]
    mel = torch.stack(feats, dim=0)
    return mel.to(device)


def random_time_mask(mel, mask_ratio: float, rng: np.random.Generator):
    import torch

    bsz, n_frames, _ = mel.shape
    n_mask = max(1, int(n_frames * mask_ratio))
    mask = torch.zeros((bsz, n_frames), dtype=torch.bool, device=mel.device)
    for b in range(bsz):
        idx = rng.choice(n_frames, size=n_mask, replace=False)
        mask[b, idx] = True
    masked = mel.clone()
    masked[mask] = 0.0
    return masked, mask


def extract_frame_output(model, mel_batch, device):
    import torch

    x = mel_batch.unsqueeze(1).to(device)
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
    return feat


def pool_p1_mean_std(frames):
    import torch

    mean = frames.mean(dim=1)
    std = frames.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=1)


def compute_loss(student, teacher, recon_head, mel_batch, cfg: Stage21DTrainConfig, device, rng: np.random.Generator):
    import torch
    import torch.nn.functional as F

    masked_mel, time_mask = random_time_mask(mel_batch, cfg.mask_ratio, rng)
    student_frames = extract_frame_output(student, masked_mel, device)
    pred_mel = recon_head(student_frames)
    target_mel = mel_batch

    t_out = int(pred_mel.shape[1])
    target_down = F.interpolate(
        target_mel.transpose(1, 2),
        size=t_out,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    mask_down = F.interpolate(
        time_mask.float().unsqueeze(1),
        size=t_out,
        mode="nearest",
    ).squeeze(1).bool()
    if not torch.any(mask_down):
        mask_down[:, 0] = True

    recon_loss = F.mse_loss(pred_mel[mask_down], target_down[mask_down])
    total_loss = recon_loss
    anchor = torch.zeros([], device=device)
    if cfg.anchor_loss_weight > 0:
        with torch.no_grad():
            teacher_frames = extract_frame_output(teacher, mel_batch, device)
            teacher_p1 = F.normalize(pool_p1_mean_std(teacher_frames), dim=1)
        student_p1 = F.normalize(pool_p1_mean_std(student_frames), dim=1)
        anchor = (1.0 - F.cosine_similarity(student_p1, teacher_p1, dim=1)).mean()
        total_loss = total_loss + cfg.anchor_loss_weight * anchor

    return {
        "total": total_loss,
        "masked_recon_loss": recon_loss,
        "anchor_loss": anchor,
    }


def run_check(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    allowlist_path = Path(args.allowlist_csv)
    model_path = Path(args.eat_model_path)
    stage21b_root = Path(args.stage21b_root)
    stage21c_root = Path(args.stage21c_root)

    rows = load_allowlist_rows(allowlist_path) if allowlist_path.exists() else []
    by_group, by_machine = summarize_allowlist(rows) if rows else ([], [])
    status = {
        "mode": "check",
        "eat_model_path_exists": model_path.exists(),
        "allowlist_csv_exists": allowlist_path.exists(),
        "stage21b_root_exists": stage21b_root.exists(),
        "stage21c_root_exists": stage21c_root.exists(),
        "stage21b_protocol_exists": (stage21b_root / "stage21b_protocol_freeze.json").exists(),
        "stage21c_integrity_exists": (stage21c_root / "integrity_verdict.json").exists(),
        "allowlist_count": len(rows),
        "imports": import_status(),
        "notes": "No model load, no forward, no training, no embedding extraction, no scoring.",
    }
    write_json(out_root / "check_status.json", status)
    write_csv(out_root / "allowlist_group_summary.csv", by_group)
    write_csv(out_root / "allowlist_machine_summary.csv", by_machine)


def run_inspect_trainable(args):
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.eat_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")

    cfg = Stage21DTrainConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        mask_ratio=args.mask_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        anchor_loss_weight=args.anchor_loss_weight,
    )

    model, device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(model)
    allowed_blocks = enable_last_n_blocks(model, cfg.trainable_last_blocks)
    recon_head = build_reconstruction_head(device)

    trainable_rows, frozen_rows = parameter_audit_rows(model, recon_head)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)

    write_json(out_root / "model_load_status.json", {
        "mode": "inspect_trainable",
        "eat_model_path": str(model_path),
        "device": str(device),
        "model_class": model.__class__.__name__,
        "trainable_last_blocks": cfg.trainable_last_blocks,
        "allowed_block_names": allowed_blocks,
    })
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)


def run_train_global_ssl(args):
    import torch

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    allowlist_path = Path(args.allowlist_csv)
    model_path = Path(args.eat_model_path)
    stage21c_root = Path(args.stage21c_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing clean EAT checkpoint path: {model_path}")
    if not allowlist_path.exists():
        raise FileNotFoundError(f"Missing allowlist csv: {allowlist_path}")

    integrity = json.loads((stage21c_root / "integrity_verdict.json").read_text())
    if not integrity.get("ready_for_stage21d"):
        raise RuntimeError("Stage21C integrity verdict is not ready_for_stage21d.")

    rows = load_allowlist_rows(allowlist_path)
    validate_allowlist_rows(rows)
    train_rows, val_rows = split_train_val(rows, Stage21DTrainConfig.val_ratio, args.seed)

    cfg = Stage21DTrainConfig(
        trainable_last_blocks=args.trainable_last_blocks,
        mask_ratio=args.mask_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        anchor_loss_weight=args.anchor_loss_weight,
    )
    set_seed(args.seed)
    if not train_rows or not val_rows:
        raise RuntimeError(f"Invalid global train/val split: train={len(train_rows)} val={len(val_rows)}")

    split_inventory = []
    for row in train_rows:
        split_inventory.append({**row, "split_assignment": "train"})
    for row in val_rows:
        split_inventory.append({**row, "split_assignment": "val"})
    write_csv(out_root / "global_split_inventory.csv", split_inventory)

    teacher, device = load_clean_eat_local(model_path, args.device)
    freeze_all_parameters(teacher)
    teacher.eval()

    student, _ = load_clean_eat_local(model_path, str(device))
    freeze_all_parameters(student)
    allowed_blocks = enable_last_n_blocks(student, cfg.trainable_last_blocks)
    recon_head = build_reconstruction_head(device)

    trainable_rows, frozen_rows = parameter_audit_rows(student, recon_head)
    summary = summarize_trainable(trainable_rows, frozen_rows, allowed_blocks)
    write_json(out_root / "model_load_status.json", {
        "mode": "train_global_ssl",
        "eat_model_path": str(model_path),
        "allowlist_csv": str(allowlist_path),
        "device": str(device),
        "model_class": student.__class__.__name__,
        "trainable_last_blocks": cfg.trainable_last_blocks,
        "allowed_block_names": allowed_blocks,
    })
    write_csv(out_root / "trainable_parameter_audit.csv", trainable_rows)
    write_csv(out_root / "frozen_parameter_audit.csv", frozen_rows)
    write_json(out_root / "trainable_summary.json", summary)
    if not summary["no_unexpected_earlier_layer_trainable_params"]:
        raise RuntimeError(f"Unexpected trainable parameters: {summary['unexpected_trainable_parameter_names']}")

    trainable_params = [p for p in student.parameters() if p.requires_grad] + list(recon_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_path = out_root / "checkpoints" / "stage21d_global_best.pt"
    best_val = float("inf")
    best_epoch = -1
    patience = 0
    log_rows = []

    train_batches = [train_rows[i:i + cfg.batch_size] for i in range(0, len(train_rows), cfg.batch_size)]
    val_batches = [val_rows[i:i + cfg.batch_size] for i in range(0, len(val_rows), cfg.batch_size)]

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        recon_head.train()
        train_loss_sum = 0.0
        train_mask_sum = 0.0
        train_anchor_sum = 0.0
        train_count = 0
        rng = np.random.default_rng(args.seed + epoch * 1000)

        for batch in train_batches:
            paths = [Path(row["path"]) for row in batch]
            mel_batch = build_batch(paths, cfg, device)
            optimizer.zero_grad(set_to_none=True)
            loss_dict = compute_loss(student, teacher, recon_head, mel_batch, cfg, device, rng)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            optimizer.step()

            bsz = len(batch)
            train_loss_sum += float(loss_dict["total"].item()) * bsz
            train_mask_sum += float(loss_dict["masked_recon_loss"].item()) * bsz
            train_anchor_sum += float(loss_dict["anchor_loss"].item()) * bsz
            train_count += bsz

        student.eval()
        recon_head.eval()
        val_loss_sum = 0.0
        val_mask_sum = 0.0
        val_anchor_sum = 0.0
        val_count = 0
        val_rng = np.random.default_rng(args.seed + 500000 + epoch)
        with torch.no_grad():
            for batch in val_batches:
                paths = [Path(row["path"]) for row in batch]
                mel_batch = build_batch(paths, cfg, device)
                loss_dict = compute_loss(student, teacher, recon_head, mel_batch, cfg, device, val_rng)
                bsz = len(batch)
                val_loss_sum += float(loss_dict["total"].item()) * bsz
                val_mask_sum += float(loss_dict["masked_recon_loss"].item()) * bsz
                val_anchor_sum += float(loss_dict["anchor_loss"].item()) * bsz
                val_count += bsz

        train_loss = train_loss_sum / max(train_count, 1)
        train_mask_loss = train_mask_sum / max(train_count, 1)
        train_anchor = train_anchor_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_mask_loss = val_mask_sum / max(val_count, 1)
        val_anchor = val_anchor_sum / max(val_count, 1)
        log_rows.append({
            "epoch": epoch,
            "train_total_loss": train_loss,
            "train_masked_recon_loss": train_mask_loss,
            "train_anchor_loss": train_anchor,
            "val_total_loss": val_loss,
            "val_masked_recon_loss": val_mask_loss,
            "val_anchor_loss": val_anchor,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_epoch = epoch
            patience = 0
            torch.save({
                "student_state_dict": student.state_dict(),
                "recon_head_state_dict": recon_head.state_dict(),
                "train_config": asdict(cfg),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "allowed_block_names": allowed_blocks,
                "trainable_summary": summary,
                "allowlist_csv": str(allowlist_path),
            }, best_path)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                break

    group_rows, machine_rows = summarize_allowlist(rows)
    write_csv(out_root / "allowlist_group_summary.csv", group_rows)
    write_csv(out_root / "allowlist_machine_summary.csv", machine_rows)
    write_csv(out_root / "train_log.csv", log_rows)
    write_json(out_root / "stage21d_train_config.json", asdict(cfg))
    write_json(out_root / "stage21d_training_verdict.json", {
        "allowlist_csv": str(allowlist_path),
        "total_allowlist_count": len(rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "trainable_last_blocks": cfg.trainable_last_blocks,
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "no_unexpected_earlier_layer_trainable_params": summary["no_unexpected_earlier_layer_trainable_params"],
        "notes": [
            "Training scaffold only.",
            "No ASD scoring performed in Stage21D.",
            "Stage21E should evaluate fan first, then gearbox, with the frozen min_clamp__mean + ldn_diff_K1_scorek5 backend.",
        ],
    })

    write_markdown(Path(args.doc_path), [
        "# CTAF Stage21D Global DAP SSL Sanity Audit",
        "",
        f"- allowlist csv: `{allowlist_path}`",
        f"- total allowlist count: `{len(rows)}`",
        f"- train count: `{len(train_rows)}`",
        f"- val count: `{len(val_rows)}`",
        f"- best checkpoint: `{best_path}`",
        f"- best epoch: `{best_epoch}`",
        f"- best validation loss: `{best_val:.8f}`",
        "",
        "No ASD scoring was performed in this stage.",
    ])


def main():
    args = parse_args()
    if args.mode == "check":
        run_check(args)
        return
    if args.mode == "inspect_trainable":
        run_inspect_trainable(args)
        return
    if args.mode == "train_global_ssl":
        run_train_global_ssl(args)
        return
    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
