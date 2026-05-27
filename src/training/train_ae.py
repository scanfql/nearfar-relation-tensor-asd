"""Training loop for Conv2D AE on normal-only data."""
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_autoencoder(
    model: nn.Module,
    train_data: np.ndarray,    # [N, V, F, T]
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 13711,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
    noise_mode: str = "none",
    mask_prob: float = 0.0,
    loss_view_mode: str = "all_views",
    ssl_task: str = "reconstruction",
    tail_compact_loss: bool = False,
    tail_compact_lambda: float = 0.01,
    tail_compact_top_frac: float = 0.10,
    hard_normal_flags: np.ndarray | None = None,
    hard_normal_consistency_loss: bool = False,
    hard_normal_consistency_lambda: float = 0.10,
    hard_normal_time_mask_frac: float = 0.10,
    hard_normal_freq_mask_frac: float = 0.10,
) -> dict:
    """Train autoencoder on normal-only data. Returns stats dict."""
    from src.utils.seed import set_seed
    set_seed(seed)

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)
    if noise_mode not in ("none", "tf_mask", "view_dropout"):
        raise ValueError(f"unknown noise_mode: {noise_mode}")
    if loss_view_mode not in ("all_views", "logmel_only", "masked_view_only", "future_only"):
        raise ValueError(f"unknown loss_view_mode: {loss_view_mode}")
    if ssl_task not in ("reconstruction", "masked_cross_view_prediction", "temporal_predictive_ssl"):
        raise ValueError(f"unknown ssl_task: {ssl_task}")
    if not 0.0 <= mask_prob < 1.0:
        raise ValueError("mask_prob must be in [0, 1)")
    if tail_compact_lambda < 0.0:
        raise ValueError("tail_compact_lambda must be >= 0")
    if not 0.0 < tail_compact_top_frac <= 1.0:
        raise ValueError("tail_compact_top_frac must be in (0, 1]")
    if hard_normal_consistency_lambda < 0.0:
        raise ValueError("hard_normal_consistency_lambda must be >= 0")
    if not 0.0 < hard_normal_time_mask_frac <= 1.0:
        raise ValueError("hard_normal_time_mask_frac must be in (0, 1]")
    if not 0.0 < hard_normal_freq_mask_frac <= 1.0:
        raise ValueError("hard_normal_freq_mask_frac must be in (0, 1]")
    if hard_normal_flags is not None and len(hard_normal_flags) != len(train_data):
        raise ValueError("hard_normal_flags length must match train_data length")

    # Split 80/20 for train/val (val used only for loss monitoring, not model selection)
    n = len(train_data)
    n_val = max(1, n // 5)
    indices = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    if hard_normal_flags is None:
        hard_normal_flags = np.zeros((n,), dtype=np.float32)
    train_ds = TensorDataset(
        torch.from_numpy(train_data[train_idx]),
        torch.from_numpy(hard_normal_flags[train_idx].astype(np.float32)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(train_data[val_idx]),
        torch.from_numpy(hard_normal_flags[val_idx].astype(np.float32)),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    stats = {
        "train_loss": [],
        "val_loss": [],
        "epochs": epochs,
        "train_time_sec": 0,
        "early_stopping": bool(early_stopping),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "stopped_early": False,
        "noise_mode": noise_mode,
        "mask_prob": float(mask_prob),
        "loss_view_mode": loss_view_mode,
        "ssl_task": ssl_task,
        "tail_compact_loss": bool(tail_compact_loss),
        "tail_compact_lambda": float(tail_compact_lambda),
        "tail_compact_top_frac": float(tail_compact_top_frac),
        "hard_normal_consistency_loss": bool(hard_normal_consistency_loss),
        "hard_normal_consistency_lambda": float(hard_normal_consistency_lambda),
        "hard_normal_time_mask_frac": float(hard_normal_time_mask_frac),
        "hard_normal_freq_mask_frac": float(hard_normal_freq_mask_frac),
        "hard_normal_train_count": int(np.sum(hard_normal_flags[train_idx] > 0.5)),
        "hard_normal_val_count": int(np.sum(hard_normal_flags[val_idx] > 0.5)),
    }
    t0 = time.time()
    best_state = None
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, hard_flags in train_loader:
            x_batch = x_batch.to(device)
            hard_flags = hard_flags.to(device)
            x_input, task_meta = _prepare_ssl_input(
                x_batch, ssl_task=ssl_task, noise_mode=noise_mode, mask_prob=mask_prob
            )
            optimizer.zero_grad()
            recon = model(x_input)
            loss = _compute_loss(criterion, recon, x_batch, loss_view_mode, task_meta)
            if tail_compact_loss:
                latent = model.encode(x_batch)
                tail_loss = _compute_tail_compactness_loss(latent, tail_compact_top_frac)
                loss = loss + tail_compact_lambda * tail_loss
            if hard_normal_consistency_loss:
                cons_loss = _compute_hard_normal_consistency_loss(
                    model=model,
                    x_batch=x_batch,
                    hard_flags=hard_flags,
                    time_mask_frac=hard_normal_time_mask_frac,
                    freq_mask_frac=hard_normal_freq_mask_frac,
                )
                loss = loss + hard_normal_consistency_lambda * cons_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        stats["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, hard_flags in val_loader:
                x_batch = x_batch.to(device)
                hard_flags = hard_flags.to(device)
                x_input, task_meta = _prepare_ssl_input(
                    x_batch, ssl_task=ssl_task, noise_mode=noise_mode, mask_prob=mask_prob
                )
                recon = model(x_input)
                loss = _compute_loss(criterion, recon, x_batch, loss_view_mode, task_meta)
                if tail_compact_loss:
                    latent = model.encode(x_batch)
                    tail_loss = _compute_tail_compactness_loss(latent, tail_compact_top_frac)
                    loss = loss + tail_compact_lambda * tail_loss
                if hard_normal_consistency_loss:
                    cons_loss = _compute_hard_normal_consistency_loss(
                        model=model,
                        x_batch=x_batch,
                        hard_flags=hard_flags,
                        time_mask_frac=hard_normal_time_mask_frac,
                        freq_mask_frac=hard_normal_freq_mask_frac,
                    )
                    loss = loss + hard_normal_consistency_lambda * cons_loss
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        stats["val_loss"].append(val_loss)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            stats["best_epoch"] = epoch + 1
            stats["best_val_loss"] = best_val
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            stats["stopped_early"] = True
            break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    stats["train_time_sec"] = time.time() - t0
    stats["final_train_loss"] = stats["train_loss"][-1]
    stats["final_val_loss"] = stats["val_loss"][-1]
    stats["epochs_ran"] = len(stats["train_loss"])
    stats["model_params"] = sum(p.numel() for p in model.parameters())
    return stats


def train_teacher_student_local_distillation(
    teacher: nn.Module,
    student: nn.Module,
    train_data: np.ndarray,    # [N, V, F, T]
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 13711,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
) -> dict:
    """Train student to predict frozen teacher local latent features."""
    from src.utils.seed import set_seed
    set_seed(seed)

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n = len(train_data)
    n_val = max(1, n // 5)
    indices = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_ds = TensorDataset(torch.from_numpy(train_data[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(train_data[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr, weight_decay=weight_decay)

    stats = {
        "train_loss": [],
        "val_loss": [],
        "epochs": epochs,
        "train_time_sec": 0,
        "early_stopping": bool(early_stopping),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "stopped_early": False,
        "noise_mode": "teacher_student_tf_block_mask",
        "mask_prob": 0.0,
        "loss_view_mode": "teacher_student_local_distill",
        "ssl_task": "teacher_student_local_distillation",
        "tail_compact_loss": False,
        "tail_compact_lambda": 0.0,
        "tail_compact_top_frac": 0.0,
    }
    t0 = time.time()
    best_state = None
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            x_corrupt = _apply_distill_tf_block_mask(x_batch, deterministic=False)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_latent = teacher.encode(x_batch)
            student_latent = student.encode(x_corrupt)
            loss = _compute_local_distill_loss(student_latent, teacher_latent)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)
        stats["train_loss"].append(train_loss)

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                x_corrupt = _apply_distill_tf_block_mask(x_batch, deterministic=True)
                teacher_latent = teacher.encode(x_batch)
                student_latent = student.encode(x_corrupt)
                loss = _compute_local_distill_loss(student_latent, teacher_latent)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        stats["val_loss"].append(val_loss)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            stats["best_epoch"] = epoch + 1
            stats["best_val_loss"] = best_val
            best_state = copy.deepcopy(student.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            stats["stopped_early"] = True
            break

    if early_stopping and best_state is not None:
        student.load_state_dict(best_state)

    stats["train_time_sec"] = time.time() - t0
    stats["final_train_loss"] = stats["train_loss"][-1]
    stats["final_val_loss"] = stats["val_loss"][-1]
    stats["epochs_ran"] = len(stats["train_loss"])
    stats["teacher_model_params"] = sum(p.numel() for p in teacher.parameters())
    stats["student_model_params"] = sum(p.numel() for p in student.parameters())
    stats["model_params"] = stats["teacher_model_params"] + stats["student_model_params"]
    return stats


def train_vicreg_invariance_ssl(
    model: nn.Module,
    train_data: np.ndarray,    # [N, V, F, T]
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    seed: int = 13711,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.0,
) -> dict:
    """Train frozen-protocol VICReg invariance SSL on full_5view."""
    from src.utils.seed import set_seed
    set_seed(seed)

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)

    n = len(train_data)
    n_val = max(1, n // 5)
    indices = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_ds = TensorDataset(torch.from_numpy(train_data[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(train_data[val_idx]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    projector = nn.Sequential(
        nn.Linear(192, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 256),
    ).to(device)

    lambda_inv = 25.0
    lambda_var = 25.0
    lambda_cov = 1.0
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    stats = {
        "train_loss": [],
        "val_loss": [],
        "epochs": epochs,
        "train_time_sec": 0,
        "early_stopping": bool(early_stopping),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "stopped_early": False,
        "noise_mode": "vicreg_fixed_augments",
        "mask_prob": 0.0,
        "loss_view_mode": "vicreg_only",
        "ssl_task": "vicreg_invariance_ssl",
        "tail_compact_loss": False,
        "tail_compact_lambda": 0.0,
        "tail_compact_top_frac": 0.0,
        "vicreg_lambda_inv": lambda_inv,
        "vicreg_lambda_var": lambda_var,
        "vicreg_lambda_cov": lambda_cov,
        "vicreg_time_mask_frac": 0.05,
        "vicreg_freq_mask_frac": 0.05,
        "vicreg_gain_min": 0.95,
        "vicreg_gain_max": 1.05,
        "vicreg_n_augmentations_per_view": 2,
    }

    t0 = time.time()
    best_state = None
    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        projector.train()
        train_loss = 0.0
        train_inv = 0.0
        train_var = 0.0
        train_cov = 0.0
        train_lat_std = 0.0
        train_proj_std = 0.0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            x1, x2 = _make_vicreg_views(x_batch, deterministic=False)
            optimizer.zero_grad()
            z1 = _latent_summary_tensor(model.encode(x1))
            z2 = _latent_summary_tensor(model.encode(x2))
            p1 = projector(z1)
            p2 = projector(z2)
            inv_loss = F.mse_loss(p1, p2)
            var_loss = 0.5 * (_vicreg_variance_loss(p1) + _vicreg_variance_loss(p2))
            cov_loss = 0.5 * (_vicreg_covariance_loss(p1) + _vicreg_covariance_loss(p2))
            loss = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss
            loss.backward()
            optimizer.step()

            bsz = x_batch.size(0)
            train_loss += loss.item() * bsz
            train_inv += inv_loss.item() * bsz
            train_var += var_loss.item() * bsz
            train_cov += cov_loss.item() * bsz
            train_lat_std += 0.5 * (_feature_std_mean(z1).item() + _feature_std_mean(z2).item()) * bsz
            train_proj_std += 0.5 * (_feature_std_mean(p1).item() + _feature_std_mean(p2).item()) * bsz

        train_loss /= len(train_loader.dataset)
        stats["train_loss"].append(train_loss)

        model.eval()
        projector.eval()
        val_loss = 0.0
        val_inv = 0.0
        val_var = 0.0
        val_cov = 0.0
        val_lat_std = 0.0
        val_proj_std = 0.0
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                x1, x2 = _make_vicreg_views(x_batch, deterministic=True)
                z1 = _latent_summary_tensor(model.encode(x1))
                z2 = _latent_summary_tensor(model.encode(x2))
                p1 = projector(z1)
                p2 = projector(z2)
                inv_loss = F.mse_loss(p1, p2)
                var_loss = 0.5 * (_vicreg_variance_loss(p1) + _vicreg_variance_loss(p2))
                cov_loss = 0.5 * (_vicreg_covariance_loss(p1) + _vicreg_covariance_loss(p2))
                loss = lambda_inv * inv_loss + lambda_var * var_loss + lambda_cov * cov_loss

                bsz = x_batch.size(0)
                val_loss += loss.item() * bsz
                val_inv += inv_loss.item() * bsz
                val_var += var_loss.item() * bsz
                val_cov += cov_loss.item() * bsz
                val_lat_std += 0.5 * (_feature_std_mean(z1).item() + _feature_std_mean(z2).item()) * bsz
                val_proj_std += 0.5 * (_feature_std_mean(p1).item() + _feature_std_mean(p2).item()) * bsz
        val_loss /= len(val_loader.dataset)
        stats["val_loss"].append(val_loss)

        if val_loss < best_val - min_delta:
            best_val = val_loss
            stats["best_epoch"] = epoch + 1
            stats["best_val_loss"] = best_val
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "projector": copy.deepcopy(projector.state_dict()),
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping and epochs_without_improvement >= patience:
            stats["stopped_early"] = True
            break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])

    n_train = len(train_loader.dataset)
    n_valset = len(val_loader.dataset)
    stats["train_time_sec"] = time.time() - t0
    stats["final_train_loss"] = stats["train_loss"][-1]
    stats["final_val_loss"] = stats["val_loss"][-1]
    stats["epochs_ran"] = len(stats["train_loss"])
    stats["model_params"] = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in projector.parameters())
    stats["projector_params"] = sum(p.numel() for p in projector.parameters())
    stats["vicreg_train_inv_loss"] = train_inv / n_train
    stats["vicreg_train_var_loss"] = train_var / n_train
    stats["vicreg_train_cov_loss"] = train_cov / n_train
    stats["vicreg_train_latent_summary_std"] = train_lat_std / n_train
    stats["vicreg_train_projector_std"] = train_proj_std / n_train
    stats["vicreg_val_inv_loss"] = val_inv / n_valset
    stats["vicreg_val_var_loss"] = val_var / n_valset
    stats["vicreg_val_cov_loss"] = val_cov / n_valset
    stats["vicreg_val_latent_summary_std"] = val_lat_std / n_valset
    stats["vicreg_val_projector_std"] = val_proj_std / n_valset
    return stats


def _apply_training_noise(x_batch: torch.Tensor, noise_mode: str, mask_prob: float) -> torch.Tensor:
    """Apply fixed-rule train-time corruption for denoising AE training.

    The target remains the original normal tensor. This uses no labels and is
    disabled at scoring time.
    """
    if noise_mode == "none" or mask_prob <= 0.0:
        return x_batch
    if noise_mode == "tf_mask":
        keep = torch.rand_like(x_batch) >= mask_prob
        return x_batch * keep.to(dtype=x_batch.dtype)
    if noise_mode == "view_dropout":
        keep = (torch.rand(
            x_batch.size(0), x_batch.size(1), 1, 1, device=x_batch.device
        ) >= mask_prob)
        if x_batch.size(1) > 1:
            missing = ~keep.squeeze(-1).squeeze(-1).any(dim=1)
            if missing.any():
                restore_idx = torch.randint(
                    low=0,
                    high=x_batch.size(1),
                    size=(int(missing.sum().item()),),
                    device=x_batch.device,
                )
                missing_rows = torch.nonzero(missing, as_tuple=False).squeeze(1)
                keep[missing_rows, restore_idx, 0, 0] = True
        return x_batch * keep.to(dtype=x_batch.dtype)
    raise ValueError(f"unknown noise_mode: {noise_mode}")


def _apply_small_time_mask(x: torch.Tensor, frac: float, deterministic: bool) -> torch.Tensor:
    out = x.clone()
    t = out.shape[-1]
    width = max(1, int(np.ceil(t * frac)))
    if deterministic:
        t0 = max(0, (t - width) // 2)
        out[..., t0:t0 + width] = 0.0
        return out
    max_t0 = max(1, t - width + 1)
    t0 = torch.randint(low=0, high=max_t0, size=(out.shape[0],), device=out.device)
    for i in range(out.shape[0]):
        out[i, :, :, t0[i]:t0[i] + width] = 0.0
    return out


def _apply_small_freq_mask(x: torch.Tensor, frac: float, deterministic: bool) -> torch.Tensor:
    out = x.clone()
    f = out.shape[-2]
    width = max(1, int(np.ceil(f * frac)))
    if deterministic:
        f0 = max(0, (f - width) // 2)
        out[..., f0:f0 + width, :] = 0.0
        return out
    max_f0 = max(1, f - width + 1)
    f0 = torch.randint(low=0, high=max_f0, size=(out.shape[0],), device=out.device)
    for i in range(out.shape[0]):
        out[i, :, f0[i]:f0[i] + width, :] = 0.0
    return out


def _apply_gain_jitter(x: torch.Tensor, lo: float, hi: float, deterministic: bool) -> torch.Tensor:
    if deterministic:
        gain = torch.full((x.shape[0], 1, 1, 1), (lo + hi) / 2.0, device=x.device, dtype=x.dtype)
    else:
        gain = torch.empty((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(lo, hi)
    return x * gain


def _make_vicreg_views(x_batch: torch.Tensor, deterministic: bool):
    """Create exactly two augmented views with two augmentations per view."""
    x1 = _apply_gain_jitter(_apply_small_time_mask(x_batch, 0.05, deterministic), 0.95, 1.05, deterministic)
    x2 = _apply_gain_jitter(_apply_small_freq_mask(x_batch, 0.05, deterministic), 0.95, 1.05, deterministic)
    return x1, x2


def _prepare_ssl_input(
    x_batch: torch.Tensor,
    ssl_task: str,
    noise_mode: str,
    mask_prob: float,
):
    """Prepare training input and optional masked-view indices."""
    x_input = _apply_training_noise(x_batch, noise_mode, mask_prob)
    task_meta = {}
    if ssl_task == "masked_cross_view_prediction":
        b, v, _, _ = x_input.shape
        masked_view_idx = torch.randint(low=0, high=v, size=(b,), device=x_input.device)
        x_input = x_input.clone()
        x_input[torch.arange(b, device=x_input.device), masked_view_idx] = 0.0
        task_meta["masked_view_idx"] = masked_view_idx
    elif ssl_task == "temporal_predictive_ssl":
        t = x_input.shape[-1]
        future_start = max(1, t // 2)
        x_input = x_input.clone()
        x_input[..., future_start:] = 0.0
        task_meta["future_start"] = future_start
    return x_input, task_meta


def _apply_distill_tf_block_mask(
    x_batch: torch.Tensor,
    deterministic: bool = False,
    freq_frac: float = 0.25,
    time_frac: float = 0.25,
) -> torch.Tensor:
    """Apply fixed-rule TF block masking used by local teacher-student SSL."""
    x = x_batch.clone()
    _, _, f, t = x.shape
    block_f = max(1, int(np.ceil(f * freq_frac)))
    block_t = max(1, int(np.ceil(t * time_frac)))

    if deterministic:
        f0 = max(0, (f - block_f) // 2)
        t0 = max(0, (t - block_t) // 2)
        x[:, :, f0:f0 + block_f, t0:t0 + block_t] = 0.0
        return x

    max_f0 = max(1, f - block_f + 1)
    max_t0 = max(1, t - block_t + 1)
    f0 = torch.randint(low=0, high=max_f0, size=(x.shape[0],), device=x.device)
    t0 = torch.randint(low=0, high=max_t0, size=(x.shape[0],), device=x.device)
    for i in range(x.shape[0]):
        x[i, :, f0[i]:f0[i] + block_f, t0[i]:t0[i] + block_t] = 0.0
    return x


def _compute_loss(
    criterion: nn.Module,
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_view_mode: str,
    task_meta=None,
) -> torch.Tensor:
    task_meta = task_meta or {}
    if loss_view_mode == "all_views":
        return criterion(recon, target)
    if loss_view_mode == "logmel_only":
        return criterion(recon[:, :1], target[:, :1])
    if loss_view_mode == "masked_view_only":
        masked_view_idx = task_meta.get("masked_view_idx")
        if masked_view_idx is None:
            raise ValueError("masked_view_only requires masked_view_idx")
        batch_idx = torch.arange(target.shape[0], device=target.device)
        pred = recon[batch_idx, masked_view_idx].unsqueeze(1)
        truth = target[batch_idx, masked_view_idx].unsqueeze(1)
        return criterion(pred, truth)
    if loss_view_mode == "future_only":
        future_start = task_meta.get("future_start")
        if future_start is None:
            raise ValueError("future_only requires future_start")
        return criterion(recon[..., future_start:], target[..., future_start:])
    raise ValueError(f"unknown loss_view_mode: {loss_view_mode}")


def _compute_local_distill_loss(
    student_latent: torch.Tensor,
    teacher_latent: torch.Tensor,
) -> torch.Tensor:
    """Mean squared distance between normalized local teacher/student features."""
    student_norm = F.normalize(student_latent, dim=1)
    teacher_norm = F.normalize(teacher_latent.detach(), dim=1)
    return ((student_norm - teacher_norm) ** 2).sum(dim=1).mean()


def _compute_tail_compactness_loss(
    latent: torch.Tensor,
    top_frac: float,
) -> torch.Tensor:
    """Compact the upper tail of batch latent-summary distances to batch center."""
    mean = latent.mean(dim=(2, 3))
    std = latent.std(dim=(2, 3), unbiased=False)
    max_vals = latent.amax(dim=(2, 3))
    summary = torch.cat([mean, std, max_vals], dim=1)
    center = summary.mean(dim=0, keepdim=True)
    dist = torch.norm(summary - center, dim=1)
    n_top = max(1, int(np.ceil(dist.shape[0] * top_frac)))
    return torch.topk(dist, k=n_top, largest=True).values.mean()


def _compute_hard_normal_consistency_loss(
    model: nn.Module,
    x_batch: torch.Tensor,
    hard_flags: torch.Tensor,
    time_mask_frac: float,
    freq_mask_frac: float,
) -> torch.Tensor:
    """Consistency loss applied only to fixed hard-normal samples."""
    mask = hard_flags > 0.5
    if not mask.any():
        return x_batch.new_zeros(())
    x_hard = x_batch[mask]
    z = _latent_summary_tensor(model.encode(x_hard))
    z_t = _latent_summary_tensor(model.encode(_apply_fixed_time_mask(x_hard, time_mask_frac)))
    z_f = _latent_summary_tensor(model.encode(_apply_fixed_freq_mask(x_hard, freq_mask_frac)))
    z_n = F.normalize(z, dim=1)
    zt_n = F.normalize(z_t, dim=1)
    zf_n = F.normalize(z_f, dim=1)
    loss_t = 1.0 - torch.sum(z_n * zt_n, dim=1)
    loss_f = 1.0 - torch.sum(z_n * zf_n, dim=1)
    return 0.5 * (loss_t.mean() + loss_f.mean())


def _latent_summary_tensor(latent: torch.Tensor) -> torch.Tensor:
    mean = latent.mean(dim=(2, 3))
    std = latent.std(dim=(2, 3), unbiased=False)
    max_vals = latent.amax(dim=(2, 3))
    return torch.cat([mean, std, max_vals], dim=1)


def _feature_std_mean(x: torch.Tensor) -> torch.Tensor:
    return x.std(dim=0, unbiased=False).mean()


def _vicreg_variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(1.0 - std))


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("covariance matrix must be square")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _vicreg_covariance_loss(z: torch.Tensor) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(z.shape[0] - 1, 1)
    return (_off_diagonal(cov) ** 2).mean()


def _apply_fixed_time_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    out = x.clone()
    t = out.shape[-1]
    width = max(1, int(np.ceil(t * frac)))
    t0 = max(0, (t - width) // 2)
    out[..., t0:t0 + width] = 0.0
    return out


def _apply_fixed_freq_mask(x: torch.Tensor, frac: float) -> torch.Tensor:
    out = x.clone()
    f = out.shape[-2]
    width = max(1, int(np.ceil(f * frac)))
    f0 = max(0, (f - width) // 2)
    out[..., f0:f0 + width, :] = 0.0
    return out
