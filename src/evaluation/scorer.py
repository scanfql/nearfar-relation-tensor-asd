"""Reconstruction-error anomaly scoring."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def compute_recon_errors(
    model,
    data: np.ndarray,   # [N, V, F, T]
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Compute per-file mean squared reconstruction error. Returns [N]."""
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            recon = model.reconstruct(x_batch)
            # Per-sample mean squared error
            mse = ((recon - x_batch) ** 2).mean(dim=(1, 2, 3))
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores).astype(np.float64)


def compute_recon_error_views(
    model,
    data: np.ndarray,   # [N, V, F, T]
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Compute per-file, per-view reconstruction MSE. Returns [N, V].

    This exposes fixed view-wise residuals for label-free score diagnostics.
    Labels are not used here.
    """
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            recon = model.reconstruct(x_batch)
            mse_by_view = ((recon - x_batch) ** 2).mean(dim=(2, 3))
            scores.append(mse_by_view.cpu().numpy())

    return np.concatenate(scores, axis=0).astype(np.float64)


def compute_latent_vectors(
    model,
    data: np.ndarray,   # [N, V, F, T]
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Extract flattened encoder latent vectors. Returns [N, D].

    This is used for train-normal memory distance scoring. Labels are not used.
    """
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    vectors = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            latent = model.encode(x_batch)
            vectors.append(latent.flatten(start_dim=1).cpu().numpy())

    return np.concatenate(vectors, axis=0).astype(np.float64)


def compute_latent_summary_vectors(
    model,
    data: np.ndarray,   # [N, V, F, T]
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Extract fixed latent spatial summary vectors. Returns [N, 3*C].

    Summary = concat(channel-wise mean, std, max over latent spatial positions).
    This reduces dependence on exact time-frequency position while keeping the
    same train-normal memory scoring protocol.
    """
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    vectors = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            latent = model.encode(x_batch)
            mean = latent.mean(dim=(2, 3))
            std = latent.std(dim=(2, 3), unbiased=False)
            max_vals = latent.amax(dim=(2, 3))
            summary = torch.cat([mean, std, max_vals], dim=1)
            vectors.append(summary.cpu().numpy())

    return np.concatenate(vectors, axis=0).astype(np.float64)


def compute_local_latent_token_lists(
    model,
    data: np.ndarray,   # [N, V, F, T]
    batch_size: int = 32,
    device: str = "cpu",
) -> list:
    """Extract per-file local latent tokens. Returns list[[L_i, C], ...].

    Each latent spatial position becomes one local token. This preserves
    location-specific anomaly evidence that can be washed out by file-level
    mean/std/max summaries.
    """
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    model = model.to(device)
    model.eval()

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    token_lists = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            latent = model.encode(x_batch)  # [B, C, H, W]
            b, c, h, w = latent.shape
            tokens = latent.permute(0, 2, 3, 1).reshape(b, h * w, c)
            token_lists.extend([t.cpu().numpy().astype(np.float32) for t in tokens])

    return token_lists


def compute_local_tail_knn_scores(
    memory_tokens: np.ndarray,
    token_lists: list,
    k: int = 5,
    top_percent: float = 5.0,
    device: str = "cpu",
    memory_chunk_size: int = 32768,
) -> np.ndarray:
    """Exact local-token kNN file score using top-tail token distances.

    Algorithm:
      1. build a train-normal local token memory bank
      2. compute kNN distance for every local token in a test file
      3. file score = mean(top 5% token distances)
    """
    if k < 1:
        raise ValueError("latent_knn_k must be >= 1")
    if not token_lists:
        return np.zeros((0,), dtype=np.float64)
    if len(memory_tokens) == 0:
        raise ValueError("memory token bank is empty")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device_t = requested_device

    memory = torch.from_numpy(memory_tokens.astype(np.float32, copy=False)).to(device_t)
    memory = torch.nn.functional.normalize(memory, dim=1)
    k = min(k, memory.shape[0])
    memory_sq = (memory * memory).sum(dim=1).view(1, -1)

    scores = []
    with torch.no_grad():
        for tokens_np in token_lists:
            q = torch.from_numpy(tokens_np.astype(np.float32, copy=False)).to(device_t)
            q = torch.nn.functional.normalize(q, dim=1)
            q_sq = (q * q).sum(dim=1, keepdim=True)
            best = torch.full((q.shape[0], k), float("inf"), device=device_t)

            for start in range(0, memory.shape[0], memory_chunk_size):
                mem_chunk = memory[start:start + memory_chunk_size]
                d2 = q_sq + memory_sq[:, start:start + mem_chunk.shape[0]] - 2.0 * (q @ mem_chunk.T)
                d2 = torch.clamp(d2, min=0.0)
                merged = torch.cat([best, d2], dim=1)
                best = torch.topk(merged, k=k, dim=1, largest=False).values

            token_scores = torch.sqrt(best).mean(dim=1)
            n_top = max(1, int(np.ceil(token_scores.shape[0] * (top_percent / 100.0))))
            file_score = torch.topk(token_scores, k=n_top, largest=True).values.mean()
            scores.append(float(file_score.cpu()))

    return np.asarray(scores, dtype=np.float64)


def compute_teacher_student_top5_scores(
    teacher,
    student,
    data: np.ndarray,   # [N, V, F, T]
    batch_size: int = 32,
    device: str = "cpu",
    top_percent: float = 5.0,
) -> np.ndarray:
    """Score by local teacher-student discrepancy tail."""
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    device = requested_device
    teacher = teacher.to(device)
    student = student.to(device)
    teacher.eval()
    student.eval()

    dataset = TensorDataset(torch.from_numpy(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    scores = []
    with torch.no_grad():
        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            x_corrupt = _apply_fixed_tf_block_mask(x_batch)
            teacher_latent = teacher.encode(x_batch)
            student_latent = student.encode(x_corrupt)
            teacher_norm = F.normalize(teacher_latent, dim=1)
            student_norm = F.normalize(student_latent, dim=1)
            discrepancy = ((student_norm - teacher_norm) ** 2).sum(dim=1)  # [B, H, W]
            flat = discrepancy.flatten(start_dim=1)
            n_top = max(1, int(np.ceil(flat.shape[1] * (top_percent / 100.0))))
            top_vals = torch.topk(flat, k=n_top, dim=1, largest=True).values
            scores.extend(top_vals.mean(dim=1).cpu().numpy().tolist())

    return np.asarray(scores, dtype=np.float64)


def compute_latent_knn_scores(
    train_latents: np.ndarray,
    test_latents: np.ndarray,
    k: int = 5,
    train_feature_stats=None,
) -> np.ndarray:
    """Score by mean distance to k nearest train-normal latent vectors.

    Larger distance means farther from the normal memory bank and therefore
    more anomalous. k is a global fixed hyperparameter, not selected per machine.
    """
    if k < 1:
        raise ValueError("latent_knn_k must be >= 1")
    k = min(k, len(train_latents))
    if train_feature_stats is not None:
        train_latents = apply_feature_robust_stats(train_latents, train_feature_stats)
        test_latents = apply_feature_robust_stats(test_latents, train_feature_stats)
    train_norm = _l2_normalize(train_latents)
    test_norm = _l2_normalize(test_latents)

    scores = []
    chunk = 256
    for start in range(0, len(test_norm), chunk):
        x = test_norm[start:start + chunk]
        # Squared Euclidean distance on normalized latent vectors.
        d2 = (
            np.sum(x * x, axis=1, keepdims=True)
            + np.sum(train_norm * train_norm, axis=1)[None, :]
            - 2.0 * (x @ train_norm.T)
        )
        d2 = np.maximum(d2, 0.0)
        nearest = np.partition(d2, kth=k - 1, axis=1)[:, :k]
        scores.append(np.sqrt(nearest).mean(axis=1))
    return np.concatenate(scores).astype(np.float64)


def _apply_fixed_tf_block_mask(
    x_batch: torch.Tensor,
    freq_frac: float = 0.25,
    time_frac: float = 0.25,
) -> torch.Tensor:
    """Fixed center TF block mask for deterministic teacher-student scoring."""
    x = x_batch.clone()
    _, _, f, t = x.shape
    block_f = max(1, int(np.ceil(f * freq_frac)))
    block_t = max(1, int(np.ceil(t * time_frac)))
    f0 = max(0, (f - block_f) // 2)
    t0 = max(0, (t - block_t) // 2)
    x[:, :, f0:f0 + block_f, t0:t0 + block_t] = 0.0
    return x


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def fit_feature_robust_stats(train_features: np.ndarray, eps: float = 1e-8) -> dict:
    """Fit train-normal robust feature stats per dimension.

    This is a fixed label-free normalization rule for feature vectors such as
    latent summaries before distance-based scoring.
    """
    q25 = np.percentile(train_features, 25, axis=0)
    q75 = np.percentile(train_features, 75, axis=0)
    return {
        "median": np.median(train_features, axis=0).astype(np.float64),
        "iqr": np.maximum(q75 - q25, eps).astype(np.float64),
        "eps": eps,
    }


def apply_feature_robust_stats(features: np.ndarray, stats: dict) -> np.ndarray:
    """Apply train-normal robust feature scaling."""
    return (
        (features - stats["median"]) / (stats["iqr"] + stats["eps"])
    ).astype(np.float64)


def fit_feature_block_scales(
    train_features: np.ndarray,
    num_blocks: int = 3,
    eps: float = 1e-8,
) -> dict:
    """Fit one robust scale per feature block.

    This is a softer alternative to per-dimension robust normalization. It is
    intended for concatenated latent summaries such as [mean | std | max],
    where blocks can have different overall scales but useful within-block
    magnitude structure should be preserved.
    """
    if train_features.shape[1] % num_blocks != 0:
        raise ValueError("feature dimension must be divisible by num_blocks")
    block_size = train_features.shape[1] // num_blocks
    scales = []
    for i in range(num_blocks):
        block = train_features[:, i * block_size:(i + 1) * block_size].reshape(-1)
        q25 = np.percentile(block, 25)
        q75 = np.percentile(block, 75)
        scales.append(max(q75 - q25, eps))
    return {
        "num_blocks": num_blocks,
        "block_size": block_size,
        "scales": np.asarray(scales, dtype=np.float64),
        "eps": eps,
    }


def apply_feature_block_scales(features: np.ndarray, stats: dict) -> np.ndarray:
    """Apply one train-normal scale per feature block."""
    out = features.astype(np.float64).copy()
    block_size = stats["block_size"]
    for i, scale in enumerate(stats["scales"]):
        start = i * block_size
        end = start + block_size
        out[:, start:end] /= scale
    return out


def fit_feature_whitener(
    train_features: np.ndarray,
    eps: float = 1e-6,
) -> dict:
    """Fit a train-normal covariance whitener.

    Whitening is a fixed label-free latent-geometry transform. It centers the
    train-normal feature cloud, rotates into the covariance eigenbasis, and
    rescales each axis by the inverse square root eigenvalue.
    """
    mean = train_features.mean(axis=0).astype(np.float64)
    centered = train_features.astype(np.float64) - mean
    cov = np.cov(centered, rowvar=False).astype(np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    inv_sqrt = 1.0 / np.sqrt(np.maximum(eigvals, eps))
    return {
        "mean": mean,
        "eigvecs": eigvecs.astype(np.float64),
        "inv_sqrt": inv_sqrt.astype(np.float64),
        "eps": eps,
    }


def apply_feature_whitener(features: np.ndarray, stats: dict) -> np.ndarray:
    """Apply the train-normal covariance whitener."""
    centered = features.astype(np.float64) - stats["mean"]
    rotated = centered @ stats["eigvecs"]
    return (rotated * stats["inv_sqrt"]).astype(np.float64)


def fit_view_residual_stats(train_view_errors: np.ndarray, eps: float = 1e-8) -> dict:
    """Fit train-normal robust residual stats per view.

    The training split is normal-only by protocol, so these statistics are
    label-free and allowed for fixed score normalization.
    """
    q25 = np.percentile(train_view_errors, 25, axis=0)
    q75 = np.percentile(train_view_errors, 75, axis=0)
    return {
        "median": np.median(train_view_errors, axis=0).astype(np.float64),
        "iqr": np.maximum(q75 - q25, eps).astype(np.float64),
        "eps": eps,
    }


def aggregate_view_scores(
    view_errors: np.ndarray,
    mode: str = "mean_mse",
    train_stats=None,
) -> np.ndarray:
    """Aggregate [N, V] view residuals into one fixed anomaly score per file.

    Supported modes:
      - mean_mse: V1 score, mean residual across views.
      - view_z_mean: robust z-normalize each view from train-normal residuals,
        then average views.
      - view_z_top2: robust z-normalize each view from train-normal residuals,
        then average the largest two view scores.
    """
    if mode == "mean_mse":
        return view_errors.mean(axis=1).astype(np.float64)

    if train_stats is None:
        raise ValueError(f"score mode {mode!r} requires train_stats")

    z = (view_errors - train_stats["median"]) / (train_stats["iqr"] + train_stats["eps"])
    if mode == "view_z_mean":
        return z.mean(axis=1).astype(np.float64)
    if mode == "view_z_top2":
        k = min(2, z.shape[1])
        topk = np.partition(z, -k, axis=1)[:, -k:]
        return topk.mean(axis=1).astype(np.float64)

    raise ValueError(f"unknown score mode: {mode}")
