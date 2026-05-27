"""Conv2D Autoencoder for multi-view tensor input."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(use_batch_norm: bool, channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()


def _drop_layer(dropout_p: float) -> nn.Module:
    return nn.Dropout2d(dropout_p) if dropout_p > 0.0 else nn.Identity()


class _ConformerFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4, dropout_p: float = 0.0):
        super().__init__()
        inner = hidden_dim * expansion
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.ff(self.norm(x))


class _ConformerConvModule(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 15, dropout_p: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("conformer conv kernel must be odd")
        self.norm = nn.LayerNorm(hidden_dim)
        self.pointwise_in = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.pointwise_out = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x).transpose(1, 2)
        y = F.glu(self.pointwise_in(y), dim=1)
        y = self.depthwise(y)
        y = self.batch_norm(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y).transpose(1, 2)
        return x + y


class _ConformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        conv_kernel: int = 15,
        dropout_p: float = 0.0,
        ff_expansion: int = 4,
    ):
        super().__init__()
        self.ffn1 = _ConformerFeedForward(hidden_dim, expansion=ff_expansion, dropout_p=dropout_p)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout_p)
        self.conv = _ConformerConvModule(hidden_dim, kernel_size=conv_kernel, dropout_p=dropout_p)
        self.ffn2 = _ConformerFeedForward(hidden_dim, expansion=ff_expansion, dropout_p=dropout_p)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn1(x)
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.attn_dropout(y)
        x = self.conv(x)
        x = self.ffn2(x)
        return self.out_norm(x)


class Full5ViewConformerEncoder(nn.Module):
    """Full5View encoder with unshared per-view stems and late-fused Conformer blocks."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        conv_kernel: int = 15,
        stem_channels: int = 16,
        per_view_channels: int = 32,
        dropout_p: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.stem_channels = stem_channels
        self.per_view_channels = per_view_channels

        self.view_enc1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.GELU(),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_enc2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stem_channels, per_view_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, per_view_channels),
                nn.GELU(),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(in_channels * per_view_channels, hidden_dim, kernel_size=1),
            _norm_layer(use_batch_norm, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ConformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                conv_kernel=conv_kernel,
                dropout_p=dropout_p,
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(self.in_channels):
            xi = x[:, i:i + 1]
            yi = self.view_enc2[i](self.view_enc1[i](xi))
            feats.append(yi)
        fused = self.fusion_proj(torch.cat(feats, dim=1))
        b, c, h, w = fused.shape
        tokens = fused.flatten(2).transpose(1, 2)  # [B, H*W, C]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.out_norm(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TemporalPyramidBlock(nn.Module):
    """Lightweight multi-scale temporal mixing with depthwise temporal convs."""

    def __init__(self, channels: int, use_batch_norm: bool = True, dropout_p: float = 0.0):
        super().__init__()
        branch_specs = (
            (3, 1),
            (5, 2),
            (7, 3),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, kernel_t),
                    padding=(0, dilation * (kernel_t // 2)),
                    dilation=(1, dilation),
                    groups=channels,
                    bias=False,
                ),
                _norm_layer(use_batch_norm, channels),
                nn.ReLU(inplace=True),
            )
            for kernel_t, dilation in branch_specs
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(branch_specs), channels, kernel_size=1, bias=False),
            _norm_layer(use_batch_norm, channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = torch.cat([branch(x) for branch in self.branches], dim=1)
        return x + self.fuse(pyramid)


class TemporalPyramidResidualBranch(nn.Module):
    """Temporal pyramid branch that returns only a correction tensor."""

    def __init__(self, channels: int, use_batch_norm: bool = True, dropout_p: float = 0.0):
        super().__init__()
        branch_specs = (
            (3, 1),
            (5, 2),
            (7, 3),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, kernel_t),
                    padding=(0, dilation * (kernel_t // 2)),
                    dilation=(1, dilation),
                    groups=channels,
                    bias=False,
                ),
                _norm_layer(use_batch_norm, channels),
                nn.ReLU(inplace=True),
            )
            for kernel_t, dilation in branch_specs
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(branch_specs), channels, kernel_size=1, bias=False),
            _norm_layer(use_batch_norm, channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pyramid = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.fuse(pyramid)


class Conv2DAutoencoder(nn.Module):
    """Lightweight Conv2D autoencoder for [B, V, F, T] tensor reconstruction.

    Architecture:
      Encoder: 3x Conv2d(stride=2) -> BN -> ReLU
      Decoder: 3x ConvTranspose2d(stride=2) -> BN -> ReLU -> output
    """

    def __init__(self, in_channels: int, latent_channels: int = 64,
                 use_batch_norm: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.in_channels = in_channels

        # Encoder: V -> 16 -> 32 -> latent_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            _norm_layer(use_batch_norm, 16),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            _norm_layer(use_batch_norm, 32),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=2, padding=1),
            _norm_layer(use_batch_norm, latent_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

        # Decoder: latent_channels -> 32 -> 16 -> in_channels
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1),
            _norm_layer(use_batch_norm, 32),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            _norm_layer(use_batch_norm, 16),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.dec3 = nn.ConvTranspose2d(16, in_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x: [B, V, F, T]
        latent = self.encode(x)
        out = self.dec3(self.dec2(self.dec1(latent)))

        # Crop output to match input size (handles odd dimensions)
        if out.shape[2] != x.shape[2] or out.shape[3] != x.shape[3]:
            out = out[:, :, :x.shape[2], :x.shape[3]]
        return out

    def reconstruct(self, x):
        """Return reconstruction (no grad, for scoring)."""
        with torch.no_grad():
            return self.forward(x)

    def encode(self, x):
        """Return encoder latent tensor."""
        return self.enc3(self.enc2(self.enc1(x)))


class ViewwiseLateFusionAutoencoder(nn.Module):
    """Viewwise encoder with shared per-view stem and late fusion.

    Motivation:
      Multi-view CTAF tensors are not RGB-like channels. This model preserves
      view-specific local structure for two stages before mixing views in a
      fused latent bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 64,
        use_batch_norm: bool = True,
        per_view_channels: int = 16,
        stem_channels: int = 8,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.per_view_channels = per_view_channels

        self.view_enc1 = nn.Sequential(
            nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1),
            _norm_layer(use_batch_norm, stem_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.view_enc2 = nn.Sequential(
            nn.Conv2d(stem_channels, per_view_channels, kernel_size=3, stride=2, padding=1),
            _norm_layer(use_batch_norm, per_view_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.fusion_enc = nn.Sequential(
            nn.Conv2d(
                in_channels * per_view_channels,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, latent_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

        self.fusion_dec = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                in_channels * per_view_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, in_channels * per_view_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.view_dec1 = nn.Sequential(
            nn.ConvTranspose2d(per_view_channels, stem_channels, kernel_size=4, stride=2, padding=1),
            _norm_layer(use_batch_norm, stem_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.view_dec2 = nn.ConvTranspose2d(stem_channels, 1, kernel_size=4, stride=2, padding=1)

    def _encode_views(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        y = x.reshape(b * v, 1, f, t)
        y = self.view_enc2(self.view_enc1(y))
        _, c, f2, t2 = y.shape
        return y.reshape(b, v * c, f2, t2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_enc(self._encode_views(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        latent = self.encode(x)
        y = self.fusion_dec(latent)
        _, vc, f2, t2 = y.shape
        y = y.reshape(b * v, vc // v, f2, t2)
        y = self.view_dec2(self.view_dec1(y))
        out = y.reshape(b, v, y.shape[2], y.shape[3])

        if out.shape[2] != f or out.shape[3] != t:
            out = out[:, :, :f, :t]
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class ViewwiseUnsharedLateFusionAutoencoder(nn.Module):
    """Late-fusion AE with separate encoder/decoder stems per view.

    This relaxes the V13 assumption that one shared local operator should be
    reused for every acoustic view.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 64,
        use_batch_norm: bool = True,
        per_view_channels: int = 16,
        stem_channels: int = 8,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.per_view_channels = per_view_channels

        self.view_enc1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_enc2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stem_channels, per_view_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, per_view_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])

        self.fusion_enc = nn.Sequential(
            nn.Conv2d(
                in_channels * per_view_channels,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, latent_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.fusion_dec = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                in_channels * per_view_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, in_channels * per_view_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

        self.view_dec1 = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(per_view_channels, stem_channels, kernel_size=4, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_dec2 = nn.ModuleList([
            nn.ConvTranspose2d(stem_channels, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(in_channels)
        ])

    def _encode_views(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(self.in_channels):
            xi = x[:, i:i + 1]
            yi = self.view_enc2[i](self.view_enc1[i](xi))
            feats.append(yi)
        return torch.cat(feats, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_enc(self._encode_views(x))

    @torch.no_grad()
    def encode_staged(self, x: torch.Tensor) -> dict:
        """Return intermediate representations at each encoder stage.

        Stages (for transient_3view, in_channels=3):
          - input           : [B, V,  F, T]         raw input tensor
          - stem            : [B, V*8,  F/2, T/2]   after per-view enc1
          - enc2            : [B, V*16, F/4, T/4]   after per-view enc2
          - fused_pre       : [B, V*16, F/4, T/4]   fused (before fusion_enc)
          - latent_map      : [B, 64,   F/8, T/8]   after fusion_enc
          - latent_summary  : [B, 192]               ch-wise mean+std+max
        """
        b, v = x.shape[0], x.shape[1]

        results = {}
        results["input"] = x

        stem_list = []
        for i in range(v):
            stem_list.append(self.view_enc1[i](x[:, i:i + 1]))
        stem_cat = torch.cat(stem_list, dim=1)
        results["stem"] = stem_cat

        enc2_list = []
        for i in range(v):
            enc2_list.append(self.view_enc2[i](stem_list[i]))
        enc2_cat = torch.cat(enc2_list, dim=1)
        results["enc2"] = enc2_cat

        results["fused_pre"] = enc2_cat

        latent = self.fusion_enc(enc2_cat)
        results["latent_map"] = latent

        lmean = latent.mean(dim=(2, 3))
        lstd = latent.std(dim=(2, 3), unbiased=False)
        lamax = latent.amax(dim=(2, 3))
        results["latent_summary"] = torch.cat([lmean, lstd, lamax], dim=1)

        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        latent = self.encode(x)
        fused = self.fusion_dec(latent)
        split = torch.chunk(fused, chunks=v, dim=1)
        outs = []
        for i, yi in enumerate(split):
            oi = self.view_dec2[i](self.view_dec1[i](yi))
            outs.append(oi)
        out = torch.cat(outs, dim=1)
        if out.shape[2] != f or out.shape[3] != t:
            out = out[:, :, :f, :t]
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class Full5ViewConformerAutoencoder(nn.Module):
    """Strong full_5view AE with Conformer-style fused encoder and conv decoder."""

    def __init__(
        self,
        in_channels: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        conv_kernel: int = 15,
        stem_channels: int = 16,
        per_view_channels: int = 32,
        dropout_p: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.per_view_channels = per_view_channels
        self.encoder = Full5ViewConformerEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_kernel=conv_kernel,
            stem_channels=stem_channels,
            per_view_channels=per_view_channels,
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
        )
        self.fusion_dec = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels * per_view_channels, kernel_size=1),
            _norm_layer(use_batch_norm, in_channels * per_view_channels),
            nn.GELU(),
            _drop_layer(dropout_p),
        )
        self.view_dec1 = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(per_view_channels, stem_channels, kernel_size=4, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.GELU(),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_dec2 = nn.ModuleList([
            nn.ConvTranspose2d(stem_channels, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(in_channels)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_staged(self, x: torch.Tensor) -> dict:
        latent = self.encode(x)
        mean = latent.mean(dim=(2, 3))
        std = latent.std(dim=(2, 3), unbiased=False)
        max_vals = latent.amax(dim=(2, 3))
        return {
            "input": x,
            "latent_map": latent,
            "latent_summary": torch.cat([mean, std, max_vals], dim=1),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        latent = self.encode(x)
        fused = self.fusion_dec(latent)
        split = torch.chunk(fused, chunks=v, dim=1)
        outs = []
        for i, yi in enumerate(split):
            oi = self.view_dec2[i](self.view_dec1[i](yi))
            outs.append(oi)
        out = torch.cat(outs, dim=1)
        if out.shape[2] != f or out.shape[3] != t:
            out = out[:, :, :f, :t]
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class ViewwiseUnsharedTemporalPyramidAutoencoder(nn.Module):
    """Unshared late-fusion AE with per-view temporal pyramid blocks.

    Each view keeps its own local stem, then runs a fixed three-branch temporal
    pyramid before global late fusion.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 64,
        use_batch_norm: bool = True,
        per_view_channels: int = 16,
        stem_channels: int = 8,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.per_view_channels = per_view_channels

        self.view_enc1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_enc2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stem_channels, per_view_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, per_view_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_pyramid = nn.ModuleList([
            TemporalPyramidBlock(
                per_view_channels,
                use_batch_norm=use_batch_norm,
                dropout_p=dropout_p,
            )
            for _ in range(in_channels)
        ])

        self.fusion_enc = nn.Sequential(
            nn.Conv2d(
                in_channels * per_view_channels,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, latent_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.fusion_dec = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                in_channels * per_view_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, in_channels * per_view_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

        self.view_dec1 = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(per_view_channels, stem_channels, kernel_size=4, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_dec2 = nn.ModuleList([
            nn.ConvTranspose2d(stem_channels, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(in_channels)
        ])

    def _encode_views(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(self.in_channels):
            xi = x[:, i:i + 1]
            yi = self.view_enc2[i](self.view_enc1[i](xi))
            yi = self.view_pyramid[i](yi)
            feats.append(yi)
        return torch.cat(feats, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_enc(self._encode_views(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        latent = self.encode(x)
        fused = self.fusion_dec(latent)
        split = torch.chunk(fused, chunks=v, dim=1)
        outs = []
        for i, yi in enumerate(split):
            oi = self.view_dec2[i](self.view_dec1[i](yi))
            outs.append(oi)
        out = torch.cat(outs, dim=1)
        if out.shape[2] != f or out.shape[3] != t:
            out = out[:, :, :f, :t]
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class ViewwiseUnsharedResidualTemporalPyramidAutoencoder(nn.Module):
    """V16 main path plus residual temporal-pyramid correction per view."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 64,
        use_batch_norm: bool = True,
        per_view_channels: int = 16,
        stem_channels: int = 8,
        dropout_p: float = 0.0,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.per_view_channels = per_view_channels

        self.view_enc1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_enc2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stem_channels, per_view_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, per_view_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_residual = nn.ModuleList([
            TemporalPyramidResidualBranch(
                per_view_channels,
                use_batch_norm=use_batch_norm,
                dropout_p=dropout_p,
            )
            for _ in range(in_channels)
        ])
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.fusion_enc = nn.Sequential(
            nn.Conv2d(
                in_channels * per_view_channels,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, latent_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.fusion_dec = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                in_channels * per_view_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, in_channels * per_view_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

        self.view_dec1 = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(per_view_channels, stem_channels, kernel_size=4, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_dec2 = nn.ModuleList([
            nn.ConvTranspose2d(stem_channels, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(in_channels)
        ])

    def _encode_views(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(self.in_channels):
            xi = x[:, i:i + 1]
            h = self.view_enc2[i](self.view_enc1[i](xi))
            p = self.view_residual[i](h)
            feats.append(h + self.alpha * p)
        return torch.cat(feats, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_enc(self._encode_views(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        latent = self.encode(x)
        fused = self.fusion_dec(latent)
        split = torch.chunk(fused, chunks=v, dim=1)
        outs = []
        for i, yi in enumerate(split):
            oi = self.view_dec2[i](self.view_dec1[i](yi))
            outs.append(oi)
        out = torch.cat(outs, dim=1)
        if out.shape[2] != f or out.shape[3] != t:
            out = out[:, :, :f, :t]
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)


class ViewwiseUnsharedGatedLateFusionAutoencoder(nn.Module):
    """Unshared late-fusion AE with learned sample-wise view gates.

    This keeps separate stems per view, then learns a fixed global gating
    mechanism from train-normal reconstruction alone to adapt view importance
    per sample before fusion.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 64,
        use_batch_norm: bool = True,
        per_view_channels: int = 16,
        stem_channels: int = 8,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.per_view_channels = per_view_channels

        self.view_enc1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, stem_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_enc2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(stem_channels, per_view_channels, kernel_size=3, stride=2, padding=1),
                _norm_layer(use_batch_norm, per_view_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])

        gate_hidden = max(4, per_view_channels // 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(per_view_channels, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )

        self.fusion_enc = nn.Sequential(
            nn.Conv2d(
                in_channels * per_view_channels,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, latent_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )
        self.fusion_dec = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels,
                in_channels * per_view_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            _norm_layer(use_batch_norm, in_channels * per_view_channels),
            nn.ReLU(inplace=True),
            _drop_layer(dropout_p),
        )

        self.view_dec1 = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(per_view_channels, stem_channels, kernel_size=4, stride=2, padding=1),
                _norm_layer(use_batch_norm, stem_channels),
                nn.ReLU(inplace=True),
                _drop_layer(dropout_p),
            )
            for _ in range(in_channels)
        ])
        self.view_dec2 = nn.ModuleList([
            nn.ConvTranspose2d(stem_channels, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(in_channels)
        ])
        self.last_view_gates = None

    def _encode_views(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        logits = []
        for i in range(self.in_channels):
            xi = x[:, i:i + 1]
            yi = self.view_enc2[i](self.view_enc1[i](xi))
            feats.append(yi)
            pooled = yi.mean(dim=(2, 3))
            logits.append(self.gate_mlp(pooled))

        gate_logits = torch.cat(logits, dim=1)  # [B, V]
        gates = torch.sigmoid(gate_logits)
        gates = gates / (gates.mean(dim=1, keepdim=True) + 1e-6)
        self.last_view_gates = gates.detach()

        gated_feats = [
            yi * gates[:, i].view(-1, 1, 1, 1)
            for i, yi in enumerate(feats)
        ]
        return torch.cat(gated_feats, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion_enc(self._encode_views(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, v, f, t = x.shape
        latent = self.encode(x)
        fused = self.fusion_dec(latent)
        split = torch.chunk(fused, chunks=v, dim=1)
        outs = []
        for i, yi in enumerate(split):
            oi = self.view_dec2[i](self.view_dec1[i](yi))
            outs.append(oi)
        out = torch.cat(outs, dim=1)
        if out.shape[2] != f or out.shape[3] != t:
            out = out[:, :, :f, :t]
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)
