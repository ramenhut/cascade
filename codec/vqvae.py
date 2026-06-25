"""VQ-VAE v0.4 — Adaptive 4-level hierarchical VQ-VAE with level dropout.

Four feature levels at geometric strides 32, 16, 8, 4. Each level adds 4x
spatial detail. During training, random level dropout per 32x32 block teaches
the decoder to handle missing levels. At encode time, per-block level selection
produces content-adaptive file sizes.

Token counts at 192x192:
    L1: 6x6   = 36    tokens (stride 32, always present)
    L2: 12x12 = 144   tokens (stride 16)
    L3: 24x24 = 576   tokens (stride 8)
    L4: 48x48 = 2,304 tokens (stride 4)
    Total:     3,060 tokens (~4.1 KB at full level)

Per 32x32 block:
    L1 only:         1 token
    L1+L2:           5 tokens
    L1+L2+L3:       21 tokens
    L1+L2+L3+L4:   85 tokens

Trained at 192x192 to match existing DFC pipeline encode resolution.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from vq import VectorQuantizer


@dataclass
class VQVAE4LConfig:
    in_channels: int = 3
    hidden_dims: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 512])
    l1_codebook_size: int = 512
    l2_codebook_size: int = 2048
    l3_codebook_size: int = 2048
    l4_codebook_size: int = 2048
    l1_embedding_dim: int = 128
    l2_embedding_dim: int = 128
    l3_embedding_dim: int = 64
    l4_embedding_dim: int = 64
    commitment_cost: float = 0.25
    entropy_weight: float = 0.1
    level_dropout_prob: float = 0.3
    block_size: int = 32  # adaptive decision block size in pixels
    rate_weight: float = 0.0  # RD lambda: rate penalty on VQ codes (0 = v0.4 behaviour)
    rate_temp: float = 1.0    # soft-assignment temperature for the differentiable rate


# =========================================================================
# Building blocks
# =========================================================================

class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.conv(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        return self.res(self.conv(x))


# =========================================================================
# Encoder — 4 taps at strides 32, 16, 8, 4
# =========================================================================

class Encoder4L(nn.Module):
    """Encoder with 4 feature taps at geometric strides.

    Block 0: 3→hidden[0],   stride 2 (H/2)
    Block 1: hidden[0]→[1], stride 2 (H/4)   → L4 tap
    Block 2: hidden[1]→[2], stride 2 (H/8)   → L3 tap
    Block 3: hidden[2]→[3], stride 2 (H/16)  → L2 tap
    Block 4: hidden[3]→[4], stride 2 (H/32)  → L1 tap
    """

    def __init__(self, in_channels=3, hidden_dims=None,
                 l1_dim=128, l2_dim=128, l3_dim=64, l4_dim=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512, 512]

        self.blocks = nn.ModuleList()
        prev_ch = in_channels
        for dim in hidden_dims:
            self.blocks.append(DownBlock(prev_ch, dim))
            prev_ch = dim

        self.proj_l4 = nn.Conv2d(hidden_dims[1], l4_dim, 1)  # H/4
        self.proj_l3 = nn.Conv2d(hidden_dims[2], l3_dim, 1)  # H/8
        self.proj_l2 = nn.Conv2d(hidden_dims[3], l2_dim, 1)  # H/16
        self.proj_l1 = nn.Conv2d(hidden_dims[4], l1_dim, 1)  # H/32

    def forward(self, x):
        features = {}
        h = x
        for i, block in enumerate(self.blocks):
            h = block(h)
            if i == 1:
                features["l4"] = self.proj_l4(h)
            elif i == 2:
                features["l3"] = self.proj_l3(h)
            elif i == 3:
                features["l2"] = self.proj_l2(h)
            elif i == 4:
                features["l1"] = self.proj_l1(h)
        return features


# =========================================================================
# Decoder — fuses L1 + optional L2/L3/L4
# =========================================================================

class Decoder4L(nn.Module):
    """Decoder that reconstructs from L1 + optional L2/L3/L4.

    L1 at H/32 → proj → hidden[0]
    Up 0: hidden[0]→hidden[1] (H/16)  + L2 fused here
    Up 1: hidden[1]→hidden[2] (H/8)   + L3 fused here
    Up 2: hidden[2]→hidden[3] (H/4)   + L4 fused here
    Up 3: hidden[3]→hidden[4] (H/2)
    final_up: hidden[4]→hidden[4] (H)
    final_conv: → 3 channels
    """

    def __init__(self, out_channels=3, hidden_dims=None,
                 l1_dim=128, l2_dim=128, l3_dim=64, l4_dim=64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 512, 256, 128, 64]

        self.proj_l1 = nn.Conv2d(l1_dim, hidden_dims[0], 1)

        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.up_blocks.append(UpBlock(hidden_dims[i], hidden_dims[i + 1]))

        self.final_up = UpBlock(hidden_dims[-1], hidden_dims[-1])
        self.final_conv = nn.Sequential(
            nn.GroupNorm(32, hidden_dims[-1]),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1),
            nn.Tanh(),
        )

        # Fusion layers — concat skip features then project back
        self.fuse_l2 = nn.Conv2d(hidden_dims[1] + l2_dim, hidden_dims[1], 1)  # at H/16
        self.fuse_l3 = nn.Conv2d(hidden_dims[2] + l3_dim, hidden_dims[2], 1)  # at H/8
        self.fuse_l4 = nn.Conv2d(hidden_dims[3] + l4_dim, hidden_dims[3], 1)  # at H/4

    def forward(self, l1, l2, l3, l4):
        """Reconstruct from quantized features. Missing levels should be zeros."""
        h = self.proj_l1(l1)

        for i, up in enumerate(self.up_blocks):
            h = up(h)
            if i == 0:
                h = self._fuse(h, l2, self.fuse_l2)
            elif i == 1:
                h = self._fuse(h, l3, self.fuse_l3)
            elif i == 2:
                h = self._fuse(h, l4, self.fuse_l4)

        h = self.final_up(h)
        return self.final_conv(h)

    def _fuse(self, h, skip, fuse_layer):
        if h.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=h.shape[2:], mode="bilinear",
                                 align_corners=False)
        return fuse_layer(torch.cat([h, skip], dim=1))


# =========================================================================
# Full model with level dropout
# =========================================================================

class VQVAE4L(nn.Module):
    """Adaptive 4-level VQ-VAE.

    L1 (stride 32): global structure, always present
    L2 (stride 16): coarse detail
    L3 (stride 8):  medium detail
    L4 (stride 4):  fine detail

    During training, multi-level auxiliary losses force the decoder to
    reconstruct well at every level combination, not just all-levels.
    """

    def __init__(self, config: VQVAE4LConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder4L(
            in_channels=config.in_channels,
            hidden_dims=config.hidden_dims,
            l1_dim=config.l1_embedding_dim,
            l2_dim=config.l2_embedding_dim,
            l3_dim=config.l3_embedding_dim,
            l4_dim=config.l4_embedding_dim,
        )

        _vq_kw = dict(rate_weight=config.rate_weight, rate_temp=config.rate_temp)
        self.vq_l1 = VectorQuantizer(
            config.l1_codebook_size, config.l1_embedding_dim,
            config.commitment_cost, config.entropy_weight, **_vq_kw)
        self.vq_l2 = VectorQuantizer(
            config.l2_codebook_size, config.l2_embedding_dim,
            config.commitment_cost, config.entropy_weight, **_vq_kw)
        self.vq_l3 = VectorQuantizer(
            config.l3_codebook_size, config.l3_embedding_dim,
            config.commitment_cost, config.entropy_weight, **_vq_kw)
        self.vq_l4 = VectorQuantizer(
            config.l4_codebook_size, config.l4_embedding_dim,
            config.commitment_cost, config.entropy_weight, **_vq_kw)

        decoder_hidden = list(reversed(config.hidden_dims))
        self.decoder = Decoder4L(
            out_channels=config.in_channels,
            hidden_dims=decoder_hidden,
            l1_dim=config.l1_embedding_dim,
            l2_dim=config.l2_embedding_dim,
            l3_dim=config.l3_embedding_dim,
            l4_dim=config.l4_embedding_dim,
        )

    def forward(self, x, level_dropout=False):
        features = self.encoder(x)

        q_l1, loss_l1, idx_l1 = self.vq_l1(features["l1"])
        q_l2, loss_l2, idx_l2 = self.vq_l2(features["l2"])
        q_l3, loss_l3, idx_l3 = self.vq_l3(features["l3"])
        q_l4, loss_l4, idx_l4 = self.vq_l4(features["l4"])

        vq_loss = loss_l1 + loss_l2 + loss_l3 + loss_l4

        recon = self.decoder(q_l1, q_l2, q_l3, q_l4)

        indices = {"l1": idx_l1, "l2": idx_l2, "l3": idx_l3, "l4": idx_l4}
        return recon, vq_loss, indices

    def forward_multilevel(self, x):
        """Forward with auxiliary reconstructions at each level combination.

        One encoder pass, four decoder passes. Each level combination gets
        explicit gradient signal, preventing the model from ignoring L2/L3.

        Returns:
            recons: dict with "all", "l1l2l3", "l1l2", "l1" reconstructions
            vq_loss: scalar
            indices: dict of token indices
        """
        features = self.encoder(x)

        q_l1, loss_l1, idx_l1 = self.vq_l1(features["l1"])
        q_l2, loss_l2, idx_l2 = self.vq_l2(features["l2"])
        q_l3, loss_l3, idx_l3 = self.vq_l3(features["l3"])
        q_l4, loss_l4, idx_l4 = self.vq_l4(features["l4"])

        vq_loss = loss_l1 + loss_l2 + loss_l3 + loss_l4

        z2 = torch.zeros_like(q_l2)
        z3 = torch.zeros_like(q_l3)
        z4 = torch.zeros_like(q_l4)

        recons = {
            "all":    self.decoder(q_l1, q_l2, q_l3, q_l4),
            "l1l2l3": self.decoder(q_l1, q_l2, q_l3, z4),
            "l1l2":   self.decoder(q_l1, q_l2, z3, z4),
            "l1":     self.decoder(q_l1, z2, z3, z4),
        }

        indices = {"l1": idx_l1, "l2": idx_l2, "l3": idx_l3, "l4": idx_l4}
        return recons, vq_loss, indices

    @torch.no_grad()
    def encode(self, x):
        features = self.encoder(x)
        _, _, idx_l1 = self.vq_l1(features["l1"])
        _, _, idx_l2 = self.vq_l2(features["l2"])
        _, _, idx_l3 = self.vq_l3(features["l3"])
        _, _, idx_l4 = self.vq_l4(features["l4"])
        return {"l1": idx_l1, "l2": idx_l2, "l3": idx_l3, "l4": idx_l4}

    @torch.no_grad()
    def decode_from_indices(self, indices, level_mask=None, feather=False):
        """Decode from indices with optional per-block level mask.

        Args:
            indices: dict with l1/l2/l3/l4 index tensors
            level_mask: (B, bh, bw) tensor with values 0-3 indicating max level
                        per block. None = use all levels.
            feather: if True, upsample the per-block masks with bilinear
                     interpolation so level transitions fade smoothly instead of
                     stair-stepping. Nearest-neighbour masks create hard rectangular
                     seams in flat regions that the Gemini decoder preserves as
                     artifacts (see STATUS.md adaptive-FID finding). Bilinear feathers
                     those boundaries away.
        """
        q_l1 = self.vq_l1.lookup(indices["l1"])
        q_l2 = self.vq_l2.lookup(indices["l2"])
        q_l3 = self.vq_l3.lookup(indices["l3"])
        q_l4 = self.vq_l4.lookup(indices["l4"])

        if level_mask is not None:
            # level_mask: (B, bh, bw) → (B, 1, bh, bw)
            lm = level_mask.unsqueeze(1).float()
            mode = "bilinear" if feather else "nearest"

            def make_mask(threshold, target_shape):
                mask = (lm >= threshold).float()
                kw = {"align_corners": False} if feather else {}
                return F.interpolate(mask, size=target_shape[2:], mode=mode, **kw)

            q_l2 = q_l2 * make_mask(1, q_l2.shape)
            q_l3 = q_l3 * make_mask(2, q_l3.shape)
            q_l4 = q_l4 * make_mask(3, q_l4.shape)

        return self.decoder(q_l1, q_l2, q_l3, q_l4)

    @torch.no_grad()
    def decode_at_level(self, indices, max_level=3):
        """Decode using only levels up to max_level (uniform, all blocks).

        max_level: 0=L1 only, 1=L1+L2, 2=L1+L2+L3, 3=all
        """
        q_l1 = self.vq_l1.lookup(indices["l1"])

        def _lookup_or_zeros(vq, idx, threshold):
            if max_level >= threshold:
                return vq.lookup(idx)
            B, H, W = idx.shape
            return torch.zeros(B, vq.embedding_dim, H, W,
                               device=idx.device, dtype=q_l1.dtype)

        q_l2 = _lookup_or_zeros(self.vq_l2, indices["l2"], 1)
        q_l3 = _lookup_or_zeros(self.vq_l3, indices["l3"], 2)
        q_l4 = _lookup_or_zeros(self.vq_l4, indices["l4"], 3)

        return self.decoder(q_l1, q_l2, q_l3, q_l4)
