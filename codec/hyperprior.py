"""Hierarchical hyperprior — a learned entropy model over v0.4 VQ tokens.

Predicts each level's token distribution from the coarser levels, so an arithmetic/ANS
coder spends only the conditional information content instead of the fixed-rate
log2(codebook) per token. This is LOSSLESS on the tokens — the VQ decode (and therefore
FID) is byte-identical; we only change how many bits the tokens cost.

Factorisation (chain rule over the hierarchy):
    bits = H(L1) + H(L2|L1) + H(L3|L1,L2) + H(L4|L1,L2,L3)

L1 is tiny (36 tokens) → a learned marginal prior. Each finer level is predicted by a small
CNN conditioned on the coarser quantised feature maps, upsampled to that level's resolution.

Measured 1-parent floor (scripts/token_entropy.py): 4.1 KB → 1.78 KB. This model conditions
on ALL coarser levels, so it should match or beat that.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LN2 = math.log(2.0)


class _CondHead(nn.Module):
    """Predict categorical logits over `codebook` for every spatial position,
    conditioned on `in_ch` channels of upsampled coarser quantised features."""

    def __init__(self, in_ch, codebook, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(32, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, codebook, 1),
        )

    def forward(self, cond):
        return self.net(cond)


class Hyperprior(nn.Module):
    """Conditional entropy model for the 4-level v0.4 token hierarchy.

    Consumes the quantised feature maps (q_l1..q_l4) and integer index maps produced by a
    frozen VQVAE4L. Returns per-level coded bits (per image) and a trainable loss.
    """

    def __init__(self, config, hidden=256):
        super().__init__()
        self.cb = {
            "l1": config.l1_codebook_size, "l2": config.l2_codebook_size,
            "l3": config.l3_codebook_size, "l4": config.l4_codebook_size,
        }
        ed = {
            "l1": config.l1_embedding_dim, "l2": config.l2_embedding_dim,
            "l3": config.l3_embedding_dim, "l4": config.l4_embedding_dim,
        }
        # L1: position-independent learned marginal prior.
        self.l1_prior = nn.Parameter(torch.zeros(self.cb["l1"]))
        # Finer levels: condition on all coarser quantised maps (upsampled + concat).
        self.h_l2 = _CondHead(ed["l1"], self.cb["l2"], hidden)
        self.h_l3 = _CondHead(ed["l1"] + ed["l2"], self.cb["l3"], hidden)
        self.h_l4 = _CondHead(ed["l1"] + ed["l2"] + ed["l3"], self.cb["l4"], hidden)

    @staticmethod
    def _up(x, size):
        # Nearest, not bilinear: each finer-level position must see its EXACT parent code
        # embedding. Bilinear blends neighbouring parent codes and washes out the parent
        # identity that makes the conditional entropy (and thus the bitrate floor) achievable.
        return F.interpolate(x, size=size, mode="nearest")

    def _logits(self, q, idx):
        """Return per-level logits aligned to each level's index map."""
        B = idx["l1"].shape[0]
        s2, s3, s4 = idx["l2"].shape[-2:], idx["l3"].shape[-2:], idx["l4"].shape[-2:]
        l1 = self.l1_prior.view(1, -1, 1, 1).expand(B, -1, *idx["l1"].shape[-2:])
        l2 = self.h_l2(self._up(q["l1"], s2))
        l3 = self.h_l3(torch.cat([self._up(q["l1"], s3), self._up(q["l2"], s3)], 1))
        l4 = self.h_l4(torch.cat([self._up(q["l1"], s4), self._up(q["l2"], s4),
                                  self._up(q["l3"], s4)], 1))
        return {"l1": l1, "l2": l2, "l3": l3, "l4": l4}

    def forward(self, q, idx):
        """Args: q/idx dicts (l1..l4). Returns (loss_nats_per_token, bits_per_image dict)."""
        logits = self._logits(q, idx)
        B = idx["l1"].shape[0]
        total_ce_nats = 0.0
        n_tokens = 0
        bits_per_image = {}
        for lv in ["l1", "l2", "l3", "l4"]:
            ce_sum = F.cross_entropy(logits[lv], idx[lv], reduction="sum")  # nats
            total_ce_nats = total_ce_nats + ce_sum
            n_tokens += idx[lv].numel()
            bits_per_image[lv] = (ce_sum / B / LN2).item()
        loss = total_ce_nats / n_tokens  # mean nats/token — stable training signal
        bits_per_image["total"] = sum(bits_per_image[l] for l in ["l1", "l2", "l3", "l4"])
        return loss, bits_per_image
