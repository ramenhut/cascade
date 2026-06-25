"""Vector Quantization layer with EMA codebook updates and entropy regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector quantization with straight-through estimator.

    Uses EMA codebook updates during training for stability.
    Includes optional entropy regularization to encourage compressible distributions.
    Dead codes (unused embeddings) are periodically reset to perturbed versions of
    active codes to combat codebook collapse.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        entropy_weight: float = 0.0,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        rate_weight: float = 0.0,
        rate_temp: float = 1.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.entropy_weight = entropy_weight
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        # Rate-distortion: penalise expected codelength so the encoder concentrates usage
        # on fewer/cheaper codes (lower bitrate) in exchange for distortion. rate_weight is
        # the RD lambda; 0 = off (v0.4 behaviour unchanged). rate_temp is the soft-assignment
        # temperature used to make the (otherwise argmin) code choice differentiable.
        self.rate_weight = rate_weight
        self.rate_temp = rate_temp
        self.last_rate_bits = 0.0  # measured expected bits/token, for logging

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embedding_sum", self.embedding.weight.clone())
        self.register_buffer("_update_counter", torch.tensor(0, dtype=torch.long))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input features.

        Args:
            z: (B, C, H, W) feature map where C == embedding_dim

        Returns:
            quantized: (B, C, H, W) quantized features (straight-through)
            loss: scalar commitment + entropy loss
            indices: (B, H, W) codebook indices
        """
        B, C, H, W = z.shape
        assert C == self.embedding_dim

        # (B, C, H, W) -> (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        # Compute distances to codebook entries
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )

        indices = d.argmin(dim=1)
        quantized_flat = self.embedding(indices)

        if self.training:
            self._ema_update(z_flat, indices)

        # Commitment loss — only the commitment term (encoder → codebook)
        # The codebook is updated via EMA, not gradients, so we don't need
        # the codebook → encoder term (which would fight the EMA update)
        commitment_loss = self.commitment_cost * F.mse_loss(z_flat, quantized_flat.detach())

        # Entropy regularization: MAXIMIZE entropy to encourage uniform codebook usage
        # and prevent codebook collapse. Higher entropy = more codes used = better
        # codebook utilization. (Compressibility comes from the L3 residual design,
        # not from forcing low entropy here.)
        entropy_loss = torch.tensor(0.0, device=z.device)
        if self.entropy_weight > 0:
            counts = torch.zeros(self.num_embeddings, device=z.device)
            counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=counts.dtype))
            avg_probs = counts / counts.sum()
            eps = 1e-10
            entropy = -(avg_probs * (avg_probs + eps).log()).sum()
            max_entropy = torch.tensor(self.num_embeddings, device=z.device).float().log()
            # Negative sign: minimizing loss = maximizing entropy
            entropy_loss = -self.entropy_weight * (entropy / max_entropy)

        loss = commitment_loss + entropy_loss

        # Rate term (differentiable): expected codelength under the soft assignment
        # q(j|z) = softmax(-d_j / temp), priced by the running marginal code probability.
        # Minimising it pushes the encoder toward frequently-used (cheap) codes, lowering
        # the bitrate; the trade-off against reconstruction is the rate-distortion curve.
        if self.rate_weight > 0:
            with torch.no_grad():
                p = self.ema_cluster_size + 1.0
                p = p / p.sum()
                code_bits = -(p + 1e-10).log2()           # bits to code each entry
            q = F.softmax(-d / self.rate_temp, dim=1)      # (N, num_embeddings), differentiable in z
            rate = (q * code_bits.unsqueeze(0)).sum(dim=1).mean()  # expected bits/token
            self.last_rate_bits = rate.item()
            loss = loss + self.rate_weight * rate

        # Straight-through estimator
        quantized_flat = z_flat + (quantized_flat - z_flat).detach()

        quantized = quantized_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        indices = indices.reshape(B, H, W)

        return quantized, loss, indices

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up codebook entries by index.

        Args:
            indices: (B, H, W) codebook indices

        Returns:
            quantized: (B, C, H, W) feature map
        """
        B, H, W = indices.shape
        flat = indices.reshape(-1)
        quantized = self.embedding(flat)
        return quantized.reshape(B, H, W, self.embedding_dim).permute(0, 3, 1, 2)

    def _ema_update(self, z_flat: torch.Tensor, indices: torch.Tensor) -> None:
        # Use scatter_add instead of one_hot matmul to save memory
        cluster_size = torch.zeros(self.num_embeddings, device=z_flat.device)
        cluster_size.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
        embedding_sum = torch.zeros_like(self.ema_embedding_sum)
        embedding_sum.scatter_add_(0, indices.unsqueeze(1).expand_as(z_flat), z_flat)

        self.ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self.ema_embedding_sum.mul_(self.ema_decay).add_(embedding_sum, alpha=1 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size_smoothed = (
            (self.ema_cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
        )

        self.embedding.weight.data.copy_(
            self.ema_embedding_sum / cluster_size_smoothed.unsqueeze(1)
        )

        # Periodically reset dead codes (every 100 updates)
        self._update_counter += 1
        if self._update_counter % 100 == 0:
            self._reset_dead_codes(z_flat)

    def _reset_dead_codes(self, z_flat: torch.Tensor) -> None:
        """Replace dead codebook entries with perturbed versions of active entries."""
        dead_mask = self.ema_cluster_size < self.dead_code_threshold
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return

        # Sample active codes from the current batch
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        if len(alive_indices) == 0:
            return

        # Replace dead codes with random batch vectors + small noise
        n_batch = z_flat.shape[0]
        replace_indices = torch.randint(0, n_batch, (num_dead,), device=z_flat.device)
        new_codes = z_flat[replace_indices].detach()
        noise = torch.randn_like(new_codes) * 0.01
        new_codes = new_codes + noise

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        self.embedding.weight.data[dead_indices] = new_codes
        self.ema_embedding_sum[dead_indices] = new_codes
        self.ema_cluster_size[dead_indices] = self.dead_code_threshold
