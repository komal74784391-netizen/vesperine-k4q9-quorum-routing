"""Mixture-of-experts layer using quorum routing."""

from __future__ import annotations

import torch
import torch.nn as nn

from .router import QuorumRouter, RouterConfig


class QuorumMoELayer(nn.Module):
    def __init__(self, hidden_dim: int, config: RouterConfig):
        super().__init__()
        self.router = QuorumRouter(hidden_dim, config)
        self.experts = nn.Parameter(torch.randn(config.num_experts, hidden_dim, hidden_dim) / hidden_dim ** 0.5)
        self.norm = nn.LayerNorm(hidden_dim)

    def _expert_forward(self, weights: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return torch.einsum("tk,tkd->td", weights, hidden)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        out = self.router.dispatch(hidden, self.experts, self._expert_forward)
        return residual + out
