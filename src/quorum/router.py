"""Quorum-based sparse mixture-of-experts router."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RouterConfig:
    num_experts: int = 16
    top_k: int = 2
    quorum: int = 3
    temperature: float = 1.0


class QuorumRouter(nn.Module):
    """Route tokens only when a quorum of experts agrees on the dispatch."""

    def __init__(self, hidden_dim: int, config: RouterConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(hidden_dim, config.num_experts)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(hidden) / self.config.temperature
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, self.config.quorum, dim=-1)
        scores, indices = top.values, top.indices
        agree = (scores[..., -1] > (1.0 / self.config.num_experts)).float()
        weights = scores[..., : self.config.top_k] * agree.unsqueeze(-1)
        return weights, indices[..., : self.config.top_k]

    def dispatch(self, hidden: torch.Tensor, experts: torch.Tensor, expert_fn) -> torch.Tensor:
        weights, indices = self.forward(hidden)
        output = torch.zeros_like(hidden)
        for slot in range(self.config.top_k):
            expert_index = indices[..., slot]
            weight = weights[..., slot].unsqueeze(-1)
            expert_out = expert_fn(experts[expert_index], hidden)
            output = output + weight * expert_out
        return output
