"""Latency benchmark for quorum routing."""

from __future__ import annotations

import time

import torch

from .layer import QuorumMoELayer
from .router import RouterConfig


def benchmark(hidden_dim: int = 1024, tokens: int = 4096, trials: int = 20) -> dict:
    layer = QuorumMoELayer(hidden_dim, RouterConfig())
    x = torch.randn(1, tokens, hidden_dim)
    timings = []
    for _ in range(trials):
        start = time.perf_counter()
        with torch.no_grad():
            layer(x)
        timings.append(time.perf_counter() - start)
    return {"mean_ms": sum(timings) / len(timings) * 1000, "trials": trials}
