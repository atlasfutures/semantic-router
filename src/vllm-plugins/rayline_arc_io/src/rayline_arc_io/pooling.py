# SPDX-License-Identifier: Apache-2.0

"""Rung A FP32 masked mean and L2 normalization."""

from typing import Any

import torch

from .constants import EMBEDDING_DIMENSION, TOKEN_EMBEDDING_RANK


def fp32_masked_mean_l2(
    hidden_states: Any,
    *,
    expected_tokens: int,
    expected_dimension: int = EMBEDDING_DIMENSION,
) -> list[float]:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("vLLM token embeddings must be a torch.Tensor")
    if hidden_states.ndim != TOKEN_EMBEDDING_RANK:
        raise ValueError(
            "vLLM token embeddings must have shape [serialized_tokens, hidden_size]"
        )
    if tuple(hidden_states.shape) != (expected_tokens, expected_dimension):
        raise ValueError(
            "vLLM token embedding shape mismatch: "
            f"expected {(expected_tokens, expected_dimension)}, "
            f"got {tuple(hidden_states.shape)}"
        )
    if expected_tokens < 1:
        raise ValueError("cannot pool an empty ARC history")

    fp32_states = hidden_states.to(dtype=torch.float32)
    if not bool(torch.isfinite(fp32_states).all().item()):
        raise ValueError("vLLM token embeddings contain non-finite values")
    mean = fp32_states.sum(dim=0, dtype=torch.float32) / expected_tokens
    norm = torch.linalg.vector_norm(mean, ord=2, dtype=torch.float32)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise ValueError("vLLM token embedding mean has zero or non-finite norm")

    embedding = mean / norm
    if not bool(torch.isfinite(embedding).all().item()):
        raise ValueError("normalized ARC embedding contains non-finite values")
    return embedding.tolist()
