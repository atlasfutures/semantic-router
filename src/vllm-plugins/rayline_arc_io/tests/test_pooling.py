# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from rayline_arc_io.pooling import fp32_masked_mean_l2


def test_fp32_mean_and_l2_normalization() -> None:
    hidden_states = torch.zeros((3, 1024), dtype=torch.bfloat16)
    hidden_states[0, 0] = 1
    hidden_states[1, 1] = 2
    hidden_states[2, 0] = 3

    embedding = fp32_masked_mean_l2(hidden_states, expected_tokens=3)

    expected_norm = math.sqrt(16 + 4)
    assert embedding[0] == pytest.approx(4 / expected_norm, abs=1e-7)
    assert embedding[1] == pytest.approx(2 / expected_norm, abs=1e-7)
    assert math.sqrt(sum(value * value for value in embedding)) == pytest.approx(
        1.0, abs=1e-6
    )


@pytest.mark.parametrize(
    "hidden_states",
    [
        torch.zeros((0, 1024)),
        torch.zeros((2, 100)),
        torch.zeros((2, 1024)),
        torch.full((2, 1024), float("nan")),
    ],
)
def test_pooling_rejects_invalid_or_zero_outputs(hidden_states: torch.Tensor) -> None:
    with pytest.raises((ValueError, TypeError)):
        fp32_masked_mean_l2(hidden_states, expected_tokens=2)


def test_pooling_rejects_non_tensor() -> None:
    with pytest.raises(TypeError):
        fp32_masked_mean_l2([[1.0]], expected_tokens=1)
