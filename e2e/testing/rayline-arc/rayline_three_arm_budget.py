#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed cost envelope for the bounded PERF015 packet."""

from __future__ import annotations

from typing import Any

PREVIOUS_CONSERVATIVE_USD = 39.31282402
AUTHORIZED_CUMULATIVE_USD = 59.31282402
PACKET_CEILING_USD = 15.0
REQUIRED_RESERVE_USD = 5.0
MAX_PAID_WALL_SECONDS = 90 * 60
MAX_ORPHAN_REQUEST_SECONDS = 31 * 60
MAX_SCALEDOWN_SECONDS = 5 * 60
H100_USD_PER_SECOND = 0.001097
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
ENCODER_CPU_CORES = 8.0
ENCODER_MEMORY_GIB = 64.0
PRICING_SNAPSHOT = "modal-on-demand-2026-07-31-h100-cpu-memory"


class BudgetError(RuntimeError):
    """The experiment's conservative envelope exceeds its authority."""


def resource_rate_usd_per_second() -> float:
    return (
        H100_USD_PER_SECOND
        + ENCODER_CPU_CORES * CPU_CORE_USD_PER_SECOND
        + ENCODER_MEMORY_GIB * MEMORY_GIB_USD_PER_SECOND
    )


def budget_receipt(elapsed_seconds: float | None = None) -> dict[str, Any]:
    resource_seconds = (
        MAX_PAID_WALL_SECONDS + MAX_ORPHAN_REQUEST_SECONDS + MAX_SCALEDOWN_SECONDS
    )
    packet_max = resource_seconds * resource_rate_usd_per_second()
    cumulative_max = PREVIOUS_CONSERVATIVE_USD + packet_max
    reserve = AUTHORIZED_CUMULATIVE_USD - cumulative_max
    if packet_max > PACKET_CEILING_USD or reserve < REQUIRED_RESERVE_USD:
        raise BudgetError("PERF015 resource envelope exceeds budget authority")
    receipt: dict[str, Any] = {
        "previous_conservative_usd": PREVIOUS_CONSERVATIVE_USD,
        "authorized_cumulative_usd": AUTHORIZED_CUMULATIVE_USD,
        "packet_ceiling_usd": PACKET_CEILING_USD,
        "maximum_paid_wall_seconds": MAX_PAID_WALL_SECONDS,
        "maximum_resource_seconds": resource_seconds,
        "maximum_resource_envelope_usd": packet_max,
        "cumulative_if_full_envelope_usd": cumulative_max,
        "reserve_after_full_envelope_usd": reserve,
        "provider_spend_usd": 0.0,
        "pricing_snapshot": PRICING_SNAPSHOT,
    }
    if elapsed_seconds is not None:
        observed_upper = min(
            packet_max,
            max(0.0, elapsed_seconds) * resource_rate_usd_per_second(),
        )
        receipt["launcher_window_seconds"] = elapsed_seconds
        receipt["launcher_window_resource_upper_usd"] = observed_upper
    return receipt
