#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed cost receipts for bounded Rayline three-arm packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

H100_USD_PER_SECOND = 0.001097
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
ENCODER_CPU_CORES = 8.0
ENCODER_MEMORY_GIB = 64.0
PRICING_SNAPSHOT = "modal-on-demand-2026-07-31-h100-cpu-memory"


@dataclass(frozen=True)
class BudgetContract:
    """One preregistered packet's complete conservative authority."""

    run_id: str
    previous_conservative_usd: float
    authorized_cumulative_usd: float
    packet_ceiling_usd: float
    required_reserve_usd: float
    maximum_paid_wall_seconds: int
    maximum_orphan_request_seconds: int = 31 * 60
    maximum_scaledown_seconds: int = 5 * 60
    encoder_replicas: int = 1


class BudgetError(RuntimeError):
    """The experiment's conservative envelope exceeds its authority."""


def resource_rate_usd_per_second() -> float:
    return (
        H100_USD_PER_SECOND
        + ENCODER_CPU_CORES * CPU_CORE_USD_PER_SECOND
        + ENCODER_MEMORY_GIB * MEMORY_GIB_USD_PER_SECOND
    )


def budget_receipt(
    contract: BudgetContract, elapsed_seconds: float | None = None
) -> dict[str, Any]:
    resource_seconds = (
        contract.maximum_paid_wall_seconds
        + contract.maximum_orphan_request_seconds
        + contract.maximum_scaledown_seconds
    )
    if contract.encoder_replicas <= 0:
        raise BudgetError("encoder replica count must be positive")
    packet_max = (
        resource_seconds * resource_rate_usd_per_second() * contract.encoder_replicas
    )
    cumulative_max = contract.previous_conservative_usd + packet_max
    reserve = contract.authorized_cumulative_usd - cumulative_max
    if (
        packet_max > contract.packet_ceiling_usd
        or reserve < contract.required_reserve_usd
    ):
        raise BudgetError(
            f"{contract.run_id} resource envelope exceeds budget authority"
        )
    receipt: dict[str, Any] = {
        "previous_conservative_usd": contract.previous_conservative_usd,
        "authorized_cumulative_usd": contract.authorized_cumulative_usd,
        "packet_ceiling_usd": contract.packet_ceiling_usd,
        "required_reserve_usd": contract.required_reserve_usd,
        "maximum_paid_wall_seconds": contract.maximum_paid_wall_seconds,
        "maximum_resource_seconds": resource_seconds,
        "encoder_replicas": contract.encoder_replicas,
        "maximum_resource_envelope_usd": packet_max,
        "cumulative_if_full_envelope_usd": cumulative_max,
        "reserve_after_full_envelope_usd": reserve,
        "provider_spend_usd": 0.0,
        "pricing_snapshot": PRICING_SNAPSHOT,
    }
    if elapsed_seconds is not None:
        observed_upper = min(
            packet_max,
            max(0.0, elapsed_seconds)
            * resource_rate_usd_per_second()
            * contract.encoder_replicas,
        )
        receipt["launcher_window_seconds"] = elapsed_seconds
        receipt["launcher_window_resource_upper_usd"] = observed_upper
    return receipt
