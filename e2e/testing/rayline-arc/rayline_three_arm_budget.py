#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed cost receipts for bounded Rayline three-arm packets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

H100_USD_PER_SECOND = 0.001097
# Modal on-demand L4. Read from https://modal.com/pricing on 2026-08-11:
# `$0.000222 / s`, which is the `$0.80/hr` the published table quotes. The L4
# is the only GPU class GCP Cloud Run offers that Modal also sells, so it is
# the only silicon on which a Cloud Run capacity claim can be measured.
L4_USD_PER_SECOND = 0.000222
# Modal on-demand RTX PRO 6000. Read from https://modal.com/pricing on
# 2026-08-11: `$0.000842 / s`, the `$3.0312/hr` the published table quotes.
# It is the second of GCP Cloud Run's two GPU classes, so pricing it makes the
# deployment target's remaining silicon measurable on Modal at all.
RTX_PRO_6000_USD_PER_SECOND = 0.000842
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
ENCODER_CPU_CORES = 8.0
ENCODER_MEMORY_GIB = 64.0
DEFAULT_ENCODER_GPU = "H100"
# Per-class GPU seconds. A packet names its class; nothing infers it. Adding a
# class here is not authority to run on it -- the launch gates are separate.
GPU_USD_PER_SECOND = {
    "H100": H100_USD_PER_SECOND,
    "L4": L4_USD_PER_SECOND,
    "RTX-PRO-6000": RTX_PRO_6000_USD_PER_SECOND,
}
# The H100 snapshot keeps its exact recorded name so every closed packet's
# receipt stays byte-identical to what it recorded.
PRICING_SNAPSHOT = "modal-on-demand-2026-07-31-h100-cpu-memory"
PRICING_SNAPSHOTS = {
    "H100": PRICING_SNAPSHOT,
    "L4": "modal-on-demand-2026-08-11-l4-cpu-memory",
    "RTX-PRO-6000": "modal-on-demand-2026-08-11-rtx6000-cpu-memory",
}


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
    # The GPU class the envelope is priced against. It defaults to the H100
    # every recorded packet ran on, so no existing contract changes price; a
    # packet that deploys other silicon must say so here or be billed as an
    # H100, which is the conservative direction.
    encoder_gpu: str = DEFAULT_ENCODER_GPU


class BudgetError(RuntimeError):
    """The experiment's conservative envelope exceeds its authority."""


def resource_rate_usd_per_second(encoder_gpu: str = DEFAULT_ENCODER_GPU) -> float:
    try:
        gpu_rate = GPU_USD_PER_SECOND[encoder_gpu]
    except KeyError:
        raise BudgetError(f"no priced Modal rate for GPU class {encoder_gpu}") from None
    return (
        gpu_rate
        + ENCODER_CPU_CORES * CPU_CORE_USD_PER_SECOND
        + ENCODER_MEMORY_GIB * MEMORY_GIB_USD_PER_SECOND
    )


def minimum_viable_grant_usd(contract: BudgetContract) -> float:
    """How far the authority ceiling must move before this packet may run.

    A prepared packet ships fail-closed by arithmetic: its envelope breaches
    the reserve floor under the standing authority, so `budget_receipt`
    refuses it. Every contract so far restated that shortfall as a hand-typed
    `MINIMUM_VIABLE_GRANT_USD`, which is the figure a human is asked to grant
    -- so it is the one number in the packet that most deserves to be computed
    rather than typed. A granted packet returns exactly `0.0`, because this
    family grants the minimum and nothing more.

    The envelope arithmetic is repeated from `budget_receipt` rather than
    shared, because `budget_receipt` needs the parts as well as the total and
    the two must not drift; a test pins them equal instead.
    """

    if contract.encoder_replicas <= 0:
        raise BudgetError("encoder replica count must be positive")
    resource_seconds = (
        contract.maximum_paid_wall_seconds
        + contract.maximum_orphan_request_seconds
        + contract.maximum_scaledown_seconds
    )
    rate = resource_rate_usd_per_second(contract.encoder_gpu)
    envelope = resource_seconds * rate * contract.encoder_replicas
    shortfall = (
        contract.previous_conservative_usd
        + envelope
        + contract.required_reserve_usd
        - contract.authorized_cumulative_usd
    )
    return max(0.0, shortfall)


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
    rate = resource_rate_usd_per_second(contract.encoder_gpu)
    packet_max = resource_seconds * rate * contract.encoder_replicas
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
        "pricing_snapshot": PRICING_SNAPSHOTS[contract.encoder_gpu],
    }
    if elapsed_seconds is not None:
        observed_upper = min(
            packet_max,
            max(0.0, elapsed_seconds) * rate * contract.encoder_replicas,
        )
        receipt["launcher_window_seconds"] = elapsed_seconds
        receipt["launcher_window_resource_upper_usd"] = observed_upper
    return receipt
