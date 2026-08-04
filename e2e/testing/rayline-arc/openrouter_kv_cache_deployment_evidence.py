#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Persist deployment identity and key usage for remote KV packets."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from openrouter_encoder_runtime import packet_encoder, plugin_source_digest
from openrouter_fullstack_state import RunPacket, RuntimeState

KV_MODES = frozenset({"kv-cache", "kv-cache-flashinfer", "kv-cache-flashinfer-agt018"})
FLASHINFER_KV_MODES = frozenset({"kv-cache-flashinfer", "kv-cache-flashinfer-agt018"})


def _source_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def persist(
    *,
    repo_root: Path,
    mode: str,
    run_id: str,
    usage: float,
    packet: RunPacket,
    state: RuntimeState,
) -> None:
    if mode not in KV_MODES:
        return
    encoder = packet_encoder(packet)
    output_dir = repo_root / ".agent-harness/rayline-kv-cache" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    source_commit = _source_commit(repo_root)
    (output_dir / "remote-key-usage.json").write_text(
        json.dumps({"usage_usd": usage}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if state.provider_preflight is not None:
        (output_dir / "remote-provider-preflight.json").write_text(
            json.dumps(state.provider_preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if not state.encoder_app_id:
        return
    deployment = {
        "architecture": "semantic-router with retained remote vLLM encoder",
        "encoder_app_id": state.encoder_app_id,
        "encoder_app_name": encoder.app_name,
        "encoder_gpu": "H100",
        "encoder_build_id": encoder.build_id,
        "encoder_gdn_prefill_backend": (
            "flashinfer" if mode in FLASHINFER_KV_MODES else "torch_reference"
        ),
        "encoder_deployment_source_commit": (
            source_commit if encoder.ephemeral else encoder.deployment_source_commit
        ),
        "encoder_plugin_source_digest": (
            plugin_source_digest(repo_root)
            if encoder.ephemeral
            else encoder.plugin_source_digest
        ),
        "encoder_ephemeral": encoder.ephemeral,
        "semantic_router_commit": source_commit,
    }
    if packet.artifact_revision:
        deployment["artifact_revision"] = packet.artifact_revision
    (output_dir / "remote-deployment.json").write_text(
        json.dumps(deployment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
