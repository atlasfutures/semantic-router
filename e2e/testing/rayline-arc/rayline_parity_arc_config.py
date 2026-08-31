#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Render a generated ARC config from an already verified runtime manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from rayline_development_artifact import MANIFEST_SCHEMA

TOKEN_SCALE = 1_000_000


class ConfigError(ValueError):
    """The runtime and deployment contract cannot form a valid ARC config."""


def _load_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    return value


def _runtime_manifest(runtime_dir: Path) -> dict[str, Any]:
    path = runtime_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read runtime manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ConfigError("runtime manifest schema is unsupported")
    return manifest


def _model(worker: Mapping[str, Any], index: int, endpoint: str) -> dict[str, Any]:
    costs = {
        "prompt_per_1m": float(worker["estimated_input_cost_per_token"]) * TOKEN_SCALE,
        "cached_input_per_1m": float(worker["estimated_cache_read_cost_per_token"])
        * TOKEN_SCALE,
        "cache_write_per_1m": float(worker["estimated_cache_write_cost_per_token"])
        * TOKEN_SCALE,
        "completion_per_1m": float(worker["estimated_output_cost_per_token"])
        * TOKEN_SCALE,
    }
    return {
        "name": str(worker["id"]),
        "provider_model_id": str(worker["model"]),
        "api_format": "openai",
        "pricing": {"currency": "USD", **costs},
        "backend_refs": [
            {
                "name": f"worker-double-{index:03d}",
                "endpoint": endpoint,
                "protocol": "http",
                "type": "openai",
                "base_url": str(worker["provider_base_url"]),
                "provider": "openai",
                "api_key_env": str(worker["api_key_env"]),
            }
        ],
    }


def build_config(
    template: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    artifact_mount_path: str,
    encoder_base_url: str,
    encoder_build_id: str,
    encoder_plugin_version: str,
    worker_endpoint: str,
) -> dict[str, Any]:
    workers = manifest.get("workers")
    encoder = manifest.get("encoder")
    if (
        not isinstance(workers, list)
        or len(workers) < 2
        or not isinstance(encoder, Mapping)
    ):
        raise ConfigError("runtime manifest omits workers/encoder")
    for worker in workers:
        if (
            not isinstance(worker, Mapping)
            or worker.get("dispatch_backend") != "openai_compatible"
            or not str(worker.get("provider_base_url") or "").startswith(
                "http://worker-double:8081/"
            )
        ):
            raise ConfigError("runtime is not staged for the registered worker double")

    config = json.loads(json.dumps(template))
    config["providers"] = {
        "defaults": {"default_model": str(workers[0]["id"])},
        "models": [
            _model(worker, index, worker_endpoint)
            for index, worker in enumerate(workers)
        ],
    }
    decisions = config.get("routing", {}).get("decisions")
    if not isinstance(decisions, list) or len(decisions) != 1:
        raise ConfigError("template must contain exactly one routing decision")
    decision = decisions[0]
    decision["name"] = "rayline-three-arm-parity"
    decision["description"] = "Generated private C82 worker-double parity route"
    decision["modelRefs"] = [
        {
            "model": str(worker["id"]),
            "use_reasoning": worker["thinking_mode"] == "on",
        }
        for worker in workers
    ]
    decision["algorithm"] = {
        "type": "rayline_arc",
        "on_error": "fail_closed",
        "rayline_arc": {
            "artifact_dir": artifact_mount_path,
            "artifact_revision": str(manifest["artifact_id"]),
            "encoder": {
                "base_url": encoder_base_url,
                "model": str(encoder["model"]),
                "model_revision": str(encoder["revision"]),
                "expected_build_id": encoder_build_id,
                "expected_io_plugin_version": encoder_plugin_version,
                "serializer_version": str(encoder["serialization"]),
                "serving_rung": "B",
                "required_pooling_capabilities": [
                    "chunked_causal_mean",
                    "resumable_causal_mean",
                ],
                "modal_key_env": "RAYLINE_ARC_MODAL_KEY",
                "modal_secret_env": "RAYLINE_ARC_MODAL_SECRET",
                "connect_timeout_seconds": 10,
                "total_timeout_seconds": 180,
                "max_retries": 0,
            },
            "episode": {
                "id_header": "x-rayline-episode-id",
                "backend": "redis",
                "key_prefix": "vsr:rayline-three-arm:",
                "acquire_timeout_seconds": 5,
                "lease_ttl_seconds": 180,
                "idle_ttl_seconds": 300,
                "max_in_memory_episodes": 256,
                "redis": {
                    "address": "redis:6379",
                    "db": 0,
                    "password_env": "RAYLINE_ARC_REDIS_PASSWORD",
                    "use_tls": False,
                    "pool_size": 32,
                },
            },
        },
    }
    config["routing"]["modelCards"] = [
        {"name": str(worker["id"]), "modality": "text"} for worker in workers
    ]
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifact-mount-path", required=True)
    parser.add_argument("--encoder-base-url", required=True)
    parser.add_argument("--encoder-build-id", required=True)
    parser.add_argument("--encoder-plugin-version", required=True)
    parser.add_argument("--worker-endpoint", default="worker-double:8081")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        config = build_config(
            _load_mapping(args.template, "template"),
            _runtime_manifest(args.runtime_dir),
            artifact_mount_path=args.artifact_mount_path,
            encoder_base_url=args.encoder_base_url,
            encoder_build_id=args.encoder_build_id,
            encoder_plugin_version=args.encoder_plugin_version,
            worker_endpoint=args.worker_endpoint,
        )
        args.output.write_text(yaml.safe_dump(config, sort_keys=False))
    except (OSError, ConfigError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"cannot render ARC parity config: {error}") from error


if __name__ == "__main__":
    main()
