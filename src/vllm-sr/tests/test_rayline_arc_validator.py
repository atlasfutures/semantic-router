"""Cross-field parity tests for the Rayline ARC CLI validator."""

from types import SimpleNamespace

from cli.algorithms import AlgorithmConfig, ModelRef
from cli.rayline_arc_config import RaylineARCAlgorithmConfig
from cli.validator_rayline_arc import _validate_rayline_arc_decision


def test_valid_rayline_arc_decision():
    assert _validate_rayline_arc_decision(_valid_decision()) == []


def test_rayline_arc_requires_fail_closed_and_learning_bypass():
    decision = _valid_decision()
    decision.algorithm.on_error = "skip"
    decision.adaptations.mode = "apply"

    errors = _validate_rayline_arc_decision(decision)
    messages = [error.message for error in errors]

    assert any("on_error=fail_closed" in message for message in messages)
    assert any("adaptations.mode=bypass" in message for message in messages)


def test_rayline_arc_rejects_mutable_pins_duplicate_capabilities_and_memory():
    decision = _valid_decision()
    arc = decision.algorithm.rayline_arc
    arc.artifact_revision = "latest"
    arc.encoder.required_pooling_capabilities = [
        "all_plugin_mean",
        "all_plugin_mean",
    ]
    arc.episode.backend = "memory"
    arc.episode.development_mode = False

    errors = _validate_rayline_arc_decision(decision)
    messages = [error.message for error in errors]

    assert any("mutable value" in message for message in messages)
    assert any("cannot contain duplicates" in message for message in messages)
    assert any("development_mode=true" in message for message in messages)


def _valid_decision():
    return SimpleNamespace(
        name="arc-route",
        algorithm=AlgorithmConfig(
            type="rayline_arc",
            on_error="fail_closed",
            rayline_arc=RaylineARCAlgorithmConfig(
                artifact_dir="/var/lib/vllm-sr/rayline-arc",
                artifact_revision="public-synthetic-v1",
                encoder={
                    "base_url": "http://rayline-arc-encoder:8000",
                    "model": "Qwen/Qwen3.5-0.8B",
                    "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
                    "expected_build_id": "vllm@public-synthetic-build",
                    "expected_io_plugin_version": "rayline-arc-io@0.1.0",
                    "serializer_version": "mtrouter-token-blocks-v2",
                    "required_pooling_capabilities": ["all_plugin_mean"],
                    "connect_timeout_seconds": 5,
                    "total_timeout_seconds": 180,
                    "max_retries": 1,
                },
                episode={
                    "id_header": "x-rayline-episode-id",
                    "backend": "redis",
                    "key_prefix": "vsr:rayline-arc:",
                    "acquire_timeout_seconds": 30,
                    "lease_ttl_seconds": 60,
                    "idle_ttl_seconds": 900,
                    "max_in_memory_episodes": 1024,
                    "redis": {
                        "address": "redis:6379",
                        "password_env": "RAYLINE_ARC_REDIS_PASSWORD",
                    },
                },
            ),
        ),
        adaptations=SimpleNamespace(mode="bypass"),
        modelRefs=[
            ModelRef(model="public-arm-a"),
            ModelRef(model="public-arm-b"),
        ],
    )
