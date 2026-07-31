"""Cross-field parity tests for the Rayline ARC CLI validator."""

from types import SimpleNamespace

from cli.algorithms import AlgorithmConfig, ModelRef
from cli.rayline_arc_config import RaylineARCAlgorithmConfig
from cli.validator_rayline_arc import (
    _effective_auto_model_names,
    _valid_host_port,
    _validate_rayline_arc_auto_aliases,
    _validate_rayline_arc_decision,
    _validate_rayline_arc_replay,
)


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


def test_rayline_arc_requires_paired_modal_proxy_environment_names():
    decision = _valid_decision()
    decision.algorithm.rayline_arc.encoder.modal_secret_env = None

    errors = _validate_rayline_arc_decision(decision)

    assert any("must be configured together" in error.message for error in errors)


def test_rayline_arc_accepts_retained_session_capabilities():
    decision = _valid_decision()
    decision.algorithm.rayline_arc.encoder.serving_rung = "B"
    decision.algorithm.rayline_arc.encoder.required_pooling_capabilities = [
        "chunked_causal_mean",
        "resumable_causal_mean",
    ]

    assert _validate_rayline_arc_decision(decision) == []


def test_rayline_arc_rejects_resumable_mean_without_causal_mean():
    decision = _valid_decision()
    decision.algorithm.rayline_arc.encoder.serving_rung = "B"
    decision.algorithm.rayline_arc.encoder.required_pooling_capabilities = [
        "resumable_causal_mean"
    ]

    errors = _validate_rayline_arc_decision(decision)

    assert any(
        "resumable_causal_mean requires chunked_causal_mean" in error.message
        for error in errors
    )


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
                    "serving_rung": "A",
                    "required_pooling_capabilities": ["all_plugin_mean"],
                    "modal_key_env": "RAYLINE_ARC_MODAL_KEY",
                    "modal_secret_env": "RAYLINE_ARC_MODAL_SECRET",
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


def test_redis_address_table_matches_go_validator():
    table = {
        "redis:6379": True,
        "[::1]:6379": True,
        "redis:notaport": False,
        "redis:0": False,
        "redis:70000": False,
        "::1:6379": False,
        ":6379": False,
        "redis": False,
    }
    for address, expected in table.items():
        assert _valid_host_port(address) is expected, address


def _config_with(global_block, model_names=("worker-a", "worker-b"), plugins=None):
    decision = SimpleNamespace(
        name="arc",
        algorithm=SimpleNamespace(type="rayline_arc"),
        modelRefs=[SimpleNamespace(model=name) for name in model_names],
        plugins=plugins or [],
    )
    return SimpleNamespace(decisions=[decision], global_=global_block)


def test_auto_alias_normalization_matches_go():
    table = [
        ({}, {"vllm-sr/auto", "auto", "MoM"}),
        ({"router": {"auto_model_names": ["   "]}}, {"vllm-sr/auto", "auto", "MoM"}),
        ({"router": {"auto_model_name": "  "}}, {"vllm-sr/auto", "auto", "MoM"}),
        ({"router": {"auto_model_name": "Router"}}, {"vllm-sr/auto", "auto", "Router"}),
        ({"router": {"auto_model_names": [" pick ", "pick"]}}, {"pick"}),
    ]
    for global_block, expected in table:
        assert _effective_auto_model_names(_config_with(global_block)) == expected


def test_auto_alias_collision_trims_candidate():
    config = _config_with({}, model_names=(" auto ", "worker-b"))
    errors = _validate_rayline_arc_auto_aliases(config, config.decisions[0])
    assert len(errors) == 1
    assert "collides with an auto-routing alias" in str(errors[0])

    clean = _config_with({}, model_names=("worker-a", "worker-b"))
    assert _validate_rayline_arc_auto_aliases(clean, clean.decisions[0]) == []


def test_router_replay_null_matches_go_loader():
    # Absent key: canonical defaults enable replay, so ARC must be rejected.
    absent = _config_with({})
    assert _validate_rayline_arc_replay(absent, absent.decisions[0])

    # Explicit YAML null zeroes the Go struct (Enabled=false): accepted.
    explicit_null = _config_with({"services": {"router_replay": None}})
    assert _validate_rayline_arc_replay(explicit_null, explicit_null.decisions[0]) == []

    disabled = _config_with({"services": {"router_replay": {"enabled": False}}})
    assert _validate_rayline_arc_replay(disabled, disabled.decisions[0]) == []
