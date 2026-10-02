"""Cross-field parity tests for the Rayline ARC CLI validator."""

import pathlib
import re
from types import SimpleNamespace

import pytest
from cli.algorithms import AlgorithmConfig, ModelRef
from cli.rayline_arc_config import (
    _CHECKPOINT_LABEL,
    _MAX_CONFIG_STRING_BYTES,
    RaylineARCAlgorithmConfig,
    RaylineARCEncoderFailoverConfig,
    RaylineARCEncoderMembershipConfig,
    RaylineARCEncoderReplicaConfig,
    RaylineARCEpisodeConfig,
    RaylineARCRoutesAPIConfig,
)
from cli.validator_rayline_arc import (
    _effective_auto_model_names,
    _valid_host_port,
    _validate_rayline_arc_auto_aliases,
    _validate_rayline_arc_decision,
    _validate_rayline_arc_replay,
)
from pydantic import ValidationError


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


def test_rayline_arc_accepts_dynamic_redis_membership():
    decision = _valid_dynamic_membership_decision()

    assert _validate_rayline_arc_decision(decision) == []


def test_rayline_arc_dynamic_membership_requires_redis_and_close_header():
    decision = _valid_dynamic_membership_decision()
    decision.algorithm.rayline_arc.episode.backend = "memory"
    decision.algorithm.rayline_arc.episode.development_mode = True
    decision.algorithm.rayline_arc.episode.max_in_memory_episodes = 4
    decision.algorithm.rayline_arc.episode.close_header = None

    errors = _validate_rayline_arc_decision(decision)

    assert any("redis backend" in error.message for error in errors)
    assert any("close_header is required" in error.message for error in errors)


def test_rayline_arc_accepts_static_replica_contract_in_cli_schema():
    decision = _valid_dynamic_membership_decision()
    encoder = decision.algorithm.rayline_arc.encoder
    encoder.membership = None
    encoder.replicas = [
        RaylineARCEncoderReplicaConfig(
            id="encoder-a", base_url="http://encoder-a.test:8000", state="active"
        ),
        RaylineARCEncoderReplicaConfig(
            id="encoder-b", base_url="http://encoder-b.test:8000", state="draining"
        ),
    ]

    assert _validate_rayline_arc_decision(decision) == []


def test_rayline_arc_cli_treats_empty_reference_membership_as_absent():
    raw = _valid_decision().algorithm.rayline_arc.model_dump()
    raw["encoder"]["membership"] = {}

    parsed = RaylineARCAlgorithmConfig.model_validate(raw)

    assert parsed.encoder.membership is None


def test_rayline_arc_cli_treats_empty_close_header_as_absent():
    # The Go loader reads close_header "" as no close header (omitempty), and
    # the reference config spells it that way in artifact mode.
    decision = _valid_decision()
    decision.algorithm.rayline_arc.episode.close_header = ""

    assert _validate_rayline_arc_decision(decision) == []


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


def _valid_dynamic_membership_decision():
    decision = _valid_decision()
    arc = decision.algorithm.rayline_arc
    arc.encoder.base_url = None
    arc.encoder.replicas = []
    arc.encoder.membership = RaylineARCEncoderMembershipConfig(
        schema_version="rayline.arc.encoder-membership.v1",
        source="redis",
        refresh_seconds=5,
    )
    arc.encoder.failover = RaylineARCEncoderFailoverConfig(
        schema_version="rayline.arc.encoder-failover.v1",
        unavailable_status_codes=[503],
        unavailable_cooldown_seconds=30,
        max_remaps=1,
    )
    arc.encoder.serving_rung = "B"
    arc.encoder.required_pooling_capabilities = [
        "chunked_causal_mean",
        "resumable_causal_mean",
    ]
    arc.encoder.max_retries = 0
    arc.episode.close_header = "x-rayline-episode-close"
    return decision


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


# --- routes_api parity with the Go loader -----------------------------------
#
# The CLI model forbids unknown keys, so any bound it does not share with the
# Go validator is a configuration one of the two refuses and the other accepts.
# Reading the Go source rather than restating its numbers is the point: a
# literal copied here would keep passing after the Go side moved.

_GO_CONFIG = (
    pathlib.Path(__file__).resolve().parents[2]
    / "semantic-router"
    / "pkg"
    / "config"
    / "rayline_arc_config.go"
)


def _go_constant(name: str) -> int:
    source = _GO_CONFIG.read_text()
    match = re.search(rf"^\s*{name}\s*=\s*(\d+)\s*$", source, re.MULTILINE)
    assert match, f"{name} not found in {_GO_CONFIG}"
    return int(match.group(1))


def test_routes_api_bounds_match_the_go_loader():
    assert _MAX_CONFIG_STRING_BYTES == _go_constant("maxRaylineARCConfigStringLength")

    minimum = _go_constant("minRaylineARCRoutesDeadlineMS")
    maximum = _go_constant("maxRaylineARCRoutesDeadlineMS")

    RaylineARCRoutesAPIConfig(deadline_ms=minimum)
    RaylineARCRoutesAPIConfig(deadline_ms=maximum)
    for rejected in (minimum - 1, maximum + 1, -1):
        with pytest.raises(ValidationError):
            RaylineARCRoutesAPIConfig(deadline_ms=rejected)

    # Zero is not "below the minimum": it selects the shipped default, which
    # is how an operator asks for the deadline without naming a number.
    assert RaylineARCRoutesAPIConfig(deadline_ms=0).deadline_ms == 0


def test_routes_api_checkpoint_label_matches_the_go_validator():
    pattern = re.search(
        r"raylineARCCheckpointLabelPattern\s*=\s*regexp\.MustCompile\(`([^`]+)`\)",
        _GO_CONFIG.read_text(),
    )
    assert pattern, "checkpoint label pattern not found in the Go loader"
    assert pattern.group(1) == _CHECKPOINT_LABEL.pattern

    RaylineARCRoutesAPIConfig(checkpoint_label="a" * _MAX_CONFIG_STRING_BYTES)
    for rejected in ("a" * (_MAX_CONFIG_STRING_BYTES + 1), "Arc", "-arc", "arc.1"):
        with pytest.raises(ValidationError):
            RaylineARCRoutesAPIConfig(checkpoint_label=rejected)


def test_routes_api_checkpoint_label_rejects_a_trailing_newline():
    # A YAML block scalar produces one, and Python's `$` matches just before a
    # final newline where Go's anchors at end of text. Under `match` the CLI
    # accepted a label the router then refused at startup -- the exact split
    # this mirror exists to prevent.
    for rejected in ("arc\n", "arc\r\n", "arc\n\n"):
        with pytest.raises(ValidationError):
            RaylineARCRoutesAPIConfig(checkpoint_label=rejected)


def test_routes_api_refuses_unknown_keys():
    # extra="forbid" is why a documented-but-unmirrored field is a startup
    # failure rather than a silently ignored one.
    with pytest.raises(ValidationError):
        RaylineARCRoutesAPIConfig(enabled=True, episode_write=True)


def _policy_service_decision(close_header=None):
    arc = RaylineARCAlgorithmConfig.model_validate(
        {
            "policy_service": {
                "base_url": "http://rayline-arc-policy-service:8000",
                "total_timeout_seconds": 60,
                "package_alias": "rayline/arc-public-example",
                "package_sha256": "0" * 64,
                "bindings": [
                    {"action_id": "a" * 64, "worker": "public-arm-a"},
                    {"action_id": "b" * 64, "worker": "public-arm-b"},
                ],
            },
            "episode": {
                "id_header": "x-rayline-episode-id",
                "close_header": close_header,
                "backend": "memory",
                "development_mode": True,
                "key_prefix": "vsr:rayline-arc-policy:",
                "acquire_timeout_seconds": 30,
                "lease_ttl_seconds": 60,
                "idle_ttl_seconds": 900,
                "max_in_memory_episodes": 128,
            },
        }
    )
    decision = _valid_decision()
    decision.algorithm.rayline_arc = arc
    return decision


def test_rayline_arc_cli_accepts_a_policy_service_decision():
    assert _validate_rayline_arc_decision(_policy_service_decision()) == []


def test_rayline_arc_cli_refuses_a_policy_mode_close_header():
    errors = _validate_rayline_arc_decision(
        _policy_service_decision(close_header="x-rayline-episode-close")
    )
    assert any("close_header" in error.field for error in errors), errors


def test_rayline_arc_cli_checks_policy_service_connection_fields():
    for mutate, field in (
        (lambda policy: setattr(policy, "base_url", "not-a-url"), "base_url"),
        (
            lambda policy: setattr(policy, "base_url", "https://user:pw@host/"),
            "base_url",
        ),
        (
            lambda policy: setattr(policy, "modal_key_env", "RAYLINE_ARC_MODAL_KEY"),
            "modal_key_env",
        ),
        (
            lambda policy: setattr(policy, "connect_timeout_seconds", 61),
            "connect_timeout_seconds",
        ),
    ):
        decision = _policy_service_decision()
        mutate(decision.algorithm.rayline_arc.policy_service)
        errors = _validate_rayline_arc_decision(decision)
        assert any(error.field.endswith(field) for error in errors), (field, errors)


def _episode_with(**fields):
    episode = {
        "id_header": "x-rayline-episode-id",
        "backend": "redis",
        "key_prefix": "vsr:rayline-arc:",
        "acquire_timeout_seconds": 30,
        "lease_ttl_seconds": 60,
        "idle_ttl_seconds": 900,
        "redis": {
            "address": "redis:6379",
            "password_env": "RAYLINE_ARC_REDIS_PASSWORD",
        },
    }
    episode.update(fields)
    return episode


def test_episode_consistency_matches_the_go_loader():
    # Omitted, strict and relaxed are the Go loader's values; anything else is refused.
    for value in (None, "strict", "relaxed"):
        parsed = RaylineARCEpisodeConfig.model_validate(
            _episode_with(consistency=value)
        )
        assert parsed.consistency == value
    with pytest.raises(ValidationError):
        RaylineARCEpisodeConfig.model_validate(_episode_with(consistency="eventual"))


def test_relaxed_episode_refuses_a_close_header():
    with pytest.raises(
        ValidationError, match="close_header is not served with consistency=relaxed"
    ):
        RaylineARCEpisodeConfig.model_validate(
            _episode_with(consistency="relaxed", close_header="x-rayline-episode-close")
        )


def test_relaxed_is_served_in_policy_service_mode():
    arc = _policy_service_decision().algorithm.rayline_arc.model_dump()
    arc["episode"]["consistency"] = "relaxed"
    RaylineARCAlgorithmConfig.model_validate(arc)


def test_relaxed_is_not_served_with_retained_encoder_sessions():
    arc = _valid_decision().algorithm.rayline_arc.model_dump()
    arc["encoder"]["serving_rung"] = "B"
    arc["encoder"]["required_pooling_capabilities"] = [
        "chunked_causal_mean",
        "resumable_causal_mean",
    ]
    RaylineARCAlgorithmConfig.model_validate(arc)
    arc["episode"]["consistency"] = "relaxed"
    with pytest.raises(
        ValidationError, match="resumable_causal_mean encoder capability"
    ):
        RaylineARCAlgorithmConfig.model_validate(arc)
