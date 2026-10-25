"""Cross-field parity tests for the Rayline ARC CLI validator."""

import pathlib
import re
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cli.algorithms import AlgorithmConfig, ModelRef
from cli.rayline_arc_config import (
    _CHECKPOINT_LABEL,
    _MAX_CONFIG_STRING_BYTES,
    RaylineARCAlgorithmConfig,
    RaylineARCDecisionsV2Config,
    RaylineARCEncoderFailoverConfig,
    RaylineARCEncoderMembershipConfig,
    RaylineARCEncoderReplicaConfig,
    RaylineARCEpisodeConfig,
    RaylineARCExplorationConfig,
    RaylineARCPolicyFallbackConfig,
    RaylineARCPolicyServiceConfig,
    RaylineARCRoutesAPIConfig,
    RaylineARCRoutingPricesConfig,
)
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


def test_rayline_arc_cli_mirrors_the_cache_context_switch():
    # Off unless a cell opts in, as in the Go loader; a non-bool is refused.
    decision = _policy_service_decision()
    assert decision.algorithm.rayline_arc.policy_service.cache_context is False
    policy = decision.algorithm.rayline_arc.policy_service.model_dump()
    policy["cache_context"] = True
    assert RaylineARCAlgorithmConfig.model_validate(
        {**decision.algorithm.rayline_arc.model_dump(), "policy_service": policy}
    ).policy_service.cache_context
    policy["cache_context"] = "sometimes"
    with pytest.raises(ValidationError):
        RaylineARCAlgorithmConfig.model_validate(
            {**decision.algorithm.rayline_arc.model_dump(), "policy_service": policy}
        )


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


@pytest.mark.parametrize("models", [16, 17])
@pytest.mark.parametrize("enabled", [True, False])
def test_rayline_arc_cli_bounds_fallback_models(models, enabled):
    """Mirrors the Go loader: a fallback-enabled package serves at most 16
    distinct models, as many as an episode can exclude."""
    decision = _policy_service_decision()
    policy = decision.algorithm.rayline_arc.policy_service
    policy.bindings = [
        policy.bindings[0].model_copy(
            update={"action_id": f"{index:064x}", "worker": f"public-arm-{index}"}
        )
        for index in range(models)
    ]
    policy.fallback = RaylineARCPolicyFallbackConfig(enabled=enabled)
    refused = any(
        "distinct models" in error.message
        for error in _validate_rayline_arc_decision(decision)
    )
    assert refused == (enabled and models > 16)


@pytest.mark.parametrize("size", [128, 129])
@pytest.mark.parametrize("enabled", [True, False])
def test_rayline_arc_cli_bounds_fallback_model_names(size, enabled):
    """Mirrors the Go loader: a fallback model name fits an exclusion."""
    decision = _policy_service_decision()
    policy = decision.algorithm.rayline_arc.policy_service
    policy.bindings = [
        policy.bindings[0].model_copy(update={"model": "m" * size}),
        policy.bindings[1],
    ]
    policy.fallback = RaylineARCPolicyFallbackConfig(enabled=enabled)
    refused = any(
        "bytes" in error.message for error in _validate_rayline_arc_decision(decision)
    )
    assert refused == (enabled and size > 128)


@pytest.mark.parametrize("second", ["vendor/a", "vendor/b"])
@pytest.mark.parametrize("enabled", [True, False])
def test_rayline_arc_cli_fallback_needs_one_model_per_worker(second, enabled):
    """Mirrors the Go loader: with the fallback on, a worker serves one model."""
    decision = _policy_service_decision()
    policy = decision.algorithm.rayline_arc.policy_service
    policy.bindings = [
        policy.bindings[0].model_copy(update={"model": "vendor/a"}),
        policy.bindings[0].model_copy(update={"action_id": "c" * 64, "model": second}),
        policy.bindings[1],
    ]
    policy.fallback = RaylineARCPolicyFallbackConfig(enabled=enabled)
    refused = any(
        "one model per worker" in error.message
        for error in _validate_rayline_arc_decision(decision)
    )
    assert refused == (enabled and second != "vendor/a")


def test_rayline_arc_policy_fallback_matches_the_go_contract():
    """fallback mirrors RaylineARCPolicyFallbackConfig: off by default, 0
    seconds meaning the router's default, at most an hour, nothing else."""
    assert RaylineARCPolicyFallbackConfig().enabled is False
    parsed = RaylineARCPolicyFallbackConfig.model_validate(
        {"enabled": True, "cell_exclusion_seconds": 3600}
    )
    assert parsed.enabled and parsed.cell_exclusion_seconds == 3600
    for invalid in (
        {"cell_exclusion_seconds": -1},
        {"cell_exclusion_seconds": 3601},
        {"enabled": True, "fallback_action_id": "x"},
    ):
        with pytest.raises(ValidationError):
            RaylineARCPolicyFallbackConfig.model_validate(invalid)


def test_rayline_arc_accepts_a_context_limit_within_its_bound():
    fields = RaylineARCAlgorithmConfig.model_fields
    assert "max_context_tokens" in fields
    assert fields["max_context_tokens"].default == 0
    base = _valid_decision().algorithm.rayline_arc.model_dump(exclude_none=True)
    for tokens in (0, 262144, 1_000_000, 1 << 24):
        RaylineARCAlgorithmConfig.model_validate(
            {**base, "max_context_tokens": tokens}
        )
    for tokens in (-1, (1 << 24) + 1):
        with pytest.raises(ValidationError):
            RaylineARCAlgorithmConfig.model_validate(
                {**base, "max_context_tokens": tokens}
            )


def test_rayline_arc_accepts_readiness_wait_within_its_bound():
    fields = RaylineARCAlgorithmConfig.model_fields
    assert "readiness_wait_seconds" in fields
    assert fields["readiness_wait_seconds"].default == 0
    base = _valid_decision().algorithm.rayline_arc.model_dump(exclude_none=True)
    for seconds in (-1, 0, 30, 300):
        RaylineARCAlgorithmConfig.model_validate(
            {**base, "readiness_wait_seconds": seconds}
        )
    with pytest.raises(ValidationError):
        RaylineARCAlgorithmConfig.model_validate(
            {**base, "readiness_wait_seconds": 301}
        )


def test_rayline_arc_cli_mirrors_policy_routing_prices():
    assert _policy_service_decision().algorithm.rayline_arc.policy_service.routing_prices is None
    prices = {
        "url": "https://router-proxy.internal/arc/routing-prices",
        "credential_env": "RAYLINE_ARC_ROUTING_PRICES_TOKEN",
        "ttl_seconds": 300,
        "refresh_ahead_seconds": 60,
        "max_age_seconds": 3600,
        "fetch_timeout_ms": 2000,
    }
    assert RaylineARCRoutingPricesConfig.model_validate(prices).ttl_seconds == 300
    for field, value in (
        ("refresh_ahead_seconds", 300),
        ("max_age_seconds", 299),
        ("fetch_timeout_ms", 30001),
        ("surprise", 1),
        ("auth", "oauth"),
        ("audience", "https://feed.example"),
    ):
        with pytest.raises(ValidationError):
            RaylineARCRoutingPricesConfig.model_validate({**prices, field: value})

    id_token = {
        **{k: v for k, v in prices.items() if k != "credential_env"},
        "auth": "google_id_token",
        "audience": "https://routerproxy-abc123-uc.a.run.app",
    }
    assert RaylineARCRoutingPricesConfig.model_validate(id_token).auth == "google_id_token"
    for changed in (
        {"audience": ""},
        {"credential_env": "RAYLINE_ARC_ROUTING_PRICES_TOKEN"},
    ):
        with pytest.raises(ValidationError):
            RaylineARCRoutingPricesConfig.model_validate({**id_token, **changed})


def test_rayline_arc_cli_mirrors_decisions_v2():
    policy = _policy_service_decision().algorithm.rayline_arc.policy_service
    assert policy.decisions_v2 is None
    block = {
        "name": "keen",
        "cost_weight": 1.0,
        "tier_points": [
            {"name": "swift", "cost_weight": 2.0},
            {"name": "sage", "cost_weight": 0},
        ],
        "response_fields": ["price_per_mtok", "thinking"],
    }
    parsed = RaylineARCDecisionsV2Config.model_validate(block)
    assert parsed.tier_points[1].name == "sage"
    assert parsed.name == "keen"
    assert RaylineARCDecisionsV2Config.model_validate({}).name is None
    for name in ("swift", "sage", "tier_2-b", "s" + "x" * 31):
        assert RaylineARCDecisionsV2Config.model_validate({"name": name}).name == name
    assert RaylineARCDecisionsV2Config.model_validate({}).cost_weight is None
    for changed in (
        {"name": "Swift"},
        {"name": "1swift"},
        {"name": "sw ift"},
        {"name": ""},
        {"name": "s" + "x" * 32},
        {"cost_weight": -0.5},
        {"cost_weight": float("nan")},
        {"tier_points": [{"name": "a", "cost_weight": 1}] * 2},
        {"tier_points": [{"name": str(i), "cost_weight": 1} for i in range(9)]},
        {"tier_points": [{"name": "", "cost_weight": 1}]},
        {"tier_points": [{"name": "é" * 33, "cost_weight": 1}]},
        {"tier_points": [{"name": "a"}]},
        {"response_fields": ["predicted_usage"]},
        {"response_fields": ["thinking", "thinking"]},
        {"surprise": 1},
    ):
        with pytest.raises(ValidationError):
            RaylineARCDecisionsV2Config.model_validate({**block, **changed})


def test_rayline_arc_cli_mirrors_exploration():
    policy = _policy_service_decision().algorithm.rayline_arc.policy_service
    assert policy.exploration is None and policy.eval_cell is False
    block = {
        "tau": 0.02,
        "tau_max": 0.05,
        "basis": {
            "r": 0.05,
            "package_sha256": "0" * 64,
            "readout_pin": "qwen3.8-27b-arc-encoder@depth8",
        },
        "salt": "vsr-explore-2026-10",
    }
    assert RaylineARCExplorationConfig.model_validate(block).tau == 0.02
    assert RaylineARCExplorationConfig.model_validate({**block, "tau": 0}).tau == 0
    for changed in (
        {"tau": -0.1},
        {"tau": float("nan")},
        {"tau": 0.06},
        {"tau_max": float("inf")},
        {"basis": {**block["basis"], "r": 1.5}},
        {"basis": {**block["basis"], "package_sha256": "ABC"}},
        {"basis": {**block["basis"], "readout_pin": ""}},
        {"salt": ""},
        {"salt": "s" * 129},
        {"surprise": 1},
    ):
        with pytest.raises(ValidationError):
            RaylineARCExplorationConfig.model_validate({**block, **changed})
    raw = policy.model_dump()
    RaylineARCPolicyServiceConfig.model_validate(
        {**raw, "exploration": block, "eval_cell": True}
    )
    with pytest.raises(ValidationError):
        RaylineARCPolicyServiceConfig.model_validate(
            {
                **raw,
                "exploration": {
                    **block,
                    "basis": {**block["basis"], "package_sha256": "b" * 64},
                },
            }
        )


def test_rayline_arc_cli_checks_routing_prices_urls():
    prices = {
        "url": "https://feed.example/arc/routing-prices",
        "credential_env": "RAYLINE_ARC_ROUTING_PRICES_TOKEN",
        "ttl_seconds": 300,
        "max_age_seconds": 3600,
        "fetch_timeout_ms": 2000,
    }
    decision = _policy_service_decision()
    policy = decision.algorithm.rayline_arc.policy_service
    policy.routing_prices = RaylineARCRoutingPricesConfig.model_validate(prices)
    assert _validate_rayline_arc_decision(decision) == []
    for changed, field in (
        ({"url": "not-a-url"}, "routing_prices.url"),
        ({"url": "https://feed.example/p?x=1"}, "routing_prices.url"),
        ({"credential_env": "1BAD-NAME"}, "routing_prices.credential_env"),
        (
            {
                "credential_env": "",
                "auth": "google_id_token",
                "audience": "feed.example",
            },
            "routing_prices.audience",
        ),
    ):
        policy.routing_prices = RaylineARCRoutingPricesConfig.model_validate(
            {**prices, **changed}
        )
        errors = _validate_rayline_arc_decision(decision)
        assert any(error.field.endswith(field) for error in errors), (field, errors)


def test_rayline_arc_policy_service_replicas_mirror_the_go_loader():
    raw = _policy_service_decision().algorithm.rayline_arc.policy_service.model_dump()
    replicas = [
        {"id": "s1", "base_url": "https://s1.example", "state": "active"},
        {"id": "s2", "base_url": "https://s2.example", "state": "draining"},
    ]
    failover = {
        "schema_version": "rayline.arc.policy-failover.v1",
        "unavailable_status_codes": [503],
        "unavailable_cooldown_seconds": 30,
        "max_remaps": 1,
    }
    pooled = RaylineARCPolicyServiceConfig.model_validate(
        {**raw, "base_url": None, "replicas": replicas, "failover": failover}
    )
    assert [replica.id for replica in pooled.replicas] == ["s1", "s2"]
    decision = _policy_service_decision()
    decision.algorithm.rayline_arc.policy_service = pooled
    assert _validate_rayline_arc_decision(decision) == []
    pooled.replicas[0].base_url = "ftp://s1.example"
    assert _validate_rayline_arc_decision(decision)
    # Each contract the Go loader refuses is refused here too.
    for changed in (
        {"replicas": [replicas[0], {**replicas[1], "id": "s1"}]},
        {"replicas": [{**replicas[0], "state": "draining"}]},
        {"failover": {**failover, "unavailable_status_codes": [409]}},
        {"failover": {**failover, "schema_version": "rayline.arc.encoder-failover.v1"}},
        {"failover": {**failover, "max_remaps": 2}},
    ):
        bad = _policy_service_decision()
        bad.algorithm.rayline_arc.policy_service = (
            RaylineARCPolicyServiceConfig.model_validate(
                {**raw, "base_url": None, "replicas": replicas, "failover": failover, **changed}
            )
        )
        assert _validate_rayline_arc_decision(bad), changed
    # The reference shape: base_url with empty replicas and failover blocks.
    RaylineARCPolicyServiceConfig.model_validate({**raw, "replicas": [], "failover": {}})
    for changed in (
        {"replicas": replicas, "failover": failover},  # both modes
        {"base_url": None, "replicas": replicas, "failover": None},
        {"failover": failover},  # failover without replicas
        {"base_url": None},  # neither mode
    ):
        with pytest.raises(ValidationError):
            RaylineARCPolicyServiceConfig.model_validate({**raw, **changed})


def test_rayline_arc_policy_service_sessions_mirror_the_go_loader():
    raw = _policy_service_decision().algorithm.rayline_arc.policy_service.model_dump()
    keyed = {
        **raw,
        "modal_key_env": "RAYLINE_ARC_MODAL_KEY",
        "modal_secret_env": "RAYLINE_ARC_MODAL_SECRET",
    }
    sessioned = RaylineARCPolicyServiceConfig.model_validate(
        {**keyed, "sessions": {"enabled": True, "start_timeout_seconds": 5}}
    )
    assert sessioned.sessions.enabled
    # The reference shape: an empty sessions block is today's cell.
    plain = RaylineARCPolicyServiceConfig.model_validate({**raw, "sessions": {}})
    assert not plain.sessions.enabled
    replicas = [{"id": "s1", "base_url": "https://s1.example", "state": "active"}]
    failover = {
        "schema_version": "rayline.arc.policy-failover.v1",
        "unavailable_status_codes": [503],
        "unavailable_cooldown_seconds": 30,
        "max_remaps": 1,
    }
    for changed in (
        {"sessions": {"start_timeout_seconds": 5}},  # settings without enabled
        {"sessions": {"enabled": True, "start_timeout_seconds": 300}},
        {"sessions": {"enabled": True, "unknown": 1}},
        {
            "sessions": {"enabled": True},
            "base_url": None,
            "replicas": replicas,
            "failover": failover,
        },
    ):
        with pytest.raises(ValidationError):
            RaylineARCPolicyServiceConfig.model_validate({**keyed, **changed})
    with pytest.raises(ValidationError):
        RaylineARCPolicyServiceConfig.model_validate(
            {**raw, "modal_key_env": "", "modal_secret_env": "", "sessions": {"enabled": True}}
        )


def test_rayline_arc_cli_mirrors_decide_mode():
    policy = _policy_service_decision().algorithm.rayline_arc.policy_service
    raw = policy.model_dump()
    for mode in ("", "sync", "race"):
        RaylineARCPolicyServiceConfig.model_validate(
            {**raw, "decide_mode": mode, "race_ms": 100, "race_max_new_tokens": 0}
        )
    for changed in (
        {"decide_mode": "async"},
        {"race_ms": -1},
        {"race_ms": 5001},
        {"race_max_new_tokens": -1},
    ):
        with pytest.raises(ValidationError):
            RaylineARCPolicyServiceConfig.model_validate({**raw, **changed})
