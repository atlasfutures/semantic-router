"""Typed configuration for the experimental Rayline ARC selector."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Mirrors maxRaylineARCInflightEncoderCalls in the Go loader.
MAX_INFLIGHT_ENCODER_CALLS = 32

RAYLINE_ARC_ENCODER_MODEL = "Qwen/Qwen3.5-0.8B"
RAYLINE_ARC_ENCODER_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
RAYLINE_ARC_SERIALIZER_VERSION = "mtrouter-token-blocks-v2"

# prefix-cached MEAN is intentionally absent until the Rung C phase gate
# opens; the pinned plugin cannot report it, so accepting it here would only
# defer the failure to readiness.
RaylineARCPoolingCapability = Literal[
    "all_plugin_mean",
    "chunked_causal_mean",
    "resumable_causal_mean",
]


class RaylineARCEncoderReplicaConfig(BaseModel):
    """Stable retained-encoder identity for static membership."""

    model_config = ConfigDict(extra="forbid")

    id: str
    base_url: str
    state: Literal["active", "draining"]


class RaylineARCEncoderFailoverConfig(BaseModel):
    """The exact retained-encoder remap contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    unavailable_status_codes: list[int] = Field(min_length=1, max_length=16)
    unavailable_cooldown_seconds: int = Field(gt=0, le=3600)
    max_remaps: int


class RaylineARCEncoderMembershipConfig(BaseModel):
    """Reviewed dynamic membership source, separate from request semantics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    source: str
    refresh_seconds: int = Field(gt=0, le=300)


class RaylineARCFaultInjectionConfig(BaseModel):
    """Opt-in for the dev-only fault header; one bool, matching the Go loader."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class RaylineARCEncoderConfig(BaseModel):
    """Pinned contract for the dedicated vLLM pooling deployment."""

    model_config = ConfigDict(extra="forbid")

    base_url: str | None = None
    replicas: list[RaylineARCEncoderReplicaConfig] = Field(default_factory=list)
    membership: RaylineARCEncoderMembershipConfig | None = None
    failover: RaylineARCEncoderFailoverConfig | None = None
    model: str
    model_revision: str
    expected_build_id: str
    expected_io_plugin_version: str
    serializer_version: str
    serving_rung: Literal["A", "B"]
    required_pooling_capabilities: list[RaylineARCPoolingCapability] = Field(
        min_length=1,
        max_length=8,
    )
    modal_key_env: str | None = None
    modal_secret_env: str | None = None
    connect_timeout_seconds: int = Field(gt=0)
    total_timeout_seconds: int = Field(gt=0)
    max_retries: int = Field(ge=0, le=3)
    # Readiness re-probe schedule. 0 on either field selects the router's
    # shipped default, 5 s and 60 s.
    probe_retry_initial_seconds: int = Field(default=0, ge=0, le=3600)
    probe_retry_max_seconds: int = Field(default=0, ge=0, le=3600)
    # 0 selects the router's shipped default; the Go loader caps the value.
    max_inflight_encoder_calls: int = Field(
        default=0, ge=0, le=MAX_INFLIGHT_ENCODER_CALLS
    )

    @field_validator("membership", "failover", mode="before")
    @classmethod
    def empty_block_is_absent(cls, value):
        # The canonical reference config includes membership: {} and
        # failover: {} so their field inventory is covered without changing
        # the selected static base_url mode. The Go loader treats a zero block
        # as omitted; do the same here.
        if value in (None, {}):
            return None
        return value


class RaylineARCRedisConfig(BaseModel):
    """Non-secret Redis connection settings for fenced episode state."""

    model_config = ConfigDict(extra="forbid")

    address: str | None = None
    db: int = Field(default=0, ge=0)
    password_env: str | None = None
    use_tls: bool = False
    pool_size: int = Field(default=0, ge=0)


class RaylineARCEpisodeConfig(BaseModel):
    """Serialized episode-state configuration."""

    model_config = ConfigDict(extra="forbid")

    id_header: str
    close_header: str | None = None
    backend: Literal["redis", "memory"]
    key_prefix: str
    acquire_timeout_seconds: int = Field(gt=0)
    lease_ttl_seconds: int = Field(gt=0)
    idle_ttl_seconds: int = Field(gt=0)
    # Only meaningful for backend=memory, and the Go loader accepts it as
    # omitted for Redis; defaulting keeps both validators on the same contract.
    max_in_memory_episodes: int = Field(default=0, ge=0)
    development_mode: bool = False
    redis: RaylineARCRedisConfig | None = None
    # How concurrent turns on one episode are handled; the Go loader holds the
    # same contract. Omitted or "strict": serialized behind the episode lease
    # (eval and training cells). "relaxed": read without the lease, committed
    # only if unchanged; a collision drops that turn's state update instead of
    # failing the request (serving cells).
    consistency: Literal["strict", "relaxed"] | None = None

    @model_validator(mode="after")
    def _relaxed_holds_no_lease_to_close_under(self):
        if self.consistency == "relaxed" and self.close_header:
            raise ValueError("close_header is not served with consistency=relaxed")
        return self


_CHECKPOINT_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# maxRaylineARCConfigStringLength in pkg/config/rayline_arc_config.go. It is
# restated rather than derived because nothing links the two files; when it
# moves there it has to move here, and a test asserts the pair.
_MAX_CONFIG_STRING_BYTES = 512


class RaylineARCRoutesAPIConfig(BaseModel):
    """POST /v1/routes: the selector's choice, returned without executing it.

    Off by default and not implied by configuring the algorithm. A lookup
    drives the encoder with no paying turn behind it, on the same instance
    that serves routed traffic, so an operator opts in deliberately.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Bounds one lookup end to end. Zero selects the shipped 1500 ms default;
    # the Go validator holds the same 100..30000 window, and a value outside
    # it is a misconfiguration rather than a preference.
    deadline_ms: int = Field(default=0, ge=0, le=30_000)
    # Lets a lookup join a conversation's trajectory, with the episode lease
    # and store round trip that implies.
    episode_writes: bool = False
    # Human-readable release name published beside the artifact hash, as
    # "<label>.<hash>". Separate from artifact_revision, which is
    # deployment-private and never reaches a caller.
    checkpoint_label: str = ""

    @field_validator("deadline_ms")
    @classmethod
    def _bounded_deadline(cls, value: int) -> int:
        if value and value < 100:
            raise ValueError("deadline_ms must be at least 100")
        return value

    @field_validator("checkpoint_label")
    @classmethod
    def _label_character_set(cls, value: str) -> str:
        # Bytes, matching the Go validator's len() on the same field. The
        # character set below is ASCII today, so the two agree either way,
        # but reading it as characters would start disagreeing the moment
        # that set widens -- and a config this loader accepts and the router
        # then refuses is a failure the operator only sees at startup.
        if len(value.encode("utf-8")) > _MAX_CONFIG_STRING_BYTES:
            raise ValueError(
                f"checkpoint_label must be at most {_MAX_CONFIG_STRING_BYTES} bytes"
            )
        # fullmatch, not match: Python's `$` also matches just before a
        # trailing newline, so `match` accepts "arc\n" -- which a YAML block
        # scalar produces -- while Go's `$` anchors at end of text and rejects
        # it. The parity this validator claims has to hold on the values YAML
        # actually hands it, not only on the tidy ones.
        if value and not _CHECKPOINT_LABEL.fullmatch(value):
            raise ValueError(
                "checkpoint_label must be lowercase alphanumerics, dashes or underscores"
            )
        return value


class RaylineARCThinkingLevelConfig(BaseModel):
    """One rung of a compiled thinking-lever binding."""

    model_config = ConfigDict(extra="forbid")

    level: str
    rank: int
    suffix: str = ""
    effort: str = ""
    # The registry's digest of this level's bytes; the Go loader recomputes
    # it and refuses a mismatch.
    control_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class RaylineARCThinkingBindingConfig(BaseModel):
    """One worker's lever binding, compiled from the thinking-level registry.

    The shape is checked here; the Go loader checks the rest (placements that
    fit the lever, a declared neutral level, effort spelling) and refuses to
    start on a binding the planner would refuse.
    """

    model_config = ConfigDict(extra="forbid")

    export_sha256: str = ""
    admission: Literal["certified", "experimental"]
    lever: Literal["prompt_steering_suffix", "per_turn_effort"]
    emit: Literal["every_turn", "on_change", "on_change_v1"]
    neutral_level: str = ""
    # The marker an on_change_v1 binding writes on a return to neutral.
    neutral_text: str = ""
    placements: list[
        Literal[
            "append_tail_user_text",
            "insert_user_after_tool_run",
            "system_before_governed_turn",
            "system_after_tool_run",
        ]
    ]
    levels: list[RaylineARCThinkingLevelConfig]


class RaylineARCThinkingLeverConfig(BaseModel):
    """Per-turn thinking lever for one ARC decision. Off by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # Only "rule" is served: every governed turn asks for `level`.
    source: str = ""
    level: str = ""
    admission: Literal["", "certified", "experimental"] = ""
    min_spacing_turns: int = Field(default=0, ge=0)
    # Mirrors thinkinglever.MaxLedgerLength in the Go loader.
    max_ledger_entries: int = Field(default=0, ge=0, le=384)
    eligibility_header: str = ""
    workers: dict[str, RaylineARCThinkingBindingConfig] = Field(default_factory=dict)


class RaylineARCUpstreamAuditConfig(BaseModel):
    """Per-worker upstream extension check. Off by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False


class RaylineARCWorkerThinkingConfig(BaseModel):
    """A thinking worker's base reasoning level, as the registry compiled it."""

    model_config = ConfigDict(extra="forbid")

    level: str
    wire: Literal["effort", "budget", "provider_default"]
    effort: str = ""
    max_tokens: int = Field(default=0, ge=0)


class RaylineARCPolicyBindingConfig(BaseModel):
    """One package action bound to a worker, and what the action dispatches."""

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    worker: str = Field(min_length=1)
    level: str = ""
    # model, effort and reasoning_max_tokens declare the action's dispatch;
    # the Go loader recomputes the action_id from them.
    model: str = ""
    effort: str | None = None
    reasoning_max_tokens: int | None = Field(default=None, gt=0)


class RaylineARCPolicyServiceConfig(BaseModel):
    """The policy-service mode: an external service returns the decision."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    modal_key_env: str = ""
    modal_secret_env: str = ""
    total_timeout_seconds: int = Field(ge=1, le=900)
    connect_timeout_seconds: int = Field(default=0, ge=0)
    # Mirrors maxRaylineARCInflightEncoderCalls in the Go loader.
    max_inflight_calls: int = Field(default=0, ge=0, le=MAX_INFLIGHT_ENCODER_CALLS)
    package_alias: str = Field(min_length=1)
    package_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    bindings: list[RaylineARCPolicyBindingConfig] = Field(min_length=1)
    model_schedule: Literal["", "task_turn_compaction_v1"] = ""
    # Mirrors RaylineARCPolicyServiceConfig.DispatchEffort in the Go loader:
    # provider_default serves actions without their declared effort (v4).
    dispatch_effort: Literal["", "declared", "provider_default"] = ""
    # A v5 package's manifest path (its sha256 is package_sha256); its
    # bindings are only action_id and worker. Empty serves a v4 package.
    package_manifest: str = ""
    # Admits v5 instructions whose registry cell is experimental.
    allow_experimental_controls: bool = False
    # Mirrors TrustTurnSignalHeaders in the Go loader: read the gateway's
    # turn-signal headers (x-rayline-call-kind, x-rayline-compaction,
    # x-rayline-agent-key-source, x-rayline-parent-session,
    # x-rayline-parent-agent) only when a gateway sets or strips every one.
    trust_turn_signal_headers: bool = False
    # Per bound worker, the trained model it serves (v5 only).
    trained_models: dict[str, str] = Field(default_factory=dict)


class RaylineARCAlgorithmConfig(BaseModel):
    """Artifact, encoder, and episode pins for Rayline ARC.

    With policy_service set the decision comes from that service, and the
    artifact and encoder pins are not used.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_dir: str | None = None
    artifact_revision: str | None = None
    encoder: RaylineARCEncoderConfig | None = None
    policy_service: RaylineARCPolicyServiceConfig | None = None
    episode: RaylineARCEpisodeConfig
    include_system_text: bool = False
    drop_mid_conversation_system_text: bool = False
    # Shows the selector the turn's tool NAMES -- never their schemas. A
    # 2026-09-17 encoder probe measured the full schema block moving 40.4% of
    # first-turn decisions, level with the bar that keeps include_system_text
    # off; names move 7.6% for 42 tokens. Off, because safe to send is not the
    # same as shown to help.
    include_tool_names: bool = False
    # Keeps a tool loop on the model family that opened it: a turn answering
    # the previous arm's tool call is scored only among that family's arms,
    # and the hold is lifted (and logged) when none of them is eligible.
    # Reasoning does not carry across families. Off by default.
    hold_family_in_tool_loop: bool = False
    fault_injection: RaylineARCFaultInjectionConfig | None = None
    routes_api: RaylineARCRoutesAPIConfig | None = None
    thinking_lever: RaylineARCThinkingLeverConfig | None = None
    worker_thinking: dict[str, RaylineARCWorkerThinkingConfig] = Field(
        default_factory=dict
    )
    upstream_audit: RaylineARCUpstreamAuditConfig | None = None

    @model_validator(mode="after")
    def _artifact_pins_unless_policy_service(self):
        if self.policy_service is None and (
            self.artifact_dir is None
            or self.artifact_revision is None
            or self.encoder is None
        ):
            raise ValueError(
                "artifact_dir, artifact_revision and encoder are required without policy_service"
            )
        return self

    @model_validator(mode="after")
    def _relaxed_not_with_retained_encoder_sessions(self):
        # A retained encoder session serializes same-episode encodes on the
        # encoder side, so relaxed turns on one episode would wait there.
        if (
            self.episode.consistency == "relaxed"
            and self.encoder is not None
            and "resumable_causal_mean" in self.encoder.required_pooling_capabilities
        ):
            raise ValueError(
                "episode: consistency=relaxed is not served with the "
                "resumable_causal_mean encoder capability"
            )
        return self
