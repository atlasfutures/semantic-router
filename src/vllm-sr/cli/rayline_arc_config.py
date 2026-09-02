"""Typed configuration for the experimental Rayline ARC selector."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    # Bound on the synchronous startup probe. 0 selects the router's shipped
    # 30 s; the ceiling stays inside Cloud Run's 180 s TCP startup budget.
    probe_startup_timeout_seconds: int = Field(default=0, ge=0, le=170)

    @field_validator("membership", mode="before")
    @classmethod
    def empty_membership_is_absent(cls, value):
        # The canonical reference config includes membership: {} so its
        # field inventory is covered without changing the selected static
        # base_url mode. Treat that zero block exactly as an omitted field.
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


class RaylineARCAlgorithmConfig(BaseModel):
    """Artifact, encoder, and episode pins for Rayline ARC."""

    model_config = ConfigDict(extra="forbid")

    artifact_dir: str
    artifact_revision: str
    encoder: RaylineARCEncoderConfig
    episode: RaylineARCEpisodeConfig
