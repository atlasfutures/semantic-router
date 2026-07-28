"""Typed configuration for the experimental Rayline ARC selector."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RAYLINE_ARC_ENCODER_MODEL = "Qwen/Qwen3.5-0.8B"
RAYLINE_ARC_ENCODER_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
RAYLINE_ARC_SERIALIZER_VERSION = "mtrouter-token-blocks-v2"

# prefix-cached MEAN is intentionally absent until the Rung C phase gate
# opens; the pinned plugin cannot report it, so accepting it here would only
# defer the failure to readiness.
RaylineARCPoolingCapability = Literal[
    "all_plugin_mean",
    "chunked_causal_mean",
]


class RaylineARCEncoderConfig(BaseModel):
    """Pinned contract for the dedicated vLLM pooling deployment."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
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
    backend: Literal["redis", "memory"]
    key_prefix: str
    acquire_timeout_seconds: int = Field(gt=0)
    lease_ttl_seconds: int = Field(gt=0)
    idle_ttl_seconds: int = Field(gt=0)
    max_in_memory_episodes: int = Field(ge=0)
    development_mode: bool = False
    redis: RaylineARCRedisConfig | None = None


class RaylineARCAlgorithmConfig(BaseModel):
    """Artifact, encoder, and episode pins for Rayline ARC."""

    model_config = ConfigDict(extra="forbid")

    artifact_dir: str
    artifact_revision: str
    encoder: RaylineARCEncoderConfig
    episode: RaylineARCEpisodeConfig
