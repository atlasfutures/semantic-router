# SPDX-License-Identifier: Apache-2.0

"""Strict wire schemas for the Rayline ARC pooling endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .constants import (
    EMBEDDING_DIMENSION,
    MAX_REQUEST_BYTES,
    MAX_TURNS,
)

EpisodeIDHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ArcTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["user", "assistant"]
    text: str


class ArcPoolingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    # Required, not defaulted: an unversioned client must be rejected rather
    # than silently treated as the frozen v1 schema.
    schema_version: Literal["rayline.arc.pooling-request.v1"]
    serializer_version: Literal["mtrouter-token-blocks-v2"]
    serving_rung: Literal["A", "B"]
    episode_id_hash: EpisodeIDHash
    turns: Annotated[list[ArcTurn], Field(min_length=1, max_length=MAX_TURNS)]

    @model_validator(mode="after")
    def reject_oversized_request(self) -> "ArcPoolingRequest":
        encoded_bytes = sum(
            len(turn.role.encode("utf-8")) + len(turn.text.encode("utf-8"))
            for turn in self.turns
        )
        if encoded_bytes > MAX_REQUEST_BYTES:
            raise ValueError(
                f"turn payload exceeds the {MAX_REQUEST_BYTES}-byte ARC request limit"
            )
        return self


class ArcPoolingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    embedding: Annotated[
        list[float],
        Field(
            min_length=EMBEDDING_DIMENSION,
            max_length=EMBEDDING_DIMENSION,
        ),
    ]
    serialized_tokens: Annotated[int, Field(gt=0)]
    full_history_tokens: Annotated[int, Field(gt=0)]
    truncated_tokens: Annotated[int, Field(ge=0)]
    cached_prefix_tokens: Annotated[int, Field(ge=0)]
    serializer_version: str
    model: str
    model_revision: str
    tokenizer_revision: str
    tokenizer_sha256: str
    eos_token_id: Annotated[int, Field(ge=0)]
    engine_build_id: str
    io_plugin_version: str
    pooling_capabilities: list[str]


class ArcSessionPoolingRequest(BaseModel):
    """Full-history request for the bounded retained-state endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["rayline.arc.session-pooling-request.v1"]
    serializer_version: Literal["mtrouter-token-blocks-v2"]
    serving_rung: Literal["B"]
    episode_id_hash: EpisodeIDHash
    turns: Annotated[list[ArcTurn], Field(min_length=1, max_length=MAX_TURNS)]

    @model_validator(mode="after")
    def reject_oversized_request(self) -> "ArcSessionPoolingRequest":
        encoded_bytes = sum(
            len(turn.role.encode("utf-8")) + len(turn.text.encode("utf-8"))
            for turn in self.turns
        )
        if encoded_bytes > MAX_REQUEST_BYTES:
            raise ValueError(
                f"turn payload exceeds the {MAX_REQUEST_BYTES}-byte ARC request limit"
            )
        return self


class ArcSessionPoolingResponse(BaseModel):
    """Direct response from the versioned retained-state HTTP endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["rayline.arc.session-pooling-response.v1"]
    embedding: Annotated[
        list[float],
        Field(
            min_length=EMBEDDING_DIMENSION,
            max_length=EMBEDDING_DIMENSION,
        ),
    ]
    serialized_tokens: Annotated[int, Field(gt=0)]
    full_history_tokens: Annotated[int, Field(gt=0)]
    truncated_tokens: Annotated[int, Field(ge=0)]
    retained_prefix_tokens: Annotated[int, Field(ge=0)]
    appended_tokens: Annotated[int, Field(ge=0)]
    session_action: Literal["created", "appended", "reused", "rebuilt"]
    session_revision: Annotated[int, Field(gt=0)]
    serializer_version: str
    model: str
    model_revision: str
    tokenizer_revision: str
    tokenizer_sha256: str
    eos_token_id: Annotated[int, Field(ge=0)]
    engine_build_id: str
    io_plugin_version: str
    pooling_capabilities: list[str]

    @model_validator(mode="after")
    def validate_token_accounting(self) -> "ArcSessionPoolingResponse":
        if self.serialized_tokens + self.truncated_tokens != self.full_history_tokens:
            raise ValueError("session response full-history token accounting diverged")
        if self.retained_prefix_tokens + self.appended_tokens != self.serialized_tokens:
            raise ValueError("session response retained-token accounting diverged")
        if self.session_action == "reused" and self.appended_tokens != 0:
            raise ValueError("reused session responses cannot append tokens")
        if self.session_action in {"created", "rebuilt"} and (
            self.retained_prefix_tokens != 0
        ):
            raise ValueError("created or rebuilt sessions cannot retain a prefix")
        return self


class ArcSessionCloseResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    closed: bool


class ArcSessionHealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["ok"]
    resident_sessions: Annotated[int, Field(ge=0)]
    resident_tokens: Annotated[int, Field(ge=0)]
    max_sessions: Annotated[int, Field(gt=0)]
    max_resident_tokens: Annotated[int, Field(gt=0)]
    pooling_capabilities: list[str]


class ArcSessionCoordinatorMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tokenization_calls_total: Annotated[int, Field(ge=0)]
    tokenization_seconds_total: Annotated[float, Field(ge=0)]
    requests_started_total: Annotated[int, Field(ge=0)]
    requests_succeeded_total: Annotated[int, Field(ge=0)]
    requests_failed_total: Annotated[int, Field(ge=0)]
    requests_inflight: Annotated[int, Field(ge=0)]
    requests_inflight_max: Annotated[int, Field(ge=0)]
    request_seconds_total: Annotated[float, Field(ge=0)]
    session_lock_contentions_total: Annotated[int, Field(ge=0)]
    session_lock_waiters: Annotated[int, Field(ge=0)]
    session_lock_waiters_max: Annotated[int, Field(ge=0)]
    session_lock_wait_seconds_total: Annotated[float, Field(ge=0)]
    backend_calls_started_total: Annotated[int, Field(ge=0)]
    backend_calls_succeeded_total: Annotated[int, Field(ge=0)]
    backend_calls_failed_total: Annotated[int, Field(ge=0)]
    backend_inflight: Annotated[int, Field(ge=0)]
    backend_inflight_max: Annotated[int, Field(ge=0)]
    backend_seconds_total: Annotated[float, Field(ge=0)]
    backend_appended_tokens_total: Annotated[int, Field(ge=0)]


class ArcSessionEngineMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    available: bool
    measurement_scope: Literal["retained_append"] | None = None
    requests_running: Annotated[int, Field(ge=0)] | None = None
    requests_waiting: Annotated[int, Field(ge=0)] | None = None
    requests_running_max: Annotated[int, Field(ge=0)] | None = None
    requests_waiting_max: Annotated[int, Field(ge=0)] | None = None
    requests_scheduled_max: Annotated[int, Field(ge=0)] | None = None
    scheduler_updates_total: Annotated[int, Field(ge=0)] | None = None
    queue_time_observations: Annotated[int, Field(ge=0)] | None = None
    queue_time_seconds_total: Annotated[float, Field(ge=0)] | None = None
    inference_time_observations: Annotated[int, Field(ge=0)] | None = None
    inference_time_seconds_total: Annotated[float, Field(ge=0)] | None = None
    e2e_time_observations: Annotated[int, Field(ge=0)] | None = None
    e2e_time_seconds_total: Annotated[float, Field(ge=0)] | None = None
    prompt_token_observations: Annotated[int, Field(ge=0)] | None = None
    prompt_tokens_total: Annotated[float, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> "ArcSessionEngineMetrics":
        values = (
            self.measurement_scope,
            self.requests_running,
            self.requests_waiting,
            self.requests_running_max,
            self.requests_waiting_max,
            self.requests_scheduled_max,
            self.scheduler_updates_total,
            self.queue_time_observations,
            self.queue_time_seconds_total,
            self.inference_time_observations,
            self.inference_time_seconds_total,
            self.e2e_time_observations,
            self.e2e_time_seconds_total,
            self.prompt_token_observations,
            self.prompt_tokens_total,
        )
        if self.available != all(value is not None for value in values):
            raise ValueError("engine metric availability diverged from values")
        return self


class ArcSessionMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["rayline.arc.session-metrics-response.v4"]
    coordinator: ArcSessionCoordinatorMetrics
    engine: ArcSessionEngineMetrics
