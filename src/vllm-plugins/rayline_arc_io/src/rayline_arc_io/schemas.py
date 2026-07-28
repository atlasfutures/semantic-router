# SPDX-License-Identifier: Apache-2.0

"""Strict wire schemas for the Rayline ARC pooling endpoint."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from .constants import (
    EMBEDDING_DIMENSION,
    MAX_REQUEST_BYTES,
    MAX_TURNS,
    REQUEST_SCHEMA_VERSION,
)

EpisodeIDHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ArcTurn(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    role: Literal["user", "assistant"]
    text: str


class ArcPoolingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["rayline.arc.pooling-request.v1"] = REQUEST_SCHEMA_VERSION
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
