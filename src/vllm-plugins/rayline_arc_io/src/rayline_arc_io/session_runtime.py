# SPDX-License-Identifier: Apache-2.0

"""Adapter from the bounded coordinator to vLLM's retained pooling API."""

from __future__ import annotations

import math
import uuid
from typing import Any

from .constants import EMBEDDING_DIMENSION
from .session_coordinator import RetainedPoolingOutput

_NORMALIZED_EMBEDDING_TOLERANCE = 1e-4


class VLLMRetainedPoolingBackend:
    def __init__(self, engine: Any, episode_id_hash: str) -> None:
        from vllm.pooling_params import PoolingParams  # noqa: PLC0415
        from vllm.v1.engine.pooling_session import (  # noqa: PLC0415
            AsyncPoolingSession,
        )

        request_id = f"rayline-arc-session-{episode_id_hash[:12]}-{uuid.uuid4().hex}"
        self._session = AsyncPoolingSession(
            engine,
            pooling_params=PoolingParams(
                task="embed",
                use_activation=True,
                retain_pooling_state=True,
            ),
            request_id=request_id,
        )
        self._cumulative_token_ids: list[int] = []

    async def append(self, token_ids: tuple[int, ...]) -> RetainedPoolingOutput:
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        output = await self._session.append(
            TokensPrompt(prompt_token_ids=list(token_ids))
        )
        self._cumulative_token_ids.extend(token_ids)
        if output.finished:
            raise RuntimeError("retained pooling append unexpectedly finished")
        if output.num_cached_tokens != 0:
            raise RuntimeError("retained pooling used automatic prefix cache state")
        if output.prompt_token_ids != self._cumulative_token_ids:
            raise RuntimeError("retained pooling cumulative token identity diverged")

        values = output.outputs.data.tolist()
        if not isinstance(values, list) or len(values) != EMBEDDING_DIMENSION:
            raise RuntimeError("retained pooling returned an invalid embedding shape")
        embedding = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in embedding):
            raise RuntimeError("retained pooling returned a non-finite embedding")
        norm = math.sqrt(sum(value * value for value in embedding))
        if abs(norm - 1.0) > _NORMALIZED_EMBEDDING_TOLERANCE:
            raise RuntimeError("retained pooling returned an unnormalized embedding")
        return RetainedPoolingOutput(
            embedding=embedding,
            cumulative_token_ids=tuple(self._cumulative_token_ids),
        )

    async def close(self) -> None:
        await self._session.close()


class VLLMRetainedPoolingBackendFactory:
    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def __call__(self, episode_id_hash: str) -> VLLMRetainedPoolingBackend:
        return VLLMRetainedPoolingBackend(self._engine, episode_id_hash)
