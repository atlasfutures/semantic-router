# SPDX-License-Identifier: Apache-2.0

"""Adapter from the bounded coordinator to vLLM's retained pooling API."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from typing import Any

from .constants import EMBEDDING_DIMENSION
from .session_coordinator import RetainedPoolingOutput
from .session_metrics import SessionEngineMetricsSnapshot

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


class VLLMSessionEngineMetricsProvider:
    """Return a curated, payload-free snapshot of vLLM scheduler metrics."""

    def __init__(
        self,
        snapshot_reader: Callable[[], list[Any]] | None = None,
    ) -> None:
        self._snapshot_reader = snapshot_reader

    def _read(self) -> list[Any]:
        if self._snapshot_reader is not None:
            return self._snapshot_reader()
        from vllm.v1.metrics.reader import get_metrics_snapshot  # noqa: PLC0415

        return get_metrics_snapshot()

    def __call__(self) -> SessionEngineMetricsSnapshot:
        metrics = self._read()

        def total(name: str, attribute: str) -> float | None:
            values = [
                float(getattr(metric, attribute))
                for metric in metrics
                if getattr(metric, "name", None) == name and hasattr(metric, attribute)
            ]
            return sum(values) if values else None

        running = total("vllm:num_requests_running", "value")
        waiting = total("vllm:num_requests_waiting", "value")
        queue_count = total("vllm:request_queue_time_seconds", "count")
        queue_seconds = total("vllm:request_queue_time_seconds", "sum")
        inference_count = total("vllm:request_inference_time_seconds", "count")
        inference_seconds = total("vllm:request_inference_time_seconds", "sum")
        e2e_count = total("vllm:e2e_request_latency_seconds", "count")
        e2e_seconds = total("vllm:e2e_request_latency_seconds", "sum")
        prompt_count = total("vllm:request_prompt_tokens", "count")
        prompt_tokens = total("vllm:request_prompt_tokens", "sum")
        required = (
            running,
            waiting,
            queue_count,
            queue_seconds,
            inference_count,
            inference_seconds,
            e2e_count,
            e2e_seconds,
            prompt_count,
            prompt_tokens,
        )
        if any(value is None for value in required):
            return SessionEngineMetricsSnapshot(available=False)
        return SessionEngineMetricsSnapshot(
            available=True,
            requests_running=int(running),
            requests_waiting=int(waiting),
            queue_time_observations=int(queue_count),
            queue_time_seconds_total=queue_seconds,
            inference_time_observations=int(inference_count),
            inference_time_seconds_total=inference_seconds,
            e2e_time_observations=int(e2e_count),
            e2e_time_seconds_total=e2e_seconds,
            prompt_token_observations=int(prompt_count),
            prompt_tokens_total=prompt_tokens,
        )
