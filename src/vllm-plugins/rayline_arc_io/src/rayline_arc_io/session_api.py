# SPDX-License-Identifier: Apache-2.0

"""Versioned HTTP boundary for retained Rayline ARC pooling sessions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .constants import (
    EOS_TOKEN_ID,
    MODEL_ID,
    MODEL_REVISION,
    PLUGIN_VERSION,
    SERIALIZER_VERSION,
    SESSION_CAPABILITIES,
    SESSION_RESPONSE_SCHEMA_VERSION,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)
from .schemas import (
    ArcSessionCloseResponse,
    ArcSessionCoordinatorMetrics,
    ArcSessionEngineMetrics,
    ArcSessionHealthResponse,
    ArcSessionMetricsResponse,
    ArcSessionPoolingRequest,
    ArcSessionPoolingResponse,
    EpisodeIDHash,
)
from .serializer import TokenBlockSerializer, TokenizationResult
from .session_coordinator import (
    SessionBusyError,
    SessionCapacityError,
    SessionCoordinator,
    SessionCoordinatorError,
    SessionEncodingResult,
)
from .session_metrics import (
    SessionEngineMetricsProvider,
    SessionEngineMetricsSnapshot,
)


@dataclass(frozen=True)
class SessionAPIMetadata:
    engine_build_id: str


def _validation_details(error: RequestValidationError) -> list[dict[str, str]]:
    """Return field/type only; FastAPI's default includes request text."""
    return [
        {
            "location": ".".join(str(part) for part in item["loc"]),
            "type": item["type"],
        }
        for item in error.errors()
    ]


def _pooling_response(
    tokenization: TokenizationResult,
    result: SessionEncodingResult,
    metadata: SessionAPIMetadata,
) -> ArcSessionPoolingResponse:
    return ArcSessionPoolingResponse(
        schema_version=SESSION_RESPONSE_SCHEMA_VERSION,
        embedding=list(result.embedding),
        serialized_tokens=len(tokenization.input_ids),
        full_history_tokens=tokenization.full_tokens,
        truncated_tokens=tokenization.truncated_tokens,
        retained_prefix_tokens=result.retained_prefix_tokens,
        appended_tokens=result.appended_tokens,
        session_action=result.action,
        session_revision=result.revision,
        serializer_version=SERIALIZER_VERSION,
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_sha256=TOKENIZER_SHA256,
        eos_token_id=EOS_TOKEN_ID,
        engine_build_id=metadata.engine_build_id,
        io_plugin_version=PLUGIN_VERSION,
        pooling_capabilities=list(SESSION_CAPABILITIES),
    )


def _metrics_response(
    coordinator: SessionCoordinator,
    engine_metrics_provider: SessionEngineMetricsProvider | None,
) -> ArcSessionMetricsResponse:
    coordinator_snapshot = coordinator.metrics_snapshot()
    if engine_metrics_provider is None:
        engine_snapshot = SessionEngineMetricsSnapshot(available=False)
    else:
        try:
            engine_snapshot = engine_metrics_provider()
        except Exception as error:
            raise HTTPException(status_code=503, detail="engine_metrics") from error
    return ArcSessionMetricsResponse(
        schema_version="rayline.arc.session-metrics-response.v4",
        coordinator=ArcSessionCoordinatorMetrics(**asdict(coordinator_snapshot)),
        engine=ArcSessionEngineMetrics(**asdict(engine_snapshot)),
    )


def create_session_app(
    coordinator: SessionCoordinator,
    serializer: TokenBlockSerializer,
    metadata: SessionAPIMetadata,
    engine_metrics_provider: SessionEngineMetricsProvider | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Rayline ARC retained pooling session",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: object,
        error: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_request", "fields": _validation_details(error)},
        )

    @app.get(
        "/health",
        response_model=ArcSessionHealthResponse,
    )
    async def health() -> ArcSessionHealthResponse:
        stats = await coordinator.stats()
        return ArcSessionHealthResponse(
            status="ok",
            resident_sessions=stats.resident_sessions,
            resident_tokens=stats.resident_tokens,
            max_sessions=stats.max_sessions,
            max_resident_tokens=stats.max_resident_tokens,
            pooling_capabilities=list(SESSION_CAPABILITIES),
        )

    @app.get(
        "/v1/rayline/arc/session/metrics",
        response_model=ArcSessionMetricsResponse,
    )
    async def metrics() -> ArcSessionMetricsResponse:
        return _metrics_response(coordinator, engine_metrics_provider)

    @app.post(
        "/v1/rayline/arc/session/pooling",
        response_model=ArcSessionPoolingResponse,
    )
    async def encode(
        request: ArcSessionPoolingRequest,
    ) -> ArcSessionPoolingResponse:
        tokenization_started = time.perf_counter()
        try:
            tokenization = serializer.tokenize(request.turns)
        finally:
            coordinator.observe_tokenization(
                max(0.0, time.perf_counter() - tokenization_started)
            )
        try:
            result = await coordinator.encode(
                request.episode_id_hash,
                tokenization.input_ids,
            )
        except SessionCapacityError as error:
            raise HTTPException(
                status_code=429,
                detail="session_capacity",
            ) from error
        except SessionCoordinatorError as error:
            raise HTTPException(
                status_code=503,
                detail="session_state",
            ) from error
        except Exception as error:
            raise HTTPException(
                status_code=503,
                detail="session_backend",
            ) from error

        return _pooling_response(tokenization, result, metadata)

    @app.delete(
        "/v1/rayline/arc/session/{episode_id_hash}",
        response_model=ArcSessionCloseResponse,
    )
    async def close_session(
        episode_id_hash: EpisodeIDHash,
    ) -> ArcSessionCloseResponse:
        try:
            closed = await coordinator.close_session(episode_id_hash)
        except SessionBusyError as error:
            raise HTTPException(status_code=409, detail="session_busy") from error
        return ArcSessionCloseResponse(closed=closed)

    return app
