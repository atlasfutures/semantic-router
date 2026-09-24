# SPDX-License-Identifier: Apache-2.0

"""Versioned HTTP boundary for retained Rayline ARC pooling sessions."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict, dataclass

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .constants import (
    EOS_TOKEN_ID,
    MODEL_ID,
    MODEL_REVISION,
    PLUGIN_VERSION,
    RUNG_B_CAPABILITIES,
    SERIALIZER_VERSION,
    SESSION_CAPABILITIES,
    SESSION_RESPONSE_SCHEMA_VERSION,
    TOKENIZER_REVISION,
    TOKENIZER_SHA256,
)
from .schemas import (
    ArcPluginPoolingRequest,
    ArcPluginPoolingResponse,
    ArcPoolingResponse,
    ArcSessionCloseResponse,
    ArcSessionCoordinatorMetrics,
    ArcSessionEngineMetrics,
    ArcSessionHealthResponse,
    ArcSessionMetricsResponse,
    ArcSessionPoolingRequest,
    ArcSessionPoolingResponse,
    ArcSessionStartupLogResponse,
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

STARTUP_LOG_SCHEMA_VERSION = "rayline.arc.session-startup-log-response.v1"


@dataclass(frozen=True)
class SessionAPIMetadata:
    engine_build_id: str
    # Engine-sizing lines the deployment retained while building the engine.
    # Empty is a legitimate state and is reported as `captured: false` rather
    # than as an error, so a caller can never mistake "not observed" for
    # "observed nothing".
    startup_log: tuple[str, ...] = ()


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


def _stateless_pooling_response(
    tokenization: TokenizationResult,
    result: SessionEncodingResult,
    metadata: SessionAPIMetadata,
) -> ArcPoolingResponse:
    if (
        result.action != "created"
        or result.retained_prefix_tokens != 0
        or result.appended_tokens != len(tokenization.input_ids)
    ):
        raise RuntimeError("stateless compatibility request retained session state")
    return ArcPoolingResponse(
        embedding=list(result.embedding),
        serialized_tokens=len(tokenization.input_ids),
        full_history_tokens=tokenization.full_tokens,
        truncated_tokens=tokenization.truncated_tokens,
        cached_prefix_tokens=0,
        serializer_version=SERIALIZER_VERSION,
        model=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION,
        tokenizer_sha256=TOKENIZER_SHA256,
        eos_token_id=EOS_TOKEN_ID,
        engine_build_id=metadata.engine_build_id,
        io_plugin_version=PLUGIN_VERSION,
        pooling_capabilities=list(RUNG_B_CAPABILITIES),
    )


async def _encode_stateless(
    request: ArcPluginPoolingRequest,
    coordinator: SessionCoordinator,
    serializer: TokenBlockSerializer,
    metadata: SessionAPIMetadata,
) -> ArcPluginPoolingResponse:
    """Serve the stateless plugin wire through one ephemeral append."""
    if request.data.serving_rung != "B":
        raise HTTPException(status_code=422, detail="serving_rung")
    tokenization_started = time.perf_counter()
    try:
        tokenization = serializer.tokenize(request.data.turns)
    finally:
        coordinator.observe_tokenization(
            max(0.0, time.perf_counter() - tokenization_started)
        )
    ephemeral_episode = hashlib.sha256(
        f"{request.data.episode_id_hash}:{uuid.uuid4().hex}".encode("ascii")
    ).hexdigest()
    try:
        result = await coordinator.encode(
            ephemeral_episode,
            tokenization.input_ids,
        )
    except SessionCapacityError as error:
        raise HTTPException(status_code=429, detail="session_capacity") from error
    except SessionCoordinatorError as error:
        raise HTTPException(status_code=503, detail="session_state") from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="session_backend") from error
    try:
        closed = await coordinator.close_session(ephemeral_episode)
    except SessionCoordinatorError as error:
        raise HTTPException(status_code=503, detail="session_state") from error
    if not closed:
        raise HTTPException(status_code=503, detail="session_state")
    return ArcPluginPoolingResponse(
        request_id=f"pool-{uuid.uuid4().hex}",
        created_at=int(time.time()),
        data=_stateless_pooling_response(tokenization, result, metadata),
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


def _register_stateless_pooling_route(
    app: FastAPI,
    coordinator: SessionCoordinator,
    serializer: TokenBlockSerializer,
    metadata: SessionAPIMetadata,
) -> None:
    @app.post(
        "/pooling",
        response_model=ArcPluginPoolingResponse,
    )
    async def encode_stateless(
        request: ArcPluginPoolingRequest,
    ) -> ArcPluginPoolingResponse:
        return await _encode_stateless(
            request,
            coordinator,
            serializer,
            metadata,
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

    @app.get(
        "/v1/rayline/arc/session/startup-log",
        response_model=ArcSessionStartupLogResponse,
    )
    async def startup_log() -> ArcSessionStartupLogResponse:
        return ArcSessionStartupLogResponse(
            schema_version=STARTUP_LOG_SCHEMA_VERSION,
            engine_build_id=metadata.engine_build_id,
            captured=bool(metadata.startup_log),
            lines=list(metadata.startup_log),
        )

    _register_stateless_pooling_route(app, coordinator, serializer, metadata)

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
