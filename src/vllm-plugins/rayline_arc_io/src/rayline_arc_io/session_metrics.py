# SPDX-License-Identifier: Apache-2.0

"""Payload-free telemetry for retained Rayline ARC pooling sessions."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import Protocol


@dataclass(frozen=True)
class SessionCoordinatorMetricsSnapshot:
    tokenization_calls_total: int
    tokenization_seconds_total: float
    requests_started_total: int
    requests_succeeded_total: int
    requests_failed_total: int
    requests_inflight: int
    requests_inflight_max: int
    request_seconds_total: float
    session_lock_contentions_total: int
    session_lock_waiters: int
    session_lock_waiters_max: int
    session_lock_wait_seconds_total: float
    backend_calls_started_total: int
    backend_calls_succeeded_total: int
    backend_calls_failed_total: int
    backend_inflight: int
    backend_inflight_max: int
    backend_seconds_total: float
    backend_appended_tokens_total: int


@dataclass(frozen=True)
class SessionAppendMetricsSnapshot:
    observations: int
    queue_time_seconds_total: float
    inference_time_seconds_total: float
    e2e_time_seconds_total: float
    appended_tokens_total: int


@dataclass(frozen=True)
class SessionEngineMetricsSnapshot:
    available: bool
    measurement_scope: str | None = None
    requests_running: int | None = None
    requests_waiting: int | None = None
    requests_running_max: int | None = None
    requests_waiting_max: int | None = None
    requests_scheduled_max: int | None = None
    scheduler_updates_total: int | None = None
    queue_time_observations: int | None = None
    queue_time_seconds_total: float | None = None
    inference_time_observations: int | None = None
    inference_time_seconds_total: float | None = None
    e2e_time_observations: int | None = None
    e2e_time_seconds_total: float | None = None
    prompt_token_observations: int | None = None
    prompt_tokens_total: float | None = None


class SessionEngineMetricsProvider(Protocol):
    def __call__(self) -> SessionEngineMetricsSnapshot: ...


class SessionMetrics:
    """Track aggregate concurrency and latency without request identifiers.

    Counter mutation is synchronous and protected by a short thread lock so a
    future multi-threaded ASGI runtime cannot lose updates. Timers surround
    existing awaits but add no admission semaphore and therefore do not alter
    the service's concurrency contract.
    """

    def __init__(self, *, timer: Callable[[], float] = time.perf_counter) -> None:
        self._timer = timer
        self._lock = Lock()
        self._tokenization_calls_total = 0
        self._tokenization_seconds_total = 0.0
        self._requests_started_total = 0
        self._requests_succeeded_total = 0
        self._requests_failed_total = 0
        self._requests_inflight = 0
        self._requests_inflight_max = 0
        self._request_seconds_total = 0.0
        self._session_lock_contentions_total = 0
        self._session_lock_waiters = 0
        self._session_lock_waiters_max = 0
        self._session_lock_wait_seconds_total = 0.0
        self._backend_calls_started_total = 0
        self._backend_calls_succeeded_total = 0
        self._backend_calls_failed_total = 0
        self._backend_inflight = 0
        self._backend_inflight_max = 0
        self._backend_seconds_total = 0.0
        self._backend_appended_tokens_total = 0
        self._append_observations = 0
        self._append_queue_time_seconds_total = 0.0
        self._append_inference_time_seconds_total = 0.0
        self._append_e2e_time_seconds_total = 0.0
        self._append_tokens_total = 0

    def observe_tokenization(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("tokenization duration cannot be negative")
        with self._lock:
            self._tokenization_calls_total += 1
            self._tokenization_seconds_total += seconds

    @asynccontextmanager
    async def track_request(self) -> AsyncIterator[None]:
        started = self._timer()
        with self._lock:
            self._requests_started_total += 1
            self._requests_inflight += 1
            self._requests_inflight_max = max(
                self._requests_inflight_max,
                self._requests_inflight,
            )
        try:
            yield
        except BaseException:
            with self._lock:
                self._requests_failed_total += 1
            raise
        else:
            with self._lock:
                self._requests_succeeded_total += 1
        finally:
            elapsed = self._timer() - started
            with self._lock:
                self._requests_inflight -= 1
                self._request_seconds_total += max(0.0, elapsed)

    @asynccontextmanager
    async def track_session_lock(
        self,
        session_lock: asyncio.Lock,
    ) -> AsyncIterator[None]:
        contended = session_lock.locked()
        started = self._timer()
        if contended:
            with self._lock:
                self._session_lock_contentions_total += 1
                self._session_lock_waiters += 1
                self._session_lock_waiters_max = max(
                    self._session_lock_waiters_max,
                    self._session_lock_waiters,
                )
        acquired = False
        try:
            await session_lock.acquire()
            acquired = True
        finally:
            if contended:
                elapsed = self._timer() - started
                with self._lock:
                    self._session_lock_waiters -= 1
                    self._session_lock_wait_seconds_total += max(0.0, elapsed)
        try:
            yield
        finally:
            if acquired:
                session_lock.release()

    @asynccontextmanager
    async def track_backend(self, appended_tokens: int) -> AsyncIterator[None]:
        if appended_tokens < 0:
            raise ValueError("backend appended tokens cannot be negative")
        started = self._timer()
        with self._lock:
            self._backend_calls_started_total += 1
            self._backend_inflight += 1
            self._backend_inflight_max = max(
                self._backend_inflight_max,
                self._backend_inflight,
            )
        try:
            yield
        except BaseException:
            with self._lock:
                self._backend_calls_failed_total += 1
            raise
        else:
            with self._lock:
                self._backend_calls_succeeded_total += 1
                self._backend_appended_tokens_total += appended_tokens
        finally:
            elapsed = self._timer() - started
            with self._lock:
                self._backend_inflight -= 1
                self._backend_seconds_total += max(0.0, elapsed)

    def observe_engine_append(
        self,
        *,
        queue_time: float,
        inference_time: float,
        e2e_time: float,
        appended_tokens: int,
    ) -> None:
        values = (queue_time, inference_time, e2e_time)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("engine append duration must be finite and non-negative")
        if appended_tokens < 0:
            raise ValueError("engine appended tokens cannot be negative")
        with self._lock:
            self._append_observations += 1
            self._append_queue_time_seconds_total += queue_time
            self._append_inference_time_seconds_total += inference_time
            self._append_e2e_time_seconds_total += e2e_time
            self._append_tokens_total += appended_tokens

    def snapshot(self) -> SessionCoordinatorMetricsSnapshot:
        with self._lock:
            return SessionCoordinatorMetricsSnapshot(
                tokenization_calls_total=self._tokenization_calls_total,
                tokenization_seconds_total=self._tokenization_seconds_total,
                requests_started_total=self._requests_started_total,
                requests_succeeded_total=self._requests_succeeded_total,
                requests_failed_total=self._requests_failed_total,
                requests_inflight=self._requests_inflight,
                requests_inflight_max=self._requests_inflight_max,
                request_seconds_total=self._request_seconds_total,
                session_lock_contentions_total=self._session_lock_contentions_total,
                session_lock_waiters=self._session_lock_waiters,
                session_lock_waiters_max=self._session_lock_waiters_max,
                session_lock_wait_seconds_total=(self._session_lock_wait_seconds_total),
                backend_calls_started_total=self._backend_calls_started_total,
                backend_calls_succeeded_total=self._backend_calls_succeeded_total,
                backend_calls_failed_total=self._backend_calls_failed_total,
                backend_inflight=self._backend_inflight,
                backend_inflight_max=self._backend_inflight_max,
                backend_seconds_total=self._backend_seconds_total,
                backend_appended_tokens_total=self._backend_appended_tokens_total,
            )

    def append_snapshot(self) -> SessionAppendMetricsSnapshot:
        with self._lock:
            return SessionAppendMetricsSnapshot(
                observations=self._append_observations,
                queue_time_seconds_total=self._append_queue_time_seconds_total,
                inference_time_seconds_total=(
                    self._append_inference_time_seconds_total
                ),
                e2e_time_seconds_total=self._append_e2e_time_seconds_total,
                appended_tokens_total=self._append_tokens_total,
            )
