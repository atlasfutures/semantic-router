# SPDX-License-Identifier: Apache-2.0

"""Bounded lifecycle for explicit retained vLLM pooling sessions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .session_metrics import SessionCoordinatorMetricsSnapshot, SessionMetrics


class SessionCoordinatorError(RuntimeError):
    """Base class for bounded, payload-free session failures."""


class SessionCapacityError(SessionCoordinatorError):
    """The configured session or token residency bound is exhausted."""


class SessionBusyError(SessionCoordinatorError):
    """An explicit close raced an active request for the same session."""


@dataclass(frozen=True)
class RetainedPoolingOutput:
    embedding: tuple[float, ...]
    cumulative_token_ids: tuple[int, ...]


class RetainedPoolingBackend(Protocol):
    async def append(self, token_ids: tuple[int, ...]) -> RetainedPoolingOutput: ...

    async def close(self) -> None: ...


class RetainedPoolingBackendFactory(Protocol):
    def __call__(self, episode_id_hash: str) -> RetainedPoolingBackend: ...


@dataclass(frozen=True)
class SessionEncodingResult:
    embedding: tuple[float, ...]
    action: str
    retained_prefix_tokens: int
    appended_tokens: int
    revision: int


@dataclass(frozen=True)
class SessionCoordinatorStats:
    resident_sessions: int
    resident_tokens: int
    max_sessions: int
    max_resident_tokens: int


@dataclass
class _ResidentSession:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    backend: RetainedPoolingBackend | None = None
    token_ids: tuple[int, ...] = ()
    embedding: tuple[float, ...] = ()
    revision: int = 0
    last_used: float = 0.0
    borrowers: int = 0
    accounted_tokens: int = 0


class SessionCoordinator:
    """Map full histories onto bounded, exact-prefix pooling sessions.

    Requests always carry the complete reconstructible history. An exact token
    extension appends only its suffix. An identical retry returns the preceding
    embedding. Any other token sequence closes the old engine request and
    rebuilds from the supplied full history.

    Same-episode work is serialized while different episodes may execute
    concurrently. Residency reservations happen before GPU work so concurrent
    creates cannot collectively exceed the declared token bound.
    """

    def __init__(
        self,
        backend_factory: RetainedPoolingBackendFactory,
        *,
        max_sessions: int,
        max_resident_tokens: int,
        idle_ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
        metrics: SessionMetrics | None = None,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if max_resident_tokens < 1:
            raise ValueError("max_resident_tokens must be positive")
        if idle_ttl_seconds <= 0:
            raise ValueError("idle_ttl_seconds must be positive")
        self._backend_factory = backend_factory
        self._max_sessions = max_sessions
        self._max_resident_tokens = max_resident_tokens
        self._idle_ttl_seconds = idle_ttl_seconds
        self._clock = clock
        self._metrics = metrics if metrics is not None else SessionMetrics()
        self._entries: dict[str, _ResidentSession] = {}
        self._metadata_lock = asyncio.Lock()

    async def encode(
        self,
        episode_id_hash: str,
        token_ids: tuple[int, ...],
    ) -> SessionEncodingResult:
        if not episode_id_hash:
            raise ValueError("episode_id_hash must be non-empty")
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        if len(token_ids) > self._max_resident_tokens:
            raise SessionCapacityError("history exceeds retained-token capacity")

        async with self._metrics.track_request():
            entry, victims = await self._borrow_entry(episode_id_hash)
            try:
                await self._close_backends(victims)
                async with self._metrics.track_session_lock(entry.lock):
                    return await self._encode_locked(
                        episode_id_hash,
                        entry,
                        token_ids,
                    )
            finally:
                await self._release_entry(episode_id_hash, entry)

    async def _borrow_entry(
        self,
        episode_id_hash: str,
    ) -> tuple[_ResidentSession, list[RetainedPoolingBackend]]:
        now = self._clock()
        async with self._metadata_lock:
            victims = self._evict_expired_locked(now)
            entry = self._entries.get(episode_id_hash)
            if entry is None:
                victims.extend(self._make_session_room_locked())
                if len(self._entries) >= self._max_sessions:
                    raise SessionCapacityError("retained-session capacity is busy")
                entry = _ResidentSession(last_used=now)
                self._entries[episode_id_hash] = entry
            entry.borrowers += 1
            return entry, victims

    async def _release_entry(
        self,
        episode_id_hash: str,
        entry: _ResidentSession,
    ) -> None:
        async with self._metadata_lock:
            entry.borrowers -= 1
            if entry.borrowers < 0:
                raise RuntimeError("retained-session borrower count underflow")
            if (
                entry.backend is None
                and entry.borrowers == 0
                and self._entries.get(episode_id_hash) is entry
            ):
                self._entries.pop(episode_id_hash)

    def _evict_expired_locked(
        self,
        now: float,
    ) -> list[RetainedPoolingBackend]:
        expired = [
            episode_id_hash
            for episode_id_hash, entry in self._entries.items()
            if entry.borrowers == 0 and now - entry.last_used > self._idle_ttl_seconds
        ]
        return self._remove_entries_locked(expired)

    def _make_session_room_locked(self) -> list[RetainedPoolingBackend]:
        victims: list[RetainedPoolingBackend] = []
        while len(self._entries) >= self._max_sessions:
            episode_id_hash = self._oldest_idle_episode_locked(exclude=None)
            if episode_id_hash is None:
                break
            victims.extend(self._remove_entries_locked([episode_id_hash]))
        return victims

    def _oldest_idle_episode_locked(
        self,
        *,
        exclude: _ResidentSession | None,
    ) -> str | None:
        candidates = (
            (entry.last_used, episode_id_hash)
            for episode_id_hash, entry in self._entries.items()
            if entry is not exclude and entry.borrowers == 0
        )
        return min(candidates, default=(0.0, None))[1]

    def _remove_entries_locked(
        self,
        episode_ids: list[str],
    ) -> list[RetainedPoolingBackend]:
        backends: list[RetainedPoolingBackend] = []
        for episode_id_hash in episode_ids:
            entry = self._entries.pop(episode_id_hash, None)
            if entry is not None and entry.backend is not None:
                backends.append(entry.backend)
        return backends

    async def _reserve_tokens(
        self,
        entry: _ResidentSession,
        requested_tokens: int,
    ) -> list[RetainedPoolingBackend]:
        async with self._metadata_lock:
            current_total = sum(
                candidate.accounted_tokens for candidate in self._entries.values()
            )
            required_total = current_total - entry.accounted_tokens + requested_tokens
            victims: list[RetainedPoolingBackend] = []
            while required_total > self._max_resident_tokens:
                episode_id_hash = self._oldest_idle_episode_locked(exclude=entry)
                if episode_id_hash is None:
                    break
                victim = self._entries[episode_id_hash]
                required_total -= victim.accounted_tokens
                victims.extend(self._remove_entries_locked([episode_id_hash]))
            if required_total > self._max_resident_tokens:
                raise SessionCapacityError("retained-token capacity is busy")
            entry.accounted_tokens = requested_tokens
            return victims

    async def _encode_locked(
        self,
        episode_id_hash: str,
        entry: _ResidentSession,
        token_ids: tuple[int, ...],
    ) -> SessionEncodingResult:
        victims = await self._reserve_tokens(entry, len(token_ids))
        await self._close_backends(victims)

        if entry.backend is not None and token_ids == entry.token_ids:
            entry.last_used = self._clock()
            return SessionEncodingResult(
                embedding=entry.embedding,
                action="reused",
                retained_prefix_tokens=len(token_ids),
                appended_tokens=0,
                revision=entry.revision,
            )

        retained_prefix_tokens = len(entry.token_ids)
        action = "created"
        append_ids = token_ids
        if entry.backend is not None:
            if (
                len(token_ids) > len(entry.token_ids)
                and token_ids[: len(entry.token_ids)] == entry.token_ids
            ):
                action = "appended"
                append_ids = token_ids[len(entry.token_ids) :]
            else:
                action = "rebuilt"
                retained_prefix_tokens = 0
                await entry.backend.close()
                entry.backend = None
        else:
            retained_prefix_tokens = 0

        if entry.backend is None:
            entry.backend = self._backend_factory(episode_id_hash)

        try:
            async with self._metrics.track_backend(len(append_ids)):
                output = await entry.backend.append(append_ids)
        except BaseException:
            failed_backend = entry.backend
            entry.backend = None
            entry.token_ids = ()
            entry.embedding = ()
            await self._clear_reservation(entry)
            if failed_backend is not None:
                await failed_backend.close()
            raise

        if output.cumulative_token_ids != token_ids:
            failed_backend = entry.backend
            entry.backend = None
            entry.token_ids = ()
            entry.embedding = ()
            await self._clear_reservation(entry)
            if failed_backend is not None:
                await failed_backend.close()
            raise SessionCoordinatorError(
                "retained backend returned a divergent cumulative token sequence"
            )

        entry.token_ids = token_ids
        entry.embedding = output.embedding
        entry.revision += 1
        entry.last_used = self._clock()
        return SessionEncodingResult(
            embedding=output.embedding,
            action=action,
            retained_prefix_tokens=retained_prefix_tokens,
            appended_tokens=len(append_ids),
            revision=entry.revision,
        )

    async def _clear_reservation(self, entry: _ResidentSession) -> None:
        async with self._metadata_lock:
            entry.accounted_tokens = 0

    async def close_session(self, episode_id_hash: str) -> bool:
        async with self._metadata_lock:
            entry = self._entries.get(episode_id_hash)
            if entry is None:
                return False
            if entry.borrowers > 0:
                raise SessionBusyError("retained session is busy")
            self._entries.pop(episode_id_hash)
        if entry.backend is not None:
            await entry.backend.close()
        return True

    async def close_all(self) -> None:
        async with self._metadata_lock:
            if any(entry.borrowers > 0 for entry in self._entries.values()):
                raise SessionBusyError("retained sessions are busy")
            backends = self._remove_entries_locked(list(self._entries))
        await self._close_backends(backends)

    async def stats(self) -> SessionCoordinatorStats:
        async with self._metadata_lock:
            return SessionCoordinatorStats(
                resident_sessions=len(self._entries),
                resident_tokens=sum(
                    entry.accounted_tokens for entry in self._entries.values()
                ),
                max_sessions=self._max_sessions,
                max_resident_tokens=self._max_resident_tokens,
            )

    def observe_tokenization(self, seconds: float) -> None:
        self._metrics.observe_tokenization(seconds)

    def metrics_snapshot(self) -> SessionCoordinatorMetricsSnapshot:
        return self._metrics.snapshot()

    @staticmethod
    async def _close_backends(backends: list[RetainedPoolingBackend]) -> None:
        for backend in backends:
            await backend.close()
