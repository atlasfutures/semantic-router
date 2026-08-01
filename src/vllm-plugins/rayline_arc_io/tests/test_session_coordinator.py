# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass, field

import pytest
from rayline_arc_io.session_coordinator import (
    RetainedPoolingOutput,
    SessionCapacityError,
    SessionCoordinator,
)

EXPECTED_REVISION_AFTER_REBUILD = 3
EXPECTED_RESIDENT_SESSIONS = 2
EXPECTED_RESIDENT_TOKENS = 5
EXPECTED_RECREATED_TOKENS = 3
EXPECTED_BACKEND_COUNT = 2
EXPECTED_CONCURRENT_APPENDS = 2


@dataclass
class FakeBackend:
    episode_id_hash: str
    appended: list[tuple[int, ...]] = field(default_factory=list)
    cumulative: list[int] = field(default_factory=list)
    closed: bool = False

    async def append(self, token_ids: tuple[int, ...]) -> RetainedPoolingOutput:
        assert not self.closed
        self.appended.append(token_ids)
        self.cumulative.extend(token_ids)
        return RetainedPoolingOutput(
            embedding=(float(len(self.cumulative)),),
            cumulative_token_ids=tuple(self.cumulative),
        )

    async def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self) -> None:
        self.backends: list[FakeBackend] = []

    def __call__(self, episode_id_hash: str) -> FakeBackend:
        backend = FakeBackend(episode_id_hash)
        self.backends.append(backend)
        return backend


def run(coroutine):
    return asyncio.run(coroutine)


def test_session_create_append_retry_and_prefix_mismatch_rebuild() -> None:
    async def scenario() -> None:
        factory = FakeFactory()
        coordinator = SessionCoordinator(
            factory,
            max_sessions=4,
            max_resident_tokens=100,
            idle_ttl_seconds=60,
        )

        created = await coordinator.encode("a" * 64, (1, 2))
        appended = await coordinator.encode("a" * 64, (1, 2, 3, 4))
        reused = await coordinator.encode("a" * 64, (1, 2, 3, 4))
        rebuilt = await coordinator.encode("a" * 64, (9, 10, 11))

        assert (created.action, created.appended_tokens) == ("created", 2)
        assert (
            appended.action,
            appended.retained_prefix_tokens,
            appended.appended_tokens,
        ) == ("appended", 2, 2)
        assert (
            reused.action,
            reused.retained_prefix_tokens,
            reused.appended_tokens,
        ) == ("reused", 4, 0)
        assert (
            rebuilt.action,
            rebuilt.retained_prefix_tokens,
            rebuilt.appended_tokens,
        ) == ("rebuilt", 0, 3)
        assert [backend.appended for backend in factory.backends] == [
            [(1, 2), (3, 4)],
            [(9, 10, 11)],
        ]
        assert factory.backends[0].closed is True
        assert rebuilt.revision == EXPECTED_REVISION_AFTER_REBUILD

    run(scenario())


def test_session_count_and_token_bounds_evict_oldest_idle_session() -> None:
    async def scenario() -> None:
        now = [0.0]
        factory = FakeFactory()
        coordinator = SessionCoordinator(
            factory,
            max_sessions=2,
            max_resident_tokens=5,
            idle_ttl_seconds=60,
            clock=lambda: now[0],
        )
        await coordinator.encode("a" * 64, (1, 2))
        now[0] += 1
        await coordinator.encode("b" * 64, (3, 4))
        now[0] += 1
        await coordinator.encode("c" * 64, (5, 6, 7))

        stats = await coordinator.stats()
        assert stats.resident_sessions == EXPECTED_RESIDENT_SESSIONS
        assert stats.resident_tokens == EXPECTED_RESIDENT_TOKENS
        assert factory.backends[0].closed is True
        assert factory.backends[1].closed is False
        assert factory.backends[2].closed is False

    run(scenario())


def test_expired_session_is_recreated_from_complete_history() -> None:
    async def scenario() -> None:
        now = [10.0]
        factory = FakeFactory()
        coordinator = SessionCoordinator(
            factory,
            max_sessions=2,
            max_resident_tokens=10,
            idle_ttl_seconds=5,
            clock=lambda: now[0],
        )
        await coordinator.encode("a" * 64, (1, 2))
        now[0] = 16.0
        recreated = await coordinator.encode("a" * 64, (1, 2, 3))

        assert recreated.action == "created"
        assert recreated.appended_tokens == EXPECTED_RECREATED_TOKENS
        assert len(factory.backends) == EXPECTED_BACKEND_COUNT
        assert factory.backends[0].closed is True

    run(scenario())


def test_single_history_cannot_exceed_global_token_bound() -> None:
    coordinator = SessionCoordinator(
        FakeFactory(),
        max_sessions=1,
        max_resident_tokens=2,
        idle_ttl_seconds=60,
    )
    with pytest.raises(SessionCapacityError, match="history"):
        run(coordinator.encode("a" * 64, (1, 2, 3)))


def test_different_sessions_execute_concurrently() -> None:
    async def scenario() -> None:
        entered = 0
        max_entered = 0
        both_entered = asyncio.Event()

        class BlockingBackend(FakeBackend):
            async def append(
                self,
                token_ids: tuple[int, ...],
            ) -> RetainedPoolingOutput:
                nonlocal entered, max_entered
                entered += 1
                max_entered = max(max_entered, entered)
                if entered == EXPECTED_CONCURRENT_APPENDS:
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), timeout=1)
                output = await super().append(token_ids)
                entered -= 1
                return output

        backends: list[BlockingBackend] = []

        def factory(episode_id_hash: str) -> BlockingBackend:
            backend = BlockingBackend(episode_id_hash)
            backends.append(backend)
            return backend

        coordinator = SessionCoordinator(
            factory,
            max_sessions=2,
            max_resident_tokens=10,
            idle_ttl_seconds=60,
        )
        await asyncio.gather(
            coordinator.encode("a" * 64, (1, 2)),
            coordinator.encode("b" * 64, (3, 4)),
        )
        assert max_entered == EXPECTED_CONCURRENT_APPENDS
        metrics = coordinator.metrics_snapshot()
        assert metrics.requests_started_total == EXPECTED_CONCURRENT_APPENDS
        assert metrics.requests_succeeded_total == EXPECTED_CONCURRENT_APPENDS
        assert metrics.requests_failed_total == 0
        assert metrics.requests_inflight == 0
        assert metrics.requests_inflight_max == EXPECTED_CONCURRENT_APPENDS
        assert metrics.backend_calls_succeeded_total == EXPECTED_CONCURRENT_APPENDS
        assert metrics.backend_inflight_max == EXPECTED_CONCURRENT_APPENDS
        assert metrics.session_lock_contentions_total == 0

    run(scenario())


def test_same_session_reports_lock_contention_without_parallel_backend_work() -> None:
    async def scenario() -> None:
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        class BlockingBackend(FakeBackend):
            async def append(
                self,
                token_ids: tuple[int, ...],
            ) -> RetainedPoolingOutput:
                first_entered.set()
                await asyncio.wait_for(release_first.wait(), timeout=1)
                return await super().append(token_ids)

        coordinator = SessionCoordinator(
            BlockingBackend,
            max_sessions=1,
            max_resident_tokens=10,
            idle_ttl_seconds=60,
        )
        episode_id = "a" * 64
        first = asyncio.create_task(coordinator.encode(episode_id, (1, 2)))
        await asyncio.wait_for(first_entered.wait(), timeout=1)
        second = asyncio.create_task(coordinator.encode(episode_id, (1, 2)))
        for _attempt in range(10):
            await asyncio.sleep(0)
            active = coordinator.metrics_snapshot()
            if active.session_lock_waiters == 1:
                break
        else:
            raise AssertionError("second same-session request never waited on lock")

        assert active.requests_inflight == EXPECTED_CONCURRENT_APPENDS
        assert active.backend_inflight == 1
        assert active.session_lock_waiters == 1
        release_first.set()
        results = await asyncio.gather(first, second)
        assert [result.action for result in results] == ["created", "reused"]

        final = coordinator.metrics_snapshot()
        assert final.requests_succeeded_total == EXPECTED_CONCURRENT_APPENDS
        assert final.requests_inflight_max == EXPECTED_CONCURRENT_APPENDS
        assert final.session_lock_contentions_total == 1
        assert final.session_lock_waiters == 0
        assert final.session_lock_waiters_max == 1
        assert final.session_lock_wait_seconds_total >= 0
        assert final.backend_calls_succeeded_total == 1
        assert final.backend_inflight_max == 1

    run(scenario())
