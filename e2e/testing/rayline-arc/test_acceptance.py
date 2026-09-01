#!/usr/bin/env python3
"""Rayline ARC acceptance checks that need no Docker, Envoy or Redis.

Run: python3 e2e/testing/rayline-arc/test_acceptance.py
Or:  make test-rayline-arc-acceptance

Each check names one externally visible contract. The assertions come from the
ported hermetic stack (test_stack.py, route_decision_contract.py) reduced to
what a compose-free run can still observe: the decision-only endpoint runs the
whole ARC path -- episode lease, turn projection, encoder call, policy, commit
-- and stops where dispatch would begin.

What this file does NOT cover, and why: everything past the selection seam.
Provider dispatch, response-header commit ordering, Envoy retry accounting and
the ARC credential header all live behind Bucket B, which is still stubbed on
vsr-next. Those assertions stay in the frozen compose stack until Bucket B
re-seats them.
"""

from __future__ import annotations

import concurrent.futures
import sys
import time
import traceback
from collections.abc import Callable

from acceptance_stack import Stack

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_TOO_MANY_REQUESTS = 429
HTTP_UNAVAILABLE = 503

COMMIT_METRIC = "llm_rayline_arc_episode_transactions_total"
ADMISSION_METRIC = "llm_rayline_arc_encoder_admissions_total"

# The router has no source for these, so it must omit them. An invented value
# reads as measured to whoever joins these rows offline.
UNSOURCED_FIELDS = (
    "bundle_version",
    "pricing_snapshot_version",
    "catalog_provider",
    "policy",
    "route_call_index",
)

SECOND_TURN = 2
CONCURRENT_CONSULTS = 4


def commits(stack: Stack) -> float:
    return stack.metric(COMMIT_METRIC, failure_class="", outcome="commit")


def aborts(stack: Stack, failure_class: str) -> float:
    return stack.metric(COMMIT_METRIC, failure_class=failure_class, outcome="abort")


# --- checks against the uncapped stack ---------------------------------------


def decision_names_a_worker_without_dispatching(stack: Stack) -> None:
    """The endpoint answers with a worker and calls nobody.

    The caller executes the chosen worker itself, so "the provider saw no
    request" is the load-bearing assertion here, not a detail.
    """
    dispatched = len(stack.provider_requests())
    route_id = "rt_0a1b2c3d"

    status, body, _ = stack.route(
        "episode-decision-b", "ARC_ROUTE_B", route_id=route_id
    )
    assert status == HTTP_OK, (status, body)
    assert body["decision_id"] == route_id, body
    assert body["selected_worker"] == "worker-b", body
    assert body["worker_model"] == "synthetic/provider-b", body
    assert isinstance(body["decision_latency_ms"], (int, float)), body
    for unsourced in UNSOURCED_FIELDS:
        assert unsourced not in body, (unsourced, body)

    # The encoder's routing axis is deterministic, so the other marker must
    # reach the other worker. Without this, the assertion above would also
    # pass on a router that always answered "worker-b".
    status, body, _ = stack.route(
        "episode-decision-a", "ARC_ROUTE_A", route_id="rt_0c2d"
    )
    assert status == HTTP_OK, (status, body)
    assert body["selected_worker"] == "worker-a", body
    assert body["worker_model"] == "synthetic/provider-a", body

    assert len(stack.provider_requests()) == dispatched, stack.provider_requests()


def mints_a_decision_id_when_absent(stack: Stack) -> None:
    """No route id is not an error: the router mints one and echoes it."""
    status, body, _ = stack.route("episode-minted", "ARC_ROUTE_A")
    assert status == HTTP_OK, (status, body)
    assert isinstance(body["decision_id"], str) and body["decision_id"], body


def refuses_malformed_consults(stack: Stack) -> None:
    """Malformed consults are refused before any episode turn is reserved."""
    dispatched = len(stack.provider_requests())
    before = commits(stack)
    cases = (
        ("malformed route id", {"route_id": "route-42"}),
        ("empty messages", {"route_id": "rt_0e3f", "messages": []}),
    )
    for name, kwargs in cases:
        status, body, _ = stack.route("episode-malformed", "ARC_ROUTE_A", **kwargs)
        assert status == HTTP_BAD_REQUEST, (name, status, body)
        assert isinstance(body.get("detail"), str) and body["detail"], (name, body)
    assert commits(stack) == before, commits(stack)
    assert len(stack.provider_requests()) == dispatched, stack.provider_requests()


def episode_commit_increments_once_per_consult(stack: Stack) -> None:
    """Exit criterion: the episode transaction commits, once, per decision."""
    before = commits(stack)
    for index in range(SECOND_TURN):
        status, body, _ = stack.route(
            "episode-commit-count", "ARC_ROUTE_A", route_id=f"rt_0c{index}"
        )
        assert status == HTTP_OK, (status, body)
    assert commits(stack) == before + SECOND_TURN, (before, commits(stack))


def previous_arm_is_read_back(stack: Stack) -> None:
    """Exit criterion: turn two reads turn one's arm as PreviousArm.

    A hermetic run has no episode store to peek into, so the proof is in the
    answer. The artifact's stay margin is wider than the encoder's routing
    axis, so an episode that already picked one worker stays on it even when
    the encoder points the other way -- while a fresh episode with that same
    prompt still goes the other way. Only a policy that read the first turn's
    arm can produce both answers.
    """
    status, body, _ = stack.route("episode-stay-a", "ARC_ROUTE_A", route_id="rt_0d1e")
    assert status == HTTP_OK and body["selected_worker"] == "worker-a", body

    status, body, _ = stack.route("episode-stay-a", "ARC_ROUTE_B", route_id="rt_0d2e")
    assert status == HTTP_OK, (status, body)
    assert body["selected_worker"] == "worker-a", (
        "turn two switched workers, so the previous arm was not read",
        body,
    )

    # The control: the same prompt on an episode with no history does switch.
    status, body, _ = stack.route(
        "episode-stay-fresh", "ARC_ROUTE_B", route_id="rt_0d3e"
    )
    assert status == HTTP_OK and body["selected_worker"] == "worker-b", body

    # And the mirror, so the check cannot pass on a router that always stays
    # on worker-a.
    status, body, _ = stack.route("episode-stay-b", "ARC_ROUTE_B", route_id="rt_0d4e")
    assert status == HTTP_OK and body["selected_worker"] == "worker-b", body
    status, body, _ = stack.route("episode-stay-b", "ARC_ROUTE_A", route_id="rt_0d5e")
    assert status == HTTP_OK, (status, body)
    assert body["selected_worker"] == "worker-b", (
        "turn two switched workers, so the previous arm was not read",
        body,
    )


def multi_turn_history_reaches_the_encoder(stack: Stack) -> None:
    """The projected turns satisfy the encoder's own strict wire contract.

    The fake encoder validates the pooling request the way the real plugin
    does: the schema and serializer pins, the serving rung, a lowercase
    SHA-256 episode hash, and a non-empty list of {role, text} objects whose
    role is user or assistant. Anything else is a 400 there and a fail-closed
    503 here, so a multi-turn conversation answering 200 is the projection
    contract holding end to end.

    Byte identity of the projection is pinned separately, by the token-block
    goldens in pkg/selection/raylinearc. This check pins the wire shape.
    """
    stack.request(stack.encoder_port, "/reset", method="POST", body={})
    conversation = [
        {"role": "user", "content": "first question ARC_ROUTE_A"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question ARC_ROUTE_A"},
    ]
    status, body, _ = stack.route(
        "episode-multi-turn", "ARC_ROUTE_A", messages=conversation
    )
    assert status == HTTP_OK, (status, body)
    assert body["selected_worker"] == "worker-a", body
    stats = stack.encoder_stats()
    assert stats["pooling_requests"] == 1, stats


def encoder_failure_fails_closed(stack: Stack) -> None:
    """Exit criterion: an encoder failure answers 503 and dispatches nothing."""
    dispatched = len(stack.provider_requests())
    before_commit = commits(stack)
    before_abort = aborts(stack, "selection_failed")

    status, body, _ = stack.route("episode-encoder-fail", "ARC_ENCODER_FAIL")
    assert status == HTTP_UNAVAILABLE, (status, body)
    assert commits(stack) == before_commit, commits(stack)
    assert aborts(stack, "selection_failed") == before_abort + 1, (
        "a failed selection must abort the episode transaction",
    )
    assert len(stack.provider_requests()) == dispatched, stack.provider_requests()


def same_episode_consults_serialize(stack: Stack) -> None:
    """The episode lease is exclusive: the encoder never sees two of one turn."""
    stack.request(stack.encoder_port, "/reset", method="POST", body={})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda index: stack.route(
                    "episode-serialized", f"ARC_DELAY turn-{index}"
                ),
                range(SECOND_TURN),
            )
        )
    assert all(result[0] == HTTP_OK for result in results), results
    stats = stack.encoder_stats()
    assert stats["max_same_episode"] == 1, stats


# --- checks against the admission-capped stack -------------------------------


def encoder_shed_answers_429_with_retry_after(stack: Stack) -> None:
    """Exit criterion: an admission shed is back-pressure, not an outage.

    503 sends a caller into fallback and reads as an outage. A shed is a
    healthy router at its encoder cap, so it answers 429 and names when to
    come back.
    """
    before_shed = stack.metric(ADMISSION_METRIC, outcome="shed")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_CONSULTS
    ) as executor:
        results = list(
            executor.map(
                lambda index: stack.route(f"episode-shed-{index}", "ARC_DELAY"),
                range(CONCURRENT_CONSULTS),
            )
        )
    statuses = [status for status, _, _ in results]
    shed = [result for result in results if result[0] == HTTP_TOO_MANY_REQUESTS]
    assert HTTP_OK in statuses, statuses
    assert shed, ("the cap of one admitted every concurrent consult", statuses)
    for _status, body, headers in shed:
        assert headers.get("retry-after") == "1", headers
        assert isinstance(body.get("detail"), str) and body["detail"], body
    assert HTTP_UNAVAILABLE not in statuses, (
        "a shed must never read as an outage",
        statuses,
    )
    assert stack.metric(ADMISSION_METRIC, outcome="shed") >= before_shed + len(shed)


UNCAPPED_CHECKS: tuple[Callable[[Stack], None], ...] = (
    decision_names_a_worker_without_dispatching,
    mints_a_decision_id_when_absent,
    refuses_malformed_consults,
    episode_commit_increments_once_per_consult,
    previous_arm_is_read_back,
    multi_turn_history_reaches_the_encoder,
    encoder_failure_fails_closed,
    same_episode_consults_serialize,
)

CAPPED_CHECKS: tuple[Callable[[Stack], None], ...] = (
    encoder_shed_answers_429_with_retry_after,
)


def _run(stack: Stack, checks: tuple[Callable[[Stack], None], ...]) -> list[str]:
    failures: list[str] = []
    for check in checks:
        started = time.monotonic()
        try:
            check(stack)
        except Exception:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}", flush=True)
            traceback.print_exc()
        else:
            elapsed = time.monotonic() - started
            print(f"PASS {check.__name__} ({elapsed:.2f}s)", flush=True)
    return failures


def main() -> int:
    failures: list[str] = []
    with Stack() as stack:
        failures += _run(stack, UNCAPPED_CHECKS)
    with Stack(max_inflight=1) as stack:
        failures += _run(stack, CAPPED_CHECKS)
    total = len(UNCAPPED_CHECKS) + len(CAPPED_CHECKS)
    if failures:
        print(
            f"Rayline ARC acceptance: {total - len(failures)}/{total} passed; "
            f"failed: {', '.join(failures)}",
            flush=True,
        )
        return 1
    print(f"Rayline ARC acceptance: {total}/{total} passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
