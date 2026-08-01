# TD047: Rayline Remote Selection Serializes Across Episodes

## Status

Open — the serialization fix and tests are landed; the first real-stack
receipt attempts exposed a retained-session metrics gap before concurrency
could be measured.

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

Router-only capacity qualification and vLLM continuous-batching evidence still
need a real-stack receipt. The process-wide lock is fixed; append-scoped
retained-session telemetry is now the immediate evidence blocker.

## Scope

- Pathfinder's transaction-aware policy selection path
- per-policy concurrency and thread-safety declarations
- MTRouter remote encoder concurrency across different episodes
- same-episode and mutable-policy fencing
- concurrency metrics and deterministic tests

The legacy eager route's one-thread `AsyncStateCoordinator` segment is related
capacity debt but is not the immediate `/v1/route/prepare` limiter covered by
this entry.

## Summary

Pathfinder's transaction coordinator already fences a second prepare for the
same episode and releases its journal lock while a selector runs, so different
episodes could select concurrently. The HTTP transaction adapter ultimately
called `RouterService._policy_select()` under one process-wide lock. That
implementation blocker is now removed: a default-serialized
`PolicySelectionExecutor` permits overlap only when a policy explicitly
declares `concurrent_selection_safe = True`; remote MTRouter opts in and local
MTRouter remains serialized because it may mutate KV sessions.

## Evidence

- Pathfinder `selection_transaction_http.py` invokes
  `policy_runtime._policy_select()` from the prepare selector.
- Pathfinder `serving/app.py` wraps `_policy_select()` with
  `_policy_select_lock`, shared by all policies and episodes.
- Pathfinder `selection_transactions.py` marks an episode active under its
  journal lock, releases that lock before invoking the selector, and rejects a
  concurrent prepare only when it targets the same episode.
- The stateless `VLLMMTRouterEncoder` client is safe to issue independent HTTP
  requests, while other policies may own mutable RNG or round-robin state and
  cannot be assumed concurrent-safe.
- [`atlasfutures/pathfinder@ce661e5f`](https://github.com/atlasfutures/pathfinder/commit/ce661e5ffe62301dcad307b9bc4b242324019497)
  adds the capability boundary, bounded `/readyz` metrics, and deterministic
  tests for different-episode overlap, same-episode fencing, mutable-policy
  serialization, and failure cleanup.
- Semantic Router `f9e32269` adds payload-free coordinator concurrency,
  contention, backend, and token-work counters plus a curated vLLM registry
  view. `83782ab9` makes driver failures observable and permits a bounded
  completed-metric settlement window.
- `rayline-arc-encoder-service-perf003-20260801` stopped after one successful
  append because its launcher suppressed the captured driver failure; it was
  closed without retry and cleanup returned the exact encoder inventory to
  zero.
- `rayline-arc-encoder-service-perf004-20260801` proved the underlying gap:
  after one retained append and explicit close both returned HTTP 200, vLLM's
  queue, inference, end-to-end, and prompt-token completed-request histogram
  deltas stayed `0/0/0/0` for ten seconds. Its private aggregate receipt is
  pinned at
  `rayline-ai/router-artifacts@28a3f5cf5b82a20f7b6f93f245d825a70e7f5685`.
- The same live logs show warmed full-registry metric requests taking roughly
  `165-230ms`, so the proposed `20ms` sampler would perturb the system and
  cannot observe at its requested cadence.

## Why It Matters

A single Pathfinder mutex caps remote encoder concurrency at one regardless of
GPU capacity, cache design, or request diversity. Performance results gathered
before fixing this seam would measure lock contention instead of vLLM batching.
Removing the lock globally would be unsafe because not every policy has the
same state or thread-safety contract.

## Desired End State

The implementation now declares concurrency capability at the policy boundary.
The remaining desired state is append-scoped retained telemetry inside the
vLLM seam, exposed through a cached aggregate snapshot, followed by a
router-only receipt showing independent requests reach the real encoder/vLLM
boundary concurrently. The evidence must distinguish Pathfinder contention,
same-session fencing, service admission, and vLLM scheduling without treating
terminal session completion as an append.

## Exit Criteria

- A blocking fake remote encoder proves that at least two different-episode
  prepares overlap.
- A same-episode concurrent prepare remains rejected or serialized according
  to the transaction contract.
- Mutable RNG, round-robin, and any undeclared policies retain their current
  serialized behavior.
- Cancellation, timeout, readiness failure, and shutdown release concurrency
  resources without leaking an active episode.
- Router-only receipts record policy/encoder queue time and observed in-flight
  concurrency.
- Retained append queue/inference observations advance exactly once per append;
  session-total metrics remain separately named, and no full Prometheus
  registry collection runs on the hot sampling path.
- The frozen throughput ladder can drive more than one pooling request into
  vLLM, and this debt entry is deleted.
