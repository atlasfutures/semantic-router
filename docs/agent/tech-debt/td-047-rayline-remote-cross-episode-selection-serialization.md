# TD047: Rayline Remote Selection Serializes Across Episodes

## Status

Open — the serialization fix and tests are landed; a real-stack concurrency
receipt is still required.

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

Router-only capacity qualification and vLLM continuous-batching evidence are
blocked while every transactional policy selection passes through one
process-wide lock.

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

## Why It Matters

A single Pathfinder mutex caps remote encoder concurrency at one regardless of
GPU capacity, cache design, or request diversity. Performance results gathered
before fixing this seam would measure lock contention instead of vLLM batching.
Removing the lock globally would be unsafe because not every policy has the
same state or thread-safety contract.

## Desired End State

The implementation now declares concurrency capability at the policy boundary.
The remaining desired state is a router-only receipt showing independent
requests reach the real encoder/vLLM boundary concurrently, with policy and
encoder queue/in-flight telemetry sufficient to distinguish Pathfinder
contention from vLLM saturation.

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
- The frozen throughput ladder can drive more than one pooling request into
  vLLM, and this debt entry is deleted.
