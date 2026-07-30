# TD047: Rayline Remote Selection Serializes Across Episodes

## Status

Open

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
calls `RouterService._policy_select()`, however, and that method holds one
process-wide `_policy_select_lock`. A remote MTRouter encoder therefore sends
only one pooling request to vLLM at a time, hiding the scheduler's continuous
batching capacity.

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

## Why It Matters

A single Pathfinder mutex caps remote encoder concurrency at one regardless of
GPU capacity, cache design, or request diversity. Performance results gathered
before fixing this seam would measure lock contention instead of vLLM batching.
Removing the lock globally would be unsafe because not every policy has the
same state or thread-safety contract.

## Desired End State

Declare concurrency capability at the policy implementation boundary.
Immutable MTRouter selection with the remote vLLM encoder may execute
concurrently for different prepared episodes. Same-episode prepares remain
fenced by the transaction journal, and mutable policies remain serialized.
Bounded metrics expose policy and encoder in-flight concurrency and queue time.

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
