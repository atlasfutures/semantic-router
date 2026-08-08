# TD046: Rayline Remote Pending Transactions Are Single-Replica

## Status

Open

## Owner Plan

[PL0040 Rayline Remote Router MVP](../plans/pl-0040-rayline-remote-mvp.md)

## Release Relevance

The experimental `rayline_remote` MVP is complete for one Pathfinder replica.
Production high availability remains blocked on durable pending-transaction
ownership.

## Scope

- Pathfinder's selection-transaction pending journal
- receipt leases and idempotent terminal-result retention
- committed episode-state compare-and-swap
- multi-replica readiness and restart recovery

## Summary

The MVP keeps prepared receipts and per-episode in-flight ownership in a
bounded in-process journal. Committed episode state already uses a versioned
store seam, but another Pathfinder process cannot validate, renew, commit, or
abort a receipt created by the first process. A restart safely loses the
pending prepare without advancing committed state, but availability is limited
until its lease expires and the caller retries.

## Evidence

- Pathfinder advertises
  `pending_journal: bounded_in_process_single_replica` from
  `GET /v1/route/capabilities`.
- Semantic Router readiness requires that exact journal mode and documents a
  one-replica operating boundary.
- The PL0040 integration suites prove idempotency and concurrency within one
  Pathfinder process; they intentionally do not claim cross-replica receipt
  recovery.

## Why It Matters

Load-balancing transactional calls across independent processes can turn a
valid receipt into an unknown receipt. Restarting between prepare and commit
can force a safe but avoidable failed request. Serving more than one replica
without shared fencing would weaken both availability and the exactly-once
state-advance guarantee.

## Desired End State

Provide a durable journal implementation with atomic per-episode ownership,
receipt fencing, lease expiry, bounded terminal-result retention, and
compare-and-swap against committed episode state. Advertise a new versioned
journal capability only after mixed-replica and restart tests prove the same
HTTP contract.

## Exit Criteria

- Prepared receipts can be renewed and completed by any healthy Pathfinder
  replica without duplicate state advance.
- Process termination between prepare, commit, and settle has deterministic
  recovery behavior covered by integration tests.
- Same-episode concurrent prepares serialize through durable atomic ownership.
- Terminal operation idempotency survives restart for the documented retry
  window.
- Semantic Router readiness recognizes the new versioned journal capability.
- The single-replica warning is removed from operator documentation and this
  debt entry is deleted.
