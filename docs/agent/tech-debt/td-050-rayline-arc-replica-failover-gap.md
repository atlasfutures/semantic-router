# TD050: Rayline ARC Replica Failover Is Experiment-Only

## Status

Open

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

Two explicit retained-session encoder replicas improve overloaded routing
throughput, but the current affinity proxy is an experiment harness rather than
a supported discovery, membership, and failover layer.

## Scope

- cache-aware episode affinity across Rayline ARC encoder replicas;
- bounded remap after replica unavailability;
- full-history rebuild cost and exact routing parity;
- session-close fanout across every replica visited by an episode; and
- membership, health, observability, and rollout ownership.

Shared KV storage and Pathfinder's pending-transaction journal are out of scope.
The latter remains tracked by TD046.

## Summary

PERF024 passed a same-proxy comparison between one and two explicit H100
encoders. Deterministic episode affinity kept every four-turn episode on one
process-local KV owner and improved completion throughput by `1.1442x` at the
frozen `r030` cell and `1.3990x` at `r045`. The implementation deliberately has
no production service directory, health-based membership, rebalance, or
failover contract.

Every retained-session request does carry full reconstructible history, so a
new replica can recreate a missing session without shared KV. PERF025 added
only an experiment-side forced-remap path to quantify that rebuild and verify
close fanout. It completed all correctness and cleanup gates, but an independent
audit found arm-specific episode-hash namespaces, so its performance ratios are
confounded and retained only as diagnostic evidence. PERF026 explicitly shares
the hash namespace and adds a strict primary-placement vector gate before any
performance comparison. Neither experiment makes the proxy a supported
deployment surface.

## Evidence

- PERF024's private aggregate packet is pinned at
  `rayline-ai/router-artifacts@cd832e8da7fc8dba9f6518f65b613c9afb271978`.
- PERF025 completed 64/64 measured turns, nine peer reconstructions, 18 fanout
  closes, exact trace parity, and stable-zero cleanup. Its latency, throughput,
  backlog, drain, and token ratios are inadmissible because the two arms used
  different hash namespaces; the private aggregate packet and audit receipt
  remain the diagnostic system of record.
- `e2e/testing/rayline-arc/rayline_affinity_proxy.py` owns deterministic
  experiment placement and aggregate-only accounting.
- `SessionCoordinator` rebuilds from full supplied history when a retained
  prefix is absent or divergent.
- The single-container development qualification already proves an explicit
  affinity-loss rebuild has cosine similarity above the frozen `0.9999` gate;
  it does not prove cross-replica routing or cleanup.

## Why It Matters

Process-local KV makes affinity a performance property, while full-history
rebuild preserves correctness. Without a supported remap and close contract, a
replica outage can either fail a routing decision or leave duplicate retained
state after a retry. Treating the experiment proxy as production-ready would
also hide membership and rollout behavior that materially affects latency and
capacity.

## Desired End State

Provide a supported cache-aware service layer that assigns each episode to one
healthy encoder, remaps only under a documented failure policy, reconstructs
from the full request history, and closes state on every visited owner. Expose
privacy-safe affinity, rebuild, retry, and cleanup metrics, and document how
membership changes interact with in-flight requests and deployments.

## Exit Criteria

- A versioned production configuration selects and discovers multiple Rayline
  ARC encoder replicas without experiment-only process wiring.
- Sticky traffic preserves per-episode KV locality under normal operation.
- A bounded replica failure remaps the request, preserves the selected worker,
  and reports the full-history rebuild cost.
- Ambiguous transport failure has a documented retry/idempotency boundary.
- Session close reaches every replica that may retain the episode, including
  after a remap, and stable-zero cleanup is integration-tested.
- Membership changes, readiness, and rolling replacement have deterministic
  behavior and aggregate metrics.
- Operator documentation removes the experiment-only warning and this debt
  entry is deleted.
