# Rayline ARC Retained-Encoder Replica Contract

Status: implemented as the static production contract on 2026-08-03. Dynamic
service discovery and automatic membership control remain out of scope.

## Boundary

This contract applies to Semantic Router's native `rayline_arc` selector. It
does not move Pathfinder's `rayline_remote` transaction authority into
Semantic Router and it does not make worker-generation replicas part of the
encoder pool.

```text
 client
   |
   | episode ID + complete reconstructible history
   v
 Envoy -> Semantic Router
            |  (1) Redis lease + episode-state.v2
            |      policy arm history
            |      encoder owner + visited owners
            |
            |  (2) rendezvous affinity / one bounded remap
            v
       +---------------- retained encoder pool ----------------+
       | encoder-a (active/draining)   encoder-b (active)      |
       | process-local vLLM KV + causal-MEAN accumulator       |
       +-------------------------------------------------------+
            |
            | embedding + bounded replica result
            v
       immutable ARC head -> selected generation worker/provider
            |
            | upstream 2xx
            v
       fenced Redis commit; optional close fanout to all visited owners
```

KV remains reconstructible acceleration state. Every encoder call carries the
complete canonical turn history. Losing affinity can add token work and
latency, but it must not change the policy input or selected worker.

## Configuration Modes

`encoder.base_url` preserves the existing single-service mode. Replicated mode
sets `encoder.replicas` instead and opts into the exact
`rayline.arc.encoder-failover.v1` contract. The two forms are mutually
exclusive.

Replicated mode requires:

- two to eight stable, unique replica IDs and URLs;
- at least one `active` member;
- Rung B with both `chunked_causal_mean` and
  `resumable_causal_mean`;
- `max_retries: 0` inside each encoder client;
- exactly one configured remap after an explicitly listed HTTP status;
- a positive passive-health cooldown; and
- an exact `episode.close_header` final-turn signal.

All configured members, including `draining`, must pass the full startup probe
and readiness-session close. A draining replica may own existing episodes but
receives no new rendezvous assignments.

## Request Interaction

1. Semantic Router hashes the configured episode header and acquires the
   fenced Redis lease.
2. Episode state v2 supplies the previous encoder owner and every owner that
   may retain the session. State v1 is accepted and migrates with empty
   affinity on its next successful commit.
3. An existing configured owner remains sticky even while draining. A new
   episode uses deterministic highest-score rendezvous hashing over active,
   locally healthy members.
4. The selected encoder receives full history. A success returns the replica
   ordinal, attempt count, whether a remap occurred, and the updated visited
   set. Raw stable IDs remain internal transaction data.
5. Semantic Router runs the immutable ARC head and dispatches the selected
   generation worker.
6. Only upstream 2xx commits policy state and encoder affinity. Any earlier
   failure aborts the Redis transition.

## Failure Semantics

| Observation | Action | Why |
| --- | --- | --- |
| Configured unavailable HTTP status | Mark the member locally unavailable for the cooldown and remap once to another active member | The deployment operator has declared these responses safe evidence that no retained mutation should be trusted there |
| Second replica failure | Fail the selection closed | The one-remap budget is exhausted |
| Timeout or transport failure | Fail closed with no peer call | The first member may already have mutated retained state |
| Decode, identity, or contract failure | Fail closed with no peer call | A peer must not hide protocol or deployment drift |
| Persisted owner absent from membership | Fail closed before any encoder call | Removal occurred before its state-drain boundary |

Configured HTTP statuses are an operator assertion, not a generic retry list.
Only include statuses whose deployment path guarantees an unavailable or
pre-mutation response. Provider-generation retry policy remains a separate
Envoy concern.

Passive health is deliberately process-local. A different Semantic Router
replica may make one additional explicit-status detection, but Redis affinity
and the one-remap rule keep correctness shared without creating a new health
consensus system.

## Close and Rolling Membership

The configured close header accepts only empty/`false` or exact `true`.
Following a successful provider response, Semantic Router concurrently sends
the retained-session DELETE to every visited configured owner. Confirmed
deletes and configured unavailable responses count as cleanup success. Clean
fanout clears encoder affinity while preserving ARC policy history. Partial
cleanup is logged and measured, retains the visited set for a later final
turn, and never replaces an already successful provider response.

Rolling removal is a two-step operator protocol:

1. change `active` to `draining`, roll/reload Semantic Router, and keep the
   encoder available for existing owners;
2. wait at least the configured episode `idle_ttl_seconds` after the last
   admitted owner (and verify close/session metrics), then remove the member.

Adding a member as `active` changes only new-episode rendezvous placement.
Removing an owner early is intentionally a fail-closed configuration error,
not an implicit rebuild.

## Aggregate Observability

The pool exports no endpoint, stable replica ID, prompt, embedding, or raw
episode labels. It adds:

- `llm_rayline_arc_encoder_replica_routes_total{outcome="direct|failover"}`;
- `llm_rayline_arc_encoder_replica_attempts`;
- `llm_rayline_arc_encoder_session_closes_total{outcome="closed|unavailable|failed"}`;
- bounded log fields for replica ordinal, attempt count, failover, and close
  counts.

The existing encoder token/session-action histograms expose reconstruction
work without per-request payloads.

## Verification

Focused Go tests cover deterministic affinity, active/draining behavior,
configured-status remap, cached cooldown, ambiguous transport fail-closed,
premature removal, v1-to-v2 state migration, transactional commit, and close
fanout. `make rayline-arc-test-integration` runs two independent fake encoders
through real Envoy, Semantic Router, and Redis and verifies explicit 503
failover, survivor stickiness, cooldown recovery, router restart, final-turn
fanout, stable-zero resident sessions, metrics, Redis loss, and log privacy.

PERF024, PERF026, and PERF027 remain the measured GPU evidence behind this
contract. They do not justify dynamic discovery or another paid expansion.
