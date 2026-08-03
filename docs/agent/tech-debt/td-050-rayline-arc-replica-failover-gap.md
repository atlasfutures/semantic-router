# TD050: Rayline ARC Dynamic Membership Remains Manual

## Status

Open — DYN001-DYN005 and the DYN006 source-closed implementation are complete.
The controller-driven hermetic full-stack acceptance passes, including active
capacity registration. The preregistered three-H100 live drain/stop cell and
fleet provisioning automation remain open.

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

Semantic Router now has a supported static contract plus an optional versioned
Redis membership source and standalone controller. The controller can register
reviewed capacity, but fleet provisioning remains an external operator
responsibility and the topology-changing live gate is not yet evidence.

## Scope

- optional dynamic discovery beyond the implemented static replica set;
- controller-owned active/draining publication and drain completion;
- automated removal only after the episode idle-TTL boundary; and
- preservation of the implemented affinity, health, close, privacy, and
  observability semantics during dynamic rollout.

Shared KV storage and Pathfinder's pending-transaction journal are out of
scope. The latter remains tracked by TD046.

## Summary

PERF024 proved that deterministic episode affinity across two explicit H100
encoders preserved process-local KV ownership and improved overloaded routing
throughput. PERF026 corrected the cross-arm identity of the forced-remap
experiment and quantified full-history reconstruction. PERF027 then stopped a
real exact Modal app, detected affected primaries, rebuilt them on the
survivor, fanned close out, and measured the one-survivor capacity penalty.
Those experiments did not promote their Python affinity proxy.

The subsequent zero-provider implementation phase moved the contract into
Semantic Router itself. `rayline.arc.encoder-failover.v1` now validates two to
eight stable active/draining members, deterministic rendezvous affinity, one
remap only after configured HTTP unavailability, process-local passive
cooldown, transport-ambiguity fail-closed, Redis-persisted owner/visited state,
startup probe coverage for draining owners, low-cardinality metrics, and
explicit post-2xx close fanout. Episode-state v2 reads v1 and migrates on the
next successful commit. Removing a persisted owner early fails before any
encoder call, making the documented active -> draining -> idle-TTL -> remove
sequence enforceable rather than implicit.

## Evidence

- PERF024's private aggregate packet is pinned at
  `rayline-ai/router-artifacts@cd832e8da7fc8dba9f6518f65b613c9afb271978`.
- PERF026 passed 64/64 measured turns, exact `[7,2]` cross-arm primary
  placement, nine peer reconstructions, 18 fanout closes, and stable-zero
  cleanup. Forced remap used `1.0575x` appended-token work and `1.1371x` p50
  latency in that bounded sample.
- PERF027 passed a real exact-app stop. It detected four affected primaries,
  rebuilt four sessions through eight failover pooling calls, closed all eight
  measured sessions, and reached stable-zero retained and deployment state.
  The stop converged in `10.909s`; the survivor delivered `0.5929x` control
  throughput, `3.1924x` p50 service latency, and `2.6551x` p95 service latency.
  Its ten aggregate-only files are byte-for-byte verified at
  `rayline-ai/router-artifacts@2c38ad5760961b04f80c4d2c9d5c1bd85c78ae41`.
- Production Go tests and the two-encoder compose profile pass deterministic
  affinity, configured-503 failover, survivor stickiness, cooldown recovery,
  router restart, concurrent close fanout, stable-zero resident sessions,
  Redis-loss fail-closed, aggregate metrics, and log privacy. This phase used
  no Modal GPU or provider requests and cost `$0`.
- The standalone `rayline-arc-controller` image runs non-root, accepts
  `status`, idempotent `register`, `drain`, `reconcile`, and continuous `run`,
  reads Redis topology and TTLs from the same canonical router config, and can
  receive a controller-only password environment override. A router probes
  newly registered capacity before snapshot adoption. The Compose acceptance
  now creates register revision 2, drain revision 3, and removal revision 4
  through this process rather than publishing those states from Python.
- The source-closed DYN006 harness freezes three H100 encoder apps, A/B initial
  membership, controller registration of C, a five-minute idle boundary, a
  treatment-only drain and exact stop of A, `[2,3,3]` pre-boundary placement,
  `[0,4,4]` post-stop placement, 32 post-boundary measured decisions, strict
  aggregate lifecycle/telemetry gates, and a `$12` packet ceiling. It is not
  live evidence until its distinct registry authorization is pushed and the
  one-shot run completes.
- The implemented contract is documented in
  [Rayline ARC Retained-Encoder Replica Contract](../../architecture/rayline-arc-replica-membership.md).

## Why It Matters

The static remap and close contract now covers bounded deployments without an
experiment proxy, and the standalone controller automates safe drain removal.
A larger installation still needs fleet ownership for adding reviewed
capacity and invoking drain as replicas scale down. Treating process discovery
as automatic capacity management would hide rollout behavior that materially
affects latency and capacity.

## Desired End State

Preserve the supported static contract while adding an optional versioned
membership provider/controller that can publish reviewed active/draining sets,
observe aggregate drain completion, and remove a member only after the episode
idle-TTL boundary. Dynamic membership must retain the same affinity, ambiguous
failure, privacy, and close semantics.

## Exit Criteria

- [x] A versioned static production configuration selects multiple Rayline ARC
  encoder replicas without experiment-only process wiring.
- [x] Sticky traffic preserves per-episode KV locality under normal operation.
- [x] A bounded configured-status failure remaps once and reports aggregate
  rebuild/session work.
- [x] Ambiguous transport failure has a documented fail-closed boundary.
- [x] Session close reaches every visited configured owner and stable-zero
  cleanup is integration-tested.
- [x] Readiness and active/draining rolling replacement have deterministic
  static behavior and aggregate metrics.
- [x] A reviewed dynamic membership source and controller can prove drain
  completion before removal without weakening the v1 request contract.
- [x] A standalone non-root operator command/image drives status, drain, and
  reconciliation through the reviewed source and is covered end to end.
- [x] The controller idempotently registers a new active identity, while each
  router probes the new endpoint before adopting the successor snapshot.
- [ ] The preregistered DYN006 three-encoder cell passes controller registration,
  drain-before-stop, balanced survivor capacity, idle-TTL removal, aggregate
  cleanup, and its performance gate.
- [ ] Fleet automation provisions capacity and invokes those controller
  transitions; then this debt entry is deleted.
