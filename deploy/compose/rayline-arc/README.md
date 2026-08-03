# Rayline ARC Provider Execution Contract

This compose stack keeps one Rayline decision and transaction around every
logical provider request. Semantic Router selects the worker, shapes the
request, and prepares the episode transaction. Envoy owns the upstream HTTP
attempts below that decision.

## Retry ownership

The OpenRouter topology in `envoy-openrouter.yaml` retries only HTTP 429 and
503 responses. The default hermetic topology in `envoy.yaml` mirrors that
contract. The self-hosted vLLM topology in `envoy-real-workers.yaml` has no
retry policy: a local worker's 429 is admission/backpressure and automatic
replay could amplify overload.

The signed worker manifest supplies `openrouter_max_retries` and
`attempt_deadline_seconds`. Semantic Router removes caller-supplied Envoy retry
controls, then writes the artifact-owned maximum and overall request timeout.
The current v1 OpenRouter route fixes exponential backoff at 2 seconds with a
30-second cap, matching the immutable fixtures in this directory.

Envoy honors an integer-seconds `Retry-After` value when it is within the
30-second rate-limit interval. An absent, malformed, or oversized value falls
back to the bounded exponential policy. The route timeout and artifact-owned
overall request timeout include all attempts and backoff.

`retriable_request_headers` is deliberately not used as the backend selector.
Rayline chooses the backend after Envoy's request-header phase, while that
matcher is evaluated from the earlier header view. Deployment topology is the
authoritative boundary: OpenRouter routes retry; self-hosted vLLM routes do
not. A future mixed topology must route on an explicit trusted dispatch signal
and attach the retry policy only to its OpenRouter route.

## Transaction and streaming boundary

A retry stays inside Envoy, so it reuses the selected worker, rewritten body,
credential, Rayline episode, and prepared transaction. Semantic Router sees
one final upstream response and therefore commits or aborts once. Envoy exposes
the final `x-envoy-attempt-count` to Semantic Router and the downstream test
client.

Only 429 and 503 response headers are retriable. After an upstream HTTP 200 is
accepted— including when an SSE body has started—no configured retry trigger
remains. Partial response delivery is never replayed.

## Observability

The router exports aggregate-only counters; none includes prompts, responses,
episode IDs, model payloads, or credentials:

- `llm_rayline_arc_provider_logical_requests_total{outcome}`
- `llm_rayline_arc_provider_attempts_total{outcome}`
- `llm_rayline_arc_provider_retries_total{outcome}`
- `llm_rayline_arc_provider_retry_exhaustions_total{status}`
- `llm_rayline_arc_encoder_replica_routes_total{outcome}`
- `llm_rayline_arc_encoder_replica_attempts`
- `llm_rayline_arc_encoder_session_closes_total{outcome}`

Run the hermetic contract with:

```bash
e2e/testing/rayline-arc/run.sh
```

It covers 429-to-200, 503-to-200, streaming 429-to-200, exhausted 429 and 503,
single commit/abort behavior, post-200 partial streaming, retained-encoder
replicas, explicit-status remap, survivor stickiness, cooldown recovery,
dynamic active-to-draining and controller-result removal snapshots, close
fanout, process restart, Redis loss, aggregate attempt metrics, and privacy-log
scanning.

The compose profile seeds the reviewed Redis
`rayline.arc.encoder-membership.v1` source before router startup and retains
the `rayline.arc.encoder-failover.v1` request contract. Replica IDs name
concrete retained state owners, not load balancers. New episodes rendezvous
across active members; persisted owners remain sticky; transport ambiguity
fails closed; and the exact final-turn header fans DELETE out to all visited
owners.

## Membership controller

The stack builds a separate `rayline-arc-controller` image and runs its
continuous reconciler as a non-root sidecar. The router keeps only the
read-only membership seam; membership CAS writes and the controller Redis
credential stay in the controller process.

The sidecar reads the same mounted canonical config, including the episode key
prefix and idle TTL. These commands reuse its environment and config:

```bash
docker compose -f deploy/compose/rayline-arc/compose.yaml \
  run --rm --no-deps membership-controller status
docker compose -f deploy/compose/rayline-arc/compose.yaml \
  run --rm --no-deps membership-controller register \
  --replica-id encoder-c --base-url http://encoder-c:8000
docker compose -f deploy/compose/rayline-arc/compose.yaml \
  run --rm --no-deps membership-controller drain --replica-id encoder-a
docker compose -f deploy/compose/rayline-arc/compose.yaml \
  run --rm --no-deps membership-controller reconcile
```

`RAYLINE_ARC_CONTROLLER_PASSWORD_ENV` names the controller-only Redis password
environment variable; `RAYLINE_ARC_CONTROLLER_REDIS_USERNAME` optionally
selects its named Redis ACL user. The hermetic stack uses the same default-user
test password in two isolated containers. A real Redis ACL deployment should
deny the router credential writes to the exact membership key while preserving
its episode-state writes, and give the controller user membership CAS plus
read/scan access to the bounded episode-state keys.

The integration suite invokes the real `register` command to add encoder C as
revision 2. The router probes that new endpoint before adopting the snapshot.
The suite then invokes `drain` for revision 3 and waits for the running
reconciler to prove the idle-window zero-reference condition and publish
removal revision 4.

## Bounded performance diagnostic

`run_modal_fullstack.py --mode diagnostic` compares three paths against the
same two real vLLM workers and the same frozen prompt, token, temperature,
thinking, and seed fields:

1. direct worker access;
2. a specified-model request through Envoy and Semantic Router that skips
   automatic selection; and
3. the full `model: auto` Rayline ARC path.

The diagnostic runs two waves at concurrency one and four. It contains 30
measured generations and allows at most 62 generations including direct
warmup and prompt coverage. It has no case-count or paid-provider flag, makes
zero external-provider calls, and cannot invoke the held 1,000-case packet.

Each wave snapshots vLLM success, token, TTFT, end-to-end, queue, preemption,
running, waiting, and KV-utilization metrics. Router snapshots separate the
general routing histogram from the ARC encoder histogram. Gateway responses
must contain one Envoy attempt and `x-envoy-upstream-service-time`; the report
also shows client latency minus that header as an explicitly approximate
gateway residual. Completion-token totals must match across all three paths
for each worker, or the diagnostic fails instead of publishing a confounded
throughput comparison.
