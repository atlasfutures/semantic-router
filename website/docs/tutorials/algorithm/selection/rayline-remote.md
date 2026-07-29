# Rayline Remote

## Overview

`rayline_remote` is an experimental, fail-closed selection algorithm that asks
a Pathfinder Rayline service to choose one worker from a decision's explicit
candidate set. Semantic Router remains the request data plane and provider
credential owner. Pathfinder remains the policy and episode-state authority.

It aligns to `config/algorithm/selection/rayline-remote.yaml`.

This mode is distinct from `rayline_arc`. ARC executes an immutable artifact
inside Semantic Router and owns its episode store locally. Remote mode keeps
policy and state in Pathfinder and carries only an opaque request-scoped
transaction receipt through provider dispatch.

## Key Advantages

- Keeps policy and episode state in one independently deployable authority.
- Restricts every decision to Semantic Router's explicit worker allowlist.
- Commits state at the observable provider-success boundary, not at selection.
- Keeps provider credentials and transport behavior inside Semantic Router.
- Pins the wire protocol, policy bundle, and worker catalog at readiness.

## What Problem Does It Solve?

Stateful routing policies need multi-turn context, switch costs, and observed
provider outcomes, but the request proxy must still control dispatch and
credentials. The transaction protocol bridges those ownership domains without
duplicating episode state or advancing it for requests that never reached a
successful provider.

## When to Use

Use remote mode when Pathfinder is the authoritative Rayline policy service
and its worker catalog can be mapped exactly to one decision's provider-backed
models. Use a local selector for independent requests, and use `rayline_arc`
when the desired policy is a frozen artifact intentionally executed inside
Semantic Router.

## Request Lifecycle

For each matched OpenAI Chat Completions request:

1. Semantic Router sends Pathfinder a stable decision ID, an HMAC-derived
   episode key, the pinned bundle version, the request, and only the configured
   worker allowlist.
2. Pathfinder prepares a selection and returns one allowed worker plus an
   opaque leased receipt. Preparing does not advance committed episode state.
3. Semantic Router validates or renews the receipt immediately before dispatch
   and maps the worker to its configured `modelRef`.
4. First upstream 2xx headers commit the selection exactly once. Any terminal
   path before 2xx headers aborts it without advancing episode state.
5. Response completion settles bounded status, token, cost, and latency facts
   when available. Settlement cannot change the committed worker.

Remote selection, receipt validation, worker mapping, or lifecycle failures
fail closed; there is no first-candidate fallback.

## Contract

- `on_error` must be `fail_closed`.
- `adaptations.mode` must be `bypass`, and Router Replay must be disabled for
  the decision, so there is exactly one episode-state owner.
- `bundle_version` must be immutable and must match Pathfinder readiness.
- `workers` must map the decision's complete `modelRefs` set one-to-one.
- `api_key_env` and `episode_hmac_key_env` name distinct environment
  variables. Secret values are never serialized into canonical configuration.
- `episode_id_header` and `decision_id_header` are distinct lowercase HTTP
  field names. The raw episode value is never sent to Pathfinder.
- The MVP supports OpenAI Chat Completions only, including tools and streaming.
- Pathfinder cannot provide provider credentials, arbitrary headers, or
  arbitrary request mutations.

The MVP Pathfinder pending journal is bounded and in-process. Run one
Pathfinder replica until a durable multi-replica journal is implemented.

## Configuration

```yaml
routing:
  decisions:
    - name: rayline-remote-route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: model-a
          use_reasoning: false
        - model: model-b
          use_reasoning: true
      adaptations:
        mode: bypass
      plugins:
        - type: router_replay
          configuration:
            enabled: false
      algorithm:
        type: rayline_remote
        on_error: fail_closed
        rayline_remote:
          base_url: http://rayline-router:8000
          bundle_version: mtrouter-example-immutable-revision
          api_key_env: RAYLINE_API_KEY
          episode_id_header: x-rayline-episode-id
          episode_hmac_key_env: RAYLINE_EPISODE_HMAC_KEY
          decision_id_header: x-rayline-route-id
          connect_timeout_ms: 250
          request_timeout_ms: 1000
          lease_ttl_seconds: 30
          max_retries: 1
          workers:
            - id: mock-a
              model: model-a
            - id: mock-b
              model: model-b
```

Set the two referenced environment variables in the Semantic Router process.
Treat the episode header as an authenticated caller contract: reject requests
that omit it, and do not place user content or raw identifiers in the decision
ID.

## Readiness

Startup and reload readiness validate:

- the selection-transaction protocol version and required operations;
- OpenAI Chat Completions protocol support;
- the exact immutable bundle;
- the worker catalog and worker-to-provider dispatch map; and
- the configured request, connection, retry, and lease budgets.

If Pathfinder is unavailable or any catalog pin drifts, the affected decision
must not admit traffic.

Pathfinder's liveness endpoint is not enough. Before admitting traffic, verify
that its authenticated `GET /v1/route/capabilities` and `GET /v1/workers`
responses name the configured transaction schema, immutable bundle, lease
duration, pending-journal mode, and dispatch identities. Semantic Router
performs these checks at construction time and publishes
`llm_rayline_remote_ready` as `1` only when they pass.

## Local Acceptance

Build the normal local Semantic Router image and run the hermetic stack:

```bash
make vllm-sr-build
make rayline-remote-test-integration
```

The suite starts Envoy, Semantic Router, a contract-faithful Rayline fixture,
and two fake OpenAI providers. It exercises successful multi-turn routing,
candidate masking, malformed responses, timeouts, lease loss, provider
failure, streaming, idempotency, concurrency, settlement, and privacy.

To validate the same wire contract against the actual Pathfinder source
checkout, run its cross-repository receipt:

```bash
VLLM_SEMANTIC_ROUTER_ROOT=/path/to/semantic-router \
  bash tests/integration/vsr_remote/run.sh
```

This second command is run from the Pathfinder repository. Both suites use
fake providers and incur no model or GPU cost.

## Failure Behavior

| Failure | Client/data-plane behavior | Episode-state behavior |
| --- | --- | --- |
| Missing episode header | Typed 503; no provider dispatch | No prepare or advance |
| Pathfinder timeout or malformed response | Typed 503; no provider dispatch | Prepared receipt expires or is aborted; no advance |
| Unknown worker or dispatch-contract drift | Startup not ready, or typed 503 before dispatch | No advance |
| Lease renewal failure | Typed 503 before provider dispatch | No advance |
| Provider transport error or non-2xx | Provider error is returned | Receipt aborts; retry sees the same turn |
| Commit failure on 2xx headers | Typed 503 replaces the provider response | Fail closed; no unacknowledged response is released |
| Body failure after a committed 2xx | Stream may terminate or surface a gateway error | Commit remains; bounded settlement is attempted |
| Settlement failure | Successful provider response is preserved | Commit remains; failure is observable and never re-routes |

There is deliberately no automatic first-model fallback. Recover availability
by restoring the pinned Pathfinder contract or by deploying a reviewed config
that uses a different selection algorithm.

## Observability

The remote path exposes bounded, privacy-safe Prometheus series:

- `llm_rayline_remote_ready`
- `llm_rayline_remote_selections_total{candidate_index}`
- `llm_rayline_remote_selection_latency_seconds`
- `llm_rayline_remote_failures_total{stage,class}`
- `llm_rayline_remote_transactions_total{operation,outcome}`

Logs use the same bounded stage/class vocabulary plus candidate ordinal and
bundle hash. Do not add raw worker IDs, episode IDs, prompts, tools, receipts,
authorization headers, or credential values as labels or log fields.

Alert on readiness dropping to zero, sustained prepare/renew/commit failures,
and any divergence between commit and settle counts. Settlement is
post-response observation, so a settle failure is operationally important but
must not trigger a second provider call.

## Rollback

`rayline_remote` is authoritative and fail closed; rollback is a configuration
deployment, not an in-request fallback.

1. Stop admitting new traffic to the affected Semantic Router revision and
   allow active leases to drain.
2. Restore the last known-good config and Pathfinder bundle together, or
   replace the decision's algorithm with its reviewed pre-remote selector.
3. Keep `adaptations.mode: bypass` and Router Replay disabled while any remote
   decision remains.
4. Restart through the normal local/deployment image flow and require
   readiness plus a synthetic request before restoring traffic.
5. Do not run multiple Pathfinder replicas for this MVP. Its pending journal
   is process-local; a restart discards unresolved prepares, which safely
   expire without advancing committed episode state.

## Status

This algorithm is experimental. The first release is an end-to-end MVP with a
single Pathfinder replica, a bounded in-process pending journal, and
deterministic protocol fixtures. Durable multi-replica pending transactions
remain a production follow-up.
