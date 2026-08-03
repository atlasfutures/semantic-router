# Rayline ARC

## Overview

`rayline_arc` is an experimental, artifact-verified selection algorithm for a
switch-aware orchestrator. A dedicated vLLM pooling deployment performs the
encoder inference; Semantic Router verifies and executes the small F32 policy
head, applies cache-loss and stay-margin policy, and owns transactional episode
state.

It aligns to `config/algorithm/selection/rayline-arc.yaml`.

The public implementation is schema-generic. Arm identities, provider/model
bindings, prices, and private goldens come only from an immutable mounted
runtime artifact and must not be copied into source configuration.

## Key Advantages

- Keeps the SLM encoder on vLLM while the router owns deterministic policy.
- Verifies artifact hashes, tensor shapes, arm order, and startup goldens.
- Prices switches using the artifact's immutable cache-aware cost snapshot.
- Serializes same-episode decisions and commits state only after successful
  upstream response headers.
- Fails closed on artifact, encoder, policy, or state errors.

## What Problem Does It Solve?

Ordinary prompt routers select each request independently. ARC routes a
multi-turn episode with a learned history representation while accounting for
the quality advantage of another arm, the cost of losing a warm provider KV
cache, and a stable stay margin. The split architecture keeps heavy encoder
inference in vLLM and keeps the small auditable decision policy in Semantic
Router.

## When to Use

Use ARC only for a frozen, compatible `rayline.mtrouter-runtime.v3` artifact
whose logical arms can be mapped exactly to configured providers. Keep it
experimental until CPU, Redis, CUDA, dispatch, privacy, and full-stack gates
pass for the exact artifact and vLLM build. Other selectors are a better fit
when requests are independent or an immutable orchestrator artifact is not
available.

## Contract

ARC is deliberately stricter than other selectors:

- `on_error` must be `fail_closed`; selection errors never choose the first
  candidate.
- `adaptations.mode` must be `bypass`; Router Learning cannot replace the ARC
  decision.
- `modelRefs` must be unique, remain in artifact arm order, and must not
  collide with an auto-routing alias. Startup rejects any mismatch with the
  mounted manifest, including a configured endpoint whose credential does not
  come from exactly the worker's declared `api_key_env`.
- Router Replay must be disabled for ARC decisions (a decision-level
  `router_replay` plugin with `enabled: false` when replay is globally on);
  episode requests are never persisted.
- The encoder model and revision are frozen. The vLLM build, IO plugin,
  serializer, and required serving capabilities are pinned and checked by
  readiness.
- Redis is the durable episode backend. `memory` requires
  `development_mode: true`, a positive bound, and sticky single-replica use.
- Redis passwords are named through `password_env`; the canonical config never
  contains the credential value.
- Retained-session deployments choose exactly one `encoder.base_url` or a
  static, versioned `encoder.replicas` set. Replica mode requires an explicit
  final-turn close header and disables per-client retries.

## Configuration

```yaml
routing:
  decisions:
    - name: arc-route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        # Replace these public placeholders with the artifact-declared logical
        # arms, in exact manifest order, in private deployment configuration.
        - model: public-arm-a
          use_reasoning: false
        - model: public-arm-b
          use_reasoning: true
      adaptations:
        mode: bypass
      algorithm:
        type: rayline_arc
        on_error: fail_closed
        rayline_arc:
          artifact_dir: /var/lib/vllm-sr/rayline-arc
          artifact_revision: ${RAYLINE_ARC_ARTIFACT_REVISION}
          encoder:
            base_url: http://rayline-arc-encoder:8000
            model: Qwen/Qwen3.5-0.8B
            model_revision: 2fc06364715b967f1860aea9cf38778875588b17
            expected_build_id: ${RAYLINE_ARC_VLLM_BUILD_ID}
            expected_io_plugin_version: rayline-arc-io@0.1.0
            serializer_version: mtrouter-token-blocks-v2
            serving_rung: B
            required_pooling_capabilities:
              - chunked_causal_mean
            modal_key_env: RAYLINE_ARC_MODAL_KEY
            modal_secret_env: RAYLINE_ARC_MODAL_SECRET
            connect_timeout_seconds: 5
            total_timeout_seconds: 180
            max_retries: 1
          episode:
            id_header: x-rayline-episode-id
            backend: redis
            key_prefix: "vsr:rayline-arc:"
            acquire_timeout_seconds: 30
            lease_ttl_seconds: 60
            idle_ttl_seconds: 900
            max_in_memory_episodes: 1024
            redis:
              address: redis:6379
              password_env: RAYLINE_ARC_REDIS_PASSWORD
              pool_size: 16
```

`total_timeout_seconds` is the complete encoder request budget across retries,
not a per-attempt timeout. Freeze the production value from the maximum-context
GPU canary; do not assume the example is an adequate production threshold.

`serving_rung: B` selects vLLM's in-engine causal MEAN path and requires
`chunked_causal_mean`. Rung A's `all_plugin_mean` remains a diagnostic
bootstrap and is not the production maximum-context serving shape.

The stateless comparison arm reports only `chunked_causal_mean` and uses
vLLM's standard `/pooling` wire. To select the explicit retained-session wire,
require both capabilities:

```yaml
required_pooling_capabilities:
  - chunked_causal_mean
  - resumable_causal_mean
```

That mode sends the complete reconstructible history directly to
`/v1/rayline/arc/session/pooling`. Exact token extensions compute only their
suffix; retry, mismatch, eviction, affinity loss, and restart remain correct
because the session is optional acceleration state. Automatic prefix caching
stays disabled: `resumable_causal_mean` describes a pinned live request, not a
vLLM prefix-cache hit.

### Static retained-encoder replicas

For two to eight independent vLLM retained-session services, replace
`encoder.base_url` with a static membership block. All replicas share the same
model, build, plugin, serializer, capabilities, timeout, and optional Modal
credential shape:

```yaml
encoder:
  replicas:
    - id: encoder-a
      base_url: http://rayline-arc-encoder-a:8000
      state: active
    - id: encoder-b
      base_url: http://rayline-arc-encoder-b:8000
      state: active
  failover:
    schema_version: rayline.arc.encoder-failover.v1
    unavailable_status_codes: [404, 410, 502, 503, 504]
    unavailable_cooldown_seconds: 30
    max_remaps: 1
  model: Qwen/Qwen3.5-0.8B
  model_revision: 2fc06364715b967f1860aea9cf38778875588b17
  expected_build_id: ${RAYLINE_ARC_VLLM_BUILD_ID}
  expected_io_plugin_version: rayline-arc-io@0.1.0
  serializer_version: mtrouter-token-blocks-v2
  serving_rung: B
  required_pooling_capabilities:
    - chunked_causal_mean
    - resumable_causal_mean
  connect_timeout_seconds: 5
  total_timeout_seconds: 180
  max_retries: 0
episode:
  id_header: x-rayline-episode-id
  close_header: x-rayline-episode-close
  # Keep the remaining Redis lease/TTL fields from the complete example.
```

New episodes use deterministic rendezvous placement across `active` members.
An existing episode stays on its persisted owner even when that member is
`draining`. Only an explicitly configured HTTP status can trigger one remap;
transport, timeout, decode, and identity failures fail closed without calling
a peer. Treat the status list as a deployment assertion that those responses
occur before retained mutation, not as a generic retry list.

The configured close header accepts exact `true` on a final request. After the
provider returns 2xx, Semantic Router concurrently closes every visited
encoder owner. Clean close clears encoder affinity but retains ARC policy
history. For rolling replacement, first change a member from `active` to
`draining`, wait at least `episode.idle_ttl_seconds` after its last admitted
owner while watching close/session metrics, and only then remove it. Premature
removal fails closed.

See [Rayline ARC Retained-Encoder Replica Contract](../../../../../docs/architecture/rayline-arc-replica-membership.md)
for the full interaction, failure, observability, and rollout model.

Modal proxy authentication is configured by environment-variable name, never
by embedding credentials in YAML. Configure `modal_key_env` and
`modal_secret_env` together for a protected Modal web endpoint, or omit both
for an internal endpoint that does not use Modal proxy authentication.

## Deployment

The public Helm profile is
`deploy/helm/semantic-router/values-rayline-arc.yaml`. It mounts an existing
read-only `rayline-arc-artifact` PVC and reads credentials and private pins
from an existing `rayline-arc-runtime` Secret. Before deployment:

1. Populate the PVC from the immutable artifact revision. Do not use a mutable
   tag, and do not edit the mounted artifact in place.
2. Create the Secret keys named in the profile, including the protected Modal
   encoder endpoint, Modal proxy credentials, Redis address/password, and
   OpenRouter key.
3. Create a private values overlay containing every artifact-declared logical
   arm in exact manifest order. Its provider model, provider slug, thinking
   mode, and four cache-aware prices must match the manifest exactly.
4. Deploy the protected Rung B encoder from
   `src/vllm-plugins/rayline_arc_io/modal_service.py` only after its CUDA
   correctness gate passes.
5. For replica mode, provide stable IDs and independent endpoints in the
   private values overlay. Do not place a load balancer behind one replica ID;
   the ID is the retained-state owner.

Render and deploy a pinned chart release:

```bash
helm upgrade --install semantic-router \
  deploy/helm/semantic-router \
  --namespace semantic-router \
  --create-namespace \
  --values deploy/helm/semantic-router/values-rayline-arc.yaml \
  --values /private/rayline-arc-values.yaml \
  --atomic --wait
```

Readiness is intentionally fail closed. A missing credential, unavailable
encoder or Redis store, mutable/mismatched artifact, build/plugin/capability
drift, arm-order mismatch, provider mismatch, or pricing mismatch keeps ARC
unavailable rather than selecting a default arm.

For local CPU integration, `make rayline-arc-test-integration` builds and runs
real Envoy, Semantic Router, and Redis with a generated full-shape synthetic
artifact plus contract-faithful encoder and provider doubles. It covers both
arms, dispatch ownership, 2xx/non-2xx transactions, stream abort, client
cancel, same-episode fencing, cross-episode concurrency, two-replica affinity,
explicit-status failover, cooldown recovery, close fanout, stable-zero retained
sessions, router restart, and log privacy. It does not claim Qwen/CUDA
correctness; the Modal CUDA gate is separate.

## Rollback

Keep the prior Helm release, encoder deployment, artifact PVC, private values
overlay, and Secret version available as one immutable set. If readiness or
live canaries fail, stop admitting ARC traffic and roll back the router:

```bash
helm history semantic-router --namespace semantic-router
helm rollback semantic-router <LAST_GOOD_REVISION> \
  --namespace semantic-router --wait
```

Restore the matching previous encoder deployment and Secret/PVC references if
they changed. Do not point old router code at a new artifact or rewrite an
artifact in place. Verify `/health`, one public synthetic request per arm,
Redis episode advancement on 2xx, and absence of privacy canaries in logs
before restoring traffic. Delete or scale down the failed Modal deployment
after traffic is removed so it cannot continue incurring GPU cost.

## Status

This algorithm remains experimental. It is architecture-parity support, not a
model-quality or promotion claim. Production readiness additionally requires
the GPU numerical gates, Redis transaction tests, provider dispatch checks,
privacy canary, and full-stack acceptance described by the implementation
plan.
