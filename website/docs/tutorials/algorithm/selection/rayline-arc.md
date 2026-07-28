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
- `modelRefs` must be unique and remain in artifact arm order. Startup rejects
  any mismatch with the mounted manifest.
- The encoder model and revision are frozen. The vLLM build, IO plugin,
  serializer, and required serving capabilities are pinned and checked by
  readiness.
- Redis is the durable episode backend. `memory` requires
  `development_mode: true`, a positive bound, and sticky single-replica use.
- Redis passwords are named through `password_env`; the canonical config never
  contains the credential value.

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
            required_pooling_capabilities:
              - all_plugin_mean
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

`required_pooling_capabilities` starts with `all_plugin_mean` for the uncached
bootstrap. Add `chunked_causal_mean` only after that vLLM capability is present
and tested. `prefix_cached_mean` is a separately gated optimization.

## Status

This algorithm remains experimental. It is architecture-parity support, not a
model-quality or promotion claim. Production readiness additionally requires
the GPU numerical gates, Redis transaction tests, provider dispatch checks,
privacy canary, and full-stack acceptance described by the implementation
plan.
