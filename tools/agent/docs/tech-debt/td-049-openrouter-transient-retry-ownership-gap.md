# TD049: OpenRouter Transient Retry Is Canary-Owned

## Status

Implementation complete; external confirmation pending — Envoy now owns the
production OpenRouter retry policy and the external canary no longer retries.
Hermetic transaction, streaming, attempt-accounting, and exhaustion gates pass.

## Owner Plan

[PL0041 Rayline vLLM Serving and Performance Qualification](../plans/pl-0041-rayline-vllm-serving-performance.md)

## Release Relevance

The real-provider canary can validate deployment wiring, but it cannot establish
production resilience while retries are implemented only in its client driver.

## Scope

- pre-response HTTP 429 and 503 handling for OpenAI-compatible providers;
- `Retry-After` propagation and bounded delay ownership;
- provider-attempt accounting versus logical routed-request accounting;
- Rayline episode identity and transaction behavior across a safe retry; and
- streaming requests before and after the response-header commit boundary.

Provider fallback, concurrency control, and general load shedding are out of
scope for this debt entry.

## Summary

ORC003 reached the pinned Fireworks endpoints and completed three real model
generations before the fourth coverage call returned HTTP 429. The canary
previously discarded the structured error metadata and `Retry-After` header and
failed immediately.

Envoy now retries at most once for a pre-response 429 or 503 on OpenRouter-only
routes, below one Rayline decision and prepared transaction. It honors the
bounded rate-limit delay, exposes the final attempt count, and never retries a
stream after HTTP 200. Self-hosted vLLM routes intentionally have no retry
policy because their 429 signals local admission pressure. The diagnostic
canary retains pacing and accounting but no longer supplies its own retry.

## Evidence

- `e2e/testing/rayline-arc/openrouter_fullstack_canary.py` contains the bounded
  diagnostic retry and dual request/attempt accounting.
- `src/vllm-plugins/rayline_arc_io/tests/test_openrouter_fullstack.py` proves a
  retry preserves episode identity, honors the bounded delay, emits no provider
  message or credential, and does not retry non-transient errors.
- ORC003 completed three Fireworks generations through the real ARC data plane
  before the fourth coverage request returned HTTP 429. Cleanup removed the
  one-run OpenRouter key, Modal proxy credential, compose stack, and exact H100
  encoder container.
- ORC004 reused the same Rayline episode for one bounded retry of that fourth
  logical request. Both external attempts returned 429, proving the canary
  boundary and dual accounting work but do not resolve a sustained provider
  limit. The private aggregate receipt is pinned at
  `rayline-ai/router-artifacts@e060a95e4f1a03f1e369b31b271c9fc731c8ed24`.
- The existing ARC transaction tests prove non-2xx provider responses abort
  without advancing episode state, which makes same-episode pre-response retry
  safe for the diagnostic packet.
- `deploy/compose/rayline-arc/README.md` documents Envoy as the production
  owner, the OpenRouter/self-hosted boundary, deadlines, streaming semantics,
  and aggregate metrics.
- `e2e/testing/rayline-arc/run.sh` passes deterministic 429-to-200,
  503-to-200, streaming retry, exhausted retry, post-200 partial-stream,
  restart, Redis-loss, and privacy gates. The 2026-08-01 run observed three
  logical requests, six wire attempts, and three retries; `Retry-After: 1`
  recovered in 1.425, 1.390, and 1.166 seconds.

## Why It Matters

Retry ownership affects billing, latency, observability, transaction state, and
duplicate generation risk. A retry below Semantic Router may be invisible to
Rayline, while a retry above it repeats selection and lease work. Streaming is
especially sensitive because a response must never be replayed after the
header-time commit boundary. Leaving the policy only in a canary would make a
passing deployment result look more production-representative than it is.

## Desired End State

One production data-plane layer owns the documented transient-provider policy.
It honors provider delay signals, applies a strict attempt/deadline budget,
preserves Rayline's episode and transaction invariants, reports wire attempts
separately from logical requests, and refuses any retry after response headers
commit a streaming request. The external canary then tests that production
behavior instead of supplying it.

## Exit Criteria

- The authoritative production retry owner and supported statuses are
  documented in the provider execution contract.
- A pre-response 429/503 can retry without advancing Rayline state or changing
  the selected worker for the logical request.
- Retry delays honor bounded `Retry-After` values and fit within the request
  deadline.
- Metrics distinguish logical requests, upstream attempts, retries, and retry
  exhaustion without recording prompt, response, episode, or credential data.
- Streaming and cancellation tests prove no retry occurs after HTTP 200 headers
  or partial response delivery.
- The external canary removes its client-owned retry loop and passes the same
  real-provider acceptance packet. This final external confirmation remains
  open; ORC005 is not rerun as an implicit whole-packet retry.
