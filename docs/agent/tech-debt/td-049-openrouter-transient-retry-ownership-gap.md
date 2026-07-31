# TD049: OpenRouter Transient Retry Is Canary-Owned

## Status

Open — the bounded external canary handles transient pre-response failures,
but the production data plane does not yet own an equivalent policy.

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

The diagnostic canary now retries at most once for a pre-response 429 or 503,
reuses the same Rayline episode ID, honors a clamped `Retry-After`, paces
sequential requests, never retries a stream after HTTP 200, and records logical
requests separately from external attempts. This is intentionally canary-owned
so the acceptance run can tolerate ordinary provider variance without
silently changing the production gateway contract.

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
- The existing ARC transaction tests prove non-2xx provider responses abort
  without advancing episode state, which makes same-episode pre-response retry
  safe for the diagnostic packet.

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
- The external canary removes its client-owned retry loop and still passes the
  same real-provider acceptance packet.
