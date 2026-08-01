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

Run the hermetic contract with:

```bash
e2e/testing/rayline-arc/run.sh
```

It covers 429-to-200, 503-to-200, streaming 429-to-200, exhausted 429 and 503,
single commit/abort behavior, post-200 partial streaming, process restart,
Redis loss, aggregate attempt metrics, and privacy-log scanning.
