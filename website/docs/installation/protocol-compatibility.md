---
title: Protocol Compatibility Matrix
description: Match client-facing inference APIs to supported backend model protocols and understand cross-protocol feature boundaries.
---

# Protocol Compatibility Matrix

Semantic Router supports three inference wire formats on both sides of the
data plane. A client request is decoded into a protocol-neutral form, routing
policy selects a model, and that model's `api_format` selects the backend codec.
The response is translated back to the client's original format.

```text
client endpoint -> client codec -> routing -> backend codec -> model endpoint
```

Protocol compatibility is separate from target configuration and deployment
support:

- use [Backend Target Compatibility](backend-target-compatibility) for URLs,
  weights, headers, discovery, and producer preservation; and
- use [Deployment Support](support-matrix) for
  project-maintained stacks, integrations, and hardware profiles.

## Client-facing protocols

| Client API | Inference endpoint | Buffered | Streaming | Availability |
| --- | --- | --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | Supported | Supported | Available on the public inference listener. |
| OpenAI Responses | `POST /v1/responses` | Supported | Supported | Requires `global.services.response_api` and its store to be available. Router-owned object operations are not forwarded to a model backend. |
| Anthropic Messages | `POST /v1/messages` | Supported | Supported | Send the Anthropic request shape and an appropriate `anthropic-version` header. Client authentication remains deployment-specific. |

The public listener also serves `GET /v1/models`. See the
[Router API](../api/router) for the complete method and path inventory,
Responses object operations, and request examples.

## Backend model protocols

Set `api_format` on each `providers.models[]` entry. It describes the wire
contract implemented by that model endpoint, not the provider brand.

| `api_format` | Backend request and response shape | Default upstream path | Notes |
| --- | --- | --- | --- |
| `openai` | OpenAI Chat Completions | `/v1/chat/completions` | Default when `api_format` is omitted. `backend_refs[].chat_path` can override the Chat path. |
| `responses` | OpenAI Responses | `/v1/responses` | The backend itself must implement the Responses wire contract; enabling the Router's Responses service does not add that API to a backend. |
| `anthropic` | Anthropic Messages | `/v1/messages` | Configure provider authentication and required version headers on the backend ref. |

### Several accepted formats

A model whose backends accept more than one format lists them in
`accepted_formats`, in preference order. Each request then goes out in the
client's own format when it is listed, and otherwise in the first one:

```yaml
providers:
  models:
    - name: claude
      provider_model_id: anthropic/claude-opus-5
      accepted_formats: [anthropic, openai]
      backend_refs:
        - name: openrouter-claude
          provider: openrouter
          api_key_env: OPENROUTER_API_KEY
```

A Chat client reaches this model as Chat and a Messages client as Messages; a
Responses client, whose format is not listed, is translated to Messages. The
backends are bound in the first format, so `api_format`, if set, must equal it.
Every backend must serve every listed format, and a backend with a catalog
mapping must list the format on that mapping; the loader refuses the model
otherwise. On an OpenAI backend, list `responses` first so a client in an
unlisted format, such as Messages, is translated to Responses. Without
`accepted_formats` the model takes every request in its single `api_format`.

Rayline ARC checks that depend on the format (a `worker_thinking` base, a
`per_turn_effort` lever, a policy action's reasoning) must hold for every
accepted format of the worker, since any of them may be chosen per request.

These fields are easy to confuse:

- model `api_format` selects the request, response, error, and streaming codec;
- backend-ref `protocol` selects HTTP or HTTPS transport; and
- backend-ref `provider` supplies provider-specific authentication and path
  defaults. It does not prove that the endpoint implements an API format.

For every backend format, `base_url` names the complete upstream API root. Its
path is retained and the protocol operation suffix is appended exactly once;
the protocol's default `/v1` base path is used only when the URL has no path.
`chat_path` applies only to Chat Completions.

For an HTTPS backend, the generated Envoy cluster verifies both the server
certificate chain and its DNS hostname. An HTTPS replica pool must keep one
hostname because the supported Envoy runtime shares its TLS context within a
cluster; use separate model aliases for different HTTPS hosts. IP-literal HTTPS
targets are rejected instead of silently weakening hostname verification. A
custom Envoy image used with `vllm-sr serve` must provide the system CA bundle at
`/etc/ssl/certs/ca-certificates.crt`; startup validation fails rather than
silently disabling verification when that trust store is unavailable.

## Client-to-backend matrix

Every client format can route to every backend format. Each cell is covered in
both buffered and streaming mode.

| Client protocol | `openai` backend | `responses` backend | `anthropic` backend |
| --- | --- | --- | --- |
| OpenAI Chat Completions | Supported | Supported through codec translation | Supported through codec translation |
| OpenAI Responses | Supported through codec translation | Supported | Supported through codec translation |
| Anthropic Messages | Supported through codec translation | Supported through codec translation | Supported |

"Supported" means the Router owns the request, response, transport-error, and
streaming translation path. It does not mean every field from one protocol can
be represented by every other protocol, or that every model behind an endpoint
supports the requested capability.

## Feature portability

The Router checks required semantics before encoding a backend request. A
feature that the selected backend format cannot represent fails explicitly
instead of being silently dropped.

| Semantic feature | Chat Completions | Responses | Messages |
| --- | --- | --- | --- |
| Text, image input, and file input | Supported | Supported | Supported |
| Tools, parallel tool calls, and strict tool schemas | Supported | Supported | Supported |
| Strict JSON Schema output | Supported | Supported | Supported |
| Buffered and streaming responses | Supported | Supported | Supported |
| Reasoning content and effort | Supported | Supported | Supported |
| JSON object mode without a schema | Supported | Supported | Not supported |
| Audio input | Supported | Not supported | Not supported |
| Hosted image-generation lifecycle | Not supported | Supported | Not supported |
| Multiple response candidates | Supported | Not supported | Not supported |
| Per-block prompt-cache directives | Supported | Not supported | Supported |
| Request-level automatic-cache directive | Dropped and counted | Supported extension | Supported |
| Reasoning token budget | Supported extension | Not supported | Supported |
| Seed and frequency or presence penalties | Supported | Not supported | Not supported |
| `top_k` sampling | Not supported | Not supported | Supported |
| Stop sequences | Supported | Not supported | Supported |
| Native response or conversation state fields | Not supported | Supported | Not supported |

This table describes codec representation, not model capability. For example,
an OpenAI-compatible server can accept the Chat request shape while rejecting
images or tools for a particular model. Qualify the actual endpoint and model
revision before adding them to a routing pool.

The request-level automatic-cache directive is a top-level `cache_control`
object, such as `{"type": "ephemeral"}` with an optional `ttl` of `5m` or `1h`.
Anthropic Messages defines it as automatic caching, and OpenRouter accepts the
same member as an extension on Responses requests. The Router accepts it on
Messages and Responses ingress and refuses a malformed value. Each backend
format handles it as follows:

- **Messages backend:** a Messages client's member is sent back as written. A
  request translated from another format has no top-level member to send, so
  the Router places one breakpoint where automatic caching would: on the last
  message block that can carry `cache_control`. If no message block can carry
  one, the breakpoint goes on the last system block, and if there is none, on
  the last tool. A block that already has a breakpoint keeps it. If the
  request already uses all four breakpoints that Messages allows, the Router
  adds no breakpoint and counts the drop.
- **Responses backend:** the top-level member is carried.
- **Chat Completions backend:** the directive is dropped and counted as a
  translation diagnostic.

A Chat Completions or Responses client usually sends no `cache_control`. It
relies on the automatic prefix caching that OpenAI models apply, and may name
a cache shard with `prompt_cache_key`. Anthropic caches only at explicit
breakpoints, so when such a request is dispatched over Messages to a Claude
worker (a model whose card publisher is `anthropic`, or whose provider model
id starts with `anthropic/`), the Router supplies the directive
`{"type": "ephemeral"}`. It uses the default 5-minute TTL unless the client
asks for longer retention, as described below. The breakpoint is then placed
as described above. The following rules apply:

- A client's own top-level `cache_control` is used as written. The Router
  doesn't supply its own.
- A client's own per-block breakpoints, on a tool, a system block, a message
  block, or a block inside a tool result, also count as its cache directive.
  The Router adds no breakpoint and dispatches the client's breakpoints as
  written, because an extra breakpoint would change the client's cache-write
  cost. The skip is logged as `auto_cache_skipped` with reason
  `client_breakpoints`. Breakpoints are counted on the request as it is
  dispatched, after the decision's tools plugin has run. A breakpoint on a
  tool that the plugin removes is not sent, so it doesn't prevent the
  automatic breakpoint.
- A Messages client, a non-Claude worker, and a Claude worker reached over Chat
  Completions or Responses get no breakpoint from this rule.
- Neither Chat Completions nor Responses defines a way to opt out of automatic
  caching, and the Router has no setting to turn this off.
- A client's `prompt_cache_retention` sets the TTL. `24h` asks for longer
  retention than Anthropic offers, so the Router uses Anthropic's longest TTL,
  `1h`, and records an `approximated` diagnostic. Because Anthropic gates the
  `1h` TTL on a beta, the Router adds `extended-cache-ttl-2025-04-11` to the
  dispatched `anthropic-beta` header. The value is merged into the header the
  provider profile or the client set; no existing value is replaced or
  repeated. `in_memory` keeps the
  5-minute default. Any other value also keeps the default, and the Router
  records a `dropped` diagnostic.
- The supplied directive is recorded as a `generated` translation diagnostic.
  Its reason names the client's cache intent:
  `automatic_cache_prompt_cache_key`, `automatic_cache_prompt_cache_retention`
  when the client sent a retention but no key, or `automatic_cache_default`.
  The diagnostic appears in the protocol warnings header and the
  translation-warning counter.
- The directive is applied only to the dispatched request, never to the
  client's stored request. It is the same for every turn, so a growing
  conversation keeps a cache-stable prefix.

A Responses client can still use `previous_response_id` with a Chat
Completions or Messages backend. The Router retrieves and materializes the
retained history, removes Router-owned object controls, and then encodes the
stateless request in the selected backend format.

## Configure a backend format

The client can use any supported client-facing endpoint; `api_format` controls
what the selected backend receives:

```yaml
providers:
  models:
    - name: hosted/claude
      provider_model_id: claude-model-id
      api_format: anthropic
      backend_refs:
        - name: anthropic-primary
          base_url: https://api.anthropic.com
          provider: anthropic
          api_key_env: ANTHROPIC_API_KEY
          extra_headers:
            anthropic-version: "2023-06-01"
          weight: 100
```

`api_format` chooses only the backend codec. It does not imply Anthropic,
OpenAI, or any other runtime Provider. Router-owned listeners require a
physical model to declare `backend_refs[].provider`; metadata-only
`listeners: []` configurations leave transport and credentials to the external
gateway. The local `vllm-sr serve` workflow manages Envoy transport, so it does
not accept a backendless physical model; use the external-gateway deployment
profile for that topology.

Test the backend directly with its native path and a minimal request first.
Then send the same semantic request through the Router using the client API that
your application needs. A successful health check does not validate request
schema, streaming, tools, or error translation.

## Validation and failure behavior

- Public requests are decoded into the neutral contract even when client and
  backend formats match. Unknown or unsupported request fields fail closed.
- Cross-protocol requests preserve shared semantics. Target-specific features
  that cannot be represented return a typed protocol error.
- The response keeps the client protocol's JSON or SSE shape. Provider
  transport errors and incomplete streams are translated separately from
  successful model responses.
- A backend stream keepalive (an Anthropic `ping`, or an SSE frame of comment
  lines only) reaches the client as one keepalive in the client's format:
  `event: ping` for a Messages client once the message has started, and an SSE
  comment otherwise. Nothing follows the terminal event.
- A reply cut off at a length limit while the model was writing a tool call
  ends as a length stop in the client's format, with the call marked
  incomplete. The limit is the output limit (`max_tokens`) or the model's
  context window (Anthropic's `model_context_window_exceeded`). Messages keeps
  the `tool_use` block under the provider's stop reason (with `input: {}` in a
  non-streaming response when the arguments were cut mid-object), Chat ends
  with `finish_reason: "length"` and the partial arguments, and Responses ends
  the `function_call` item with `status: "incomplete"` inside an incomplete
  response whose reason is `max_output_tokens`. Chat and Responses name only
  the output limit, so a context-window stop reaches them as that length stop
  whether or not a call was cut, and a translation diagnostic records the
  approximation.
- A refusal that stops the model during a tool call is a refusal, not an
  argument error. The final call is marked incomplete whatever its arguments,
  and the client gets the refusal as it would without a call: Messages
  `stop_reason: "refusal"`, Chat `finish_reason: "content_filter"`, and an
  incomplete Responses response whose reason is `content_filter`.
- A failure ends the turn as that failure even while a cut call is held. A
  provider's in-band error, or the Router ending the stream itself (its
  deadline, or a transport end), reaches the client as that error. A stream
  that ends without a terminal event is `stream_incomplete`.
- Truncated tool arguments fail as `invalid_stream_tool_arguments` only under
  a stop that says the reply finished (`end_turn`, `tool_use`, `stop`, or
  `pause_turn`), or when more output follows the cut call.
- A turn whose tool call was cut is not stored in the response cache.
- `x-vsr-client-protocol`, `x-vsr-upstream-protocol`, and
  `x-vsr-protocol-warnings` expose translation details when applicable. See
  [VSR routing headers](../troubleshooting/vsr-headers).

The repository verifies all three protocols pairwise in codec tests, at the
Envoy ExtProc boundary, and in an 18-cell deployment matrix: three client
formats by three backend formats by buffered or streaming mode. See the
[implemented codec design](../proposals/multi-protocol-adaptor) for the full
verification and extension contract.
