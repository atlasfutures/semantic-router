---
title: Rayline ARC Policy Service
description: Serve pathfinder SDK checkpoints through a Modal policy service that runs the encoder, head and selection rule and returns the decision.
created: 2026-09-28
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-09-28

## Problem

Rayline ARC on `vsr-next` serves an MTRouter-family artifact
(`rayline.mtrouter-runtime.v3`). The Modal encoder returns one mean-pooled
vector and a Go MLP scores arms. Pathfinder now trains a different family (the
SDK, `pathfinder.serving-policy.v1`). Those checkpoints read the final-layer
state of every context block, attend over block history with per-block arm
attribution, and cannot load here. The epic memex-desktop#7116 plans block
readouts on vLLM (#83–#88) and a Go port of the SDK head (#90).

A Go port is the wrong boundary for three reasons:

- A long session's block history is about 4 MB. Scoring next to the encoder
  avoids moving it on every decision.
- Recipes change often (MLP, GRU, retrieval, temporal difference). Each port
  is a second implementation kept in parity by hand.
- The current retrain uses temporal difference. Its decision score is a
  learned function of prices, not `quality − w·cost`. VSR cannot recompute
  it from factored outputs.

## Proposal

The Modal encoder becomes a **policy service**. It runs request projection,
serializer (`arc-role-blocks-v1`), vLLM encoder, the head (a `torch.export`
program) and pathfinder's selection rule. It returns the **decision**: the
selected action, the reason, and every action's score, predicted quality and
predicted billed tokens.

VSR keeps everything that is not model inference:

- resolving aliases and operating points (including #7111's
  `rayline/like:<reference>` grades);
- the model-switch schedule and the thinking lever;
- binding each action to a provider route, then translation and dispatch;
- episode state, the arm-attribution ledger, and accounting of actual usage.

VSR never re-ranks. It sends every input that doesn't come from tokens on
every call, so the service's sessions are caches that can rebuild from the
request alone.

The normative contract lives in pathfinder (`docs/arc_serving_contract.md`,
proposed ADR 0107):

- `POST /v1/rayline/arc/policy/decide`: request
  `rayline.arc.policy-decision-request.v1`, response
  `rayline.arc.policy-decision-response.v1`.
- `GET /v1/rayline/arc/policy/packages`: loaded packages by alias and hash.
  VSR refuses an alias that isn't loaded or whose hash differs from its config.
- Policy package `rayline.arc-policy-package.v4`: it replaces the v3 manifest
  and carries its prices. Live prices never change a decision.

`testdata/policy_service/` holds byte-identical copies of pathfinder's wire
fixtures, with `SHA256SUMS`. `policy_service_wire.go` decodes them strictly.
It is the client side the new scorer will use.

## Changes to the #7116 plan

| Issue | Change |
|---|---|
| #88 E2 | The response is the decision, not per-block rows. Block features stay in the service |
| #90 R2 | A thin client scorer behind `raylineARCScorer` replaces the Go head port. The v3 Go head keeps working until it is retired |
| #91 R3 | The ledger feeds `attribution` on every call |
| #93 | The schedule narrows `available_action_ids` or sets `held_action_id`, as pathfinder's learned session does |
| #97 | Shadow runs through the `shadow` field: one encode, several packages |

## Provider bindings

A package action (`action_id`) names a model, native effort, level and
reasoning budget, but no provider. VSR config binds each action to a worker:
provider, the provider's model name, route, credential and lever wire.
Remapping an action to another provider is a config change, and the package
and service are unaffected. The route the action was trained on stays in the
package as provenance.

Readiness refuses a binding that cannot carry the action's effort, or cannot
express its level with a qualified lever. The
selection log and usage records name both the `action_id` and the binding.

A binding declares what its action dispatches: `model` (the trained model
name), `effort` and `reasoning_max_tokens` as the catalog states them, and
`level`, whose steering suffix comes from a `thinking_lever` with source
`policy`. The loader recomputes pathfinder's `action_id` from those fields and
refuses a binding that does not reproduce it. Any worker may serve the trained
model; selection logs `policy_action_model` beside `worker_provider_model`. A thinking-off action
(`effort: none`) may carry a steer: it dispatches with thinking disabled and
the suffix, and its worker's lever must be a `prompt_steering_suffix`. At dispatch the effort or budget
replaces the derived reasoning controls: OpenRouter's `reasoning` object on
Chat, and `output_config.effort` with adaptive thinking, or enabled thinking
with the budget, on Messages. A decide response that scores an action with no
binding fails the turn.

`dispatch_effort: provider_default` sends each action without its declared
effort, so the provider's default applies; a declared reasoning budget still
travels. v4 packages were trained on turns whose effort never reached the
provider while budgets did (pathfinder #2655: the proxy dropped
`output_config`), so they are served this way until v5. The loader still checks the declared effort
against the `action_id`. The steering suffix still renders from the level, a
thinking-off action (`effort: none`) stays off, and the selection log records
`policy_declared_effort` beside `policy_dispatch_effort`.

On the neutral level the lever writes nothing, so a steer written on an earlier
turn stays in force. The record carries both `thinking_level_requested` and
`thinking_level_in_force`. This matches collection, where a "none" draw also
appended nothing.

A remap still changes cost and quality, because cache share and effort
handling differ by endpoint. So a remap on a production alias is canaried. If
the new route's rates leave the package's price scenario, the package is
re-exported with new prices.

VSR verifies a package only by `package_sha256`, the sha256 of its manifest
bytes. `package_id`, `profile_id` and `policy_id` are pathfinder-issued ids
that VSR compares as opaque strings. `action_id` is the one VSR recomputes, from
a binding's declared dispatch.

Every service error fails closed.

## Request formats

VSR sends the client's request in the format the service projects:

- **Messages and Chat:** the client's system, tools and messages, exactly as
  received.
- **Responses** (`openai_responses`): `input` is the fully materialized item
  history, and `instructions` is the system prompt. The history is the stored
  history that `previous_response_id` resolves to, followed by this turn's
  input.

A turn's items must be a byte prefix of the next turn's, because the
attribution ledger identifies its prefix by those bytes. So every item is
encoded one way, whether it comes from the store or from the body, and the
router's retention fields (item id and status) are left out. Attribution may
name any assistant-output item: an assistant message, a function call, or
reasoning.

## Transport and admission

`policy_service` bounds and sheds calls the way the artifact mode's encoder
block does:

- `total_timeout_seconds` (1 to 900) bounds each decide call and each
  readiness probe.
- `connect_timeout_seconds` bounds the dial and the TLS handshake. Zero
  selects 5 s, the encoder's shipped value, and it may not exceed the total.
- `max_inflight_calls` (0 to 32) caps concurrent decide calls per router
  process. A call beyond the cap is shed after the episode lease and before
  it reaches the service. It answers 429 with `retry-after: 1`, as encoder
  admission does, and is counted by the same admission and in-flight metrics.
  Zero disables the cap.

The service's own back-pressure errors, `session_busy` and
`session_capacity`, also answer 429. Other service errors answer 503. Their
failure class comes from the contract's error codes only; any other value is
`service_error`.

## Open questions

- A context over encoder capacity is refused, never truncated. Should VSR
  serve a fallback action or a 503? It fails closed until the operator decides.
- Where each replica loads packages from, and how the alias registry is
  rolled out.
- Session caps once block features are resident, measured on L4 and RTX PRO
  6000.
