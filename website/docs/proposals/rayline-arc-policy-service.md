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
Chat, `output_config.effort` with adaptive thinking, or enabled thinking
with the budget, on Messages, and `reasoning.effort` on Responses. Responses
has no reasoning budget, so readiness refuses a budget action bound to a
worker that accepts Responses. A thinking-off action sends effort `none` on
Responses, whatever the client asked for, and readiness admits it only on a
worker whose reasoning family has that off signal. A decide response that scores an action with no
binding fails the turn.

A Responses client running with `store: false`, such as Codex, resends every
reasoning item it has received with its `encrypted_content`, which only the
target that issued it can read. The episode records the set of targets that
issued the blobs its client can still hold (`reasoning_issuers`: at most two
truncated digests of worker, backend, provider model and the credential the
turn is sent with, since a blob is readable only by the account that issued
it), and a turn forwards the items unchanged only when that set is exactly its
own target. A rotated or per-user key is another issuer. An
OpenRouter target has no such identity: OpenRouter chooses the serving
provider per request, and providers cannot read each other's blobs. So
nothing is forwarded to one, and blobs it issues are recorded as of unknown
issuer and never forwarded anywhere.
On any other target the whole item is dropped, as on every route without an
episode, and logged as `rayline_arc_encrypted_reasoning_dropped` with a reason.
The set is written only with the turn's 2xx commit. A turn that resends no
encrypted reasoning restarts the set; once it holds two issuers it stays so,
and the episode's later turns drop the items, until such a turn.

`dispatch_effort: provider_default` sends each action without its declared
effort, so the provider's default applies; a declared reasoning budget still
travels. v4 packages were trained on turns whose effort never reached the
provider while budgets did (pathfinder #2655: the proxy dropped
`output_config`), so they are served this way until v5. The loader still checks the declared effort
against the `action_id`. The steering suffix still renders from the level, a
thinking-off action (`effort: none`) stays off, and the selection log records
`policy_declared_effort` beside `policy_dispatch_effort`.

**Package v5.** A v5 package (`rayline.arc-policy-package.v5`, pathfinder's
"Policy package v5"; ADR 0109) names each action by a trained model (the name
its rows were fit on, such as `claude-opus-5`) and a format-agnostic thinking
control. `package_manifest` points at the package's
`package.json`, whose sha256 must be `package_sha256`. At load VSR:

- recomputes each action's `control_id` from its control (RFC 8785) and refuses
  a mismatch;
- requires each control in the compiled registry it serves from (pathfinder's
  `configs/thinking_controls.compiled.json`, embedded and pinned in
  `pkg/selection/raylinearc/thinkingcontrol`);
- requires every package action bound, and each bound control admitted on its
  worker's (served model, provider, format) cell for every format the worker
  accepts, since the target format is chosen per request. The served model is
  the worker's provider model id (what its dispatch sends, such as
  `anthropic/claude-opus-5`), never the action's trained name; an experimental
  instruction needs `allow_experimental_controls`;
- requires `trained_models` to declare, for every bound worker, the trained
  model it serves, equal to its actions' model. Trained names are decoupled
  from providers, so the pairing is declared, never inferred; a missing or
  different declaration, or one no binding uses, is refused.

`thinking_controls_sha256` is informational. A v5 binding is only `action_id`
and `worker`, beside `trained_models: {<worker>: <trained name>}`. The v4
fields, `dispatch_effort`, `thinking_lever` and `worker_thinking` are refused
with a v5 package, and `trained_models` with a v4 one.

**Package v6.** A v6 package (`rayline.arc-policy-package.v6`, pathfinder's
"Policy package v6"; ADR 0122, pathfinder#3495) is a v5 package whose encoder
reads images natively. Only `encoding_profile` differs: it is the
`canonical_v2` image profile (pathfinder's `ImageEncodingProfile` in
`src/rayline_router/serving/arc_policy_contract.py`), with serializer
`arc-role-blocks-v2`, `conversation: canonical_v2`, `modalities` text and
image, and the `image_processor`, `positions` and `vision` members the
contract fixes. VSR decodes the profile strictly, only to refuse one it does
not know; the policy service encodes by it, and `profile_id` stays opaque.
Every other member keeps its v5 meaning and checks, and a v6 package loads,
binds (`action_id` and `worker`, with `trained_models`), readies and
dispatches exactly as v5.

The schemas do not cross. A v4 or v5 manifest holding any v6 member,
`conversation: canonical_v2` or serializer `arc-role-blocks-v2` is refused;
so is a v6 manifest with a text profile (no `conversation`, `canonical_v1`,
serializer `arc-role-blocks-v1`, or any image member missing), any unknown
key at any level of a v6 profile, a value outside the contract (such as
`rope: null` or a `vision.dtype` other than `float32`), and any
`schema_version` but v4, v5 and v6. The v4 binding fields are refused on a v6
package as on v5. The decide request is unchanged: client bodies reach the
service with their images, and Responses history from `previous_response_id`
keeps its `input_image` parts.

`encoding_profile.harness_shell` (v4, v5 and v6) is absent, which excludes
the harness shell, or `include` or `include_v2` (tool rule v2,
pathfinder#3653). VSR treats the value as opaque and the policy service
applies the projection; any other value is refused.

Readiness does not refuse a v6 binding to a text-only worker. An image turn
leaves a worker off the offer only when its model card sets `vision: false`;
an unmarked card counts as vision-capable, so a basket serving a v6 package
must mark its text-only workers (#215).

Route construction admits the control on the cell of the request's target
format and resumes the episode's placer. The provider boundary then renders it
from the registry, after the codec. The cell's base wire replaces every thinking field (the
client's `thinking`, `reasoning`, `reasoning_effort` and
`output_config.effort`), so a Messages effort gets no adaptive thinking block.
The instruction is placed by `turn_tail_v2`, written by `on_change_v1` and
replayed by `ledger_v1`. The placer's state lives in the episode, one per worker
and control shape. The renderer is pathfinder's reference renderer ported to Go
and held to its golden corpora byte for byte. Messages, Chat and Responses
workers are served wherever the worker's cells admit the control. In registry
3083a4b6 every Responses cell admits only the native-default control (no
instruction, empty base wire), so a Responses worker can bind a package's
native-only action and is refused an instruction-bearing one; Responses steers
become servable once pathfinder admits those cells. A Codex turn's encrypted
reasoning still follows the episode's issuer record (#109).

Under `emit: on_change_v1` (ADR 0109) an item is written only when the level
in force changes. A return from a steer to the neutral level writes the
binding's `neutral_text`, the neutral marker; the neutral level at epoch start,
or repeated, writes nothing, and a transcript rewrite re-asserts a steered
level only. Each call's routing record attributes the level in force, not the
one requested: `thinking_level_in_force`, its `thinking_control_sha256` (the
neutral level's own control under the marker), and
`thinking_instruction_state` (`never`, `neutral_marker` or `steered`), beside
`thinking_level_requested` and `thinking_written`. Under the older
`emit: on_change` the neutral level writes nothing, so a steer written on an
earlier turn stays in force (`thinking_skipped: neutral_inexpressible`).

The selection log's `thinking_level` and the routes API's `thinking_level` are
the decision's level: both are produced before, or without, any dispatch, when
no level is yet in force.

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
  input. `tools` is this turn's top-level `tools`, exactly as received (`[]`
  included). When the client sent none, or `null`, it is `[]`: on the
  Responses API an omitted `tools` means the request has none. Only VSR images
  that forward tools (#217 onward) send it; a decide without `tools` comes
  from an older router, and its tool coverage is unknown.

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

## Episode turns

A policy-service turn commits its ledger entry, turn count and held arm only
once the client has the whole 2xx response: the stream's terminal event, or
the full non-stream body. A failed, non-2xx or broken response commits
nothing. Artifact-mode turns commit at the 2xx response headers.

A side call keeps the held arm, takes no episode lease and commits nothing.
It is decided with `episode_mode: relaxed` even on a strict cell, so the
service locks no session for it and leaves the main conversation's session
as it was. A service that cannot serve relaxed (it refuses with
`unsupported_request`, or predates `episode_mode` and answers with a session
revision) gets that cell's side calls strict for ten minutes, then is asked
relaxed again.
Only an explicit compaction moves the schedule's boundary; a transcript that
stops extending the recorded prefix only starts a new context epoch. Without
headers, the router classifies a request from harness literals in its body:
Claude Code's subagent flag, session-title prompt and summarization
directive, and codex's compaction directive and post-compaction summary.

`trust_turn_signal_headers` (default false) makes the router read headers
a gateway sets instead:

- `x-rayline-call-kind: main | side`
- `x-rayline-compaction: <ordinal>`, the positive ordinal of the compaction
  this request is the first request after.
- `x-rayline-agent-key-source: agent | role | task`, how the gateway keyed a
  harness subagent's episode (`episode.id_header`).
- `x-rayline-parent-session` and `x-rayline-parent-agent`, the episode id and
  agent id of the conversation a subagent was spawned from.

A subagent the gateway keyed on its harness agent id (`agent`) is its own
conversation: a main turn of its own episode, with its own ledger, schedule
and decisions. Keyed any other way, or with no key source, it stays a side
call of the episode it arrived on, and its key source is logged as
`unknown`. The parent link is logged (hashed) on `rayline_arc_policy_turn`
as metadata; nothing routes on it.

Set the option only when a gateway in front of the router sets these
headers, or strips them from client requests. Otherwise a client could mark
its own turns as side calls, which are not counted, or as compactions. With
the setting off, the headers are ignored. Either way, none of them is
forwarded to a provider.

## Fallback

`fallback.enabled` (default false) lets a cell serve around a model that
refused a turn (ADR 0120, Phase 1a). Leave it off for evaluation cells: they
serve exactly what the policy chose, and a refusal ends the turn as it did
in collection.

With it on:

- A refused turn excludes its model, every action of it, for the rest of
  the context. The next turn is offered only the other models, and a turn
  the schedule would have held on the excluded model decides again, as at a
  boundary.
- The exclusion lasts until the context ends: a compaction or a transcript
  that stops extending the recorded prefix lifts it. Side calls keep it but
  are not narrowed by it.
- When every model is excluded, the turn fails as having no available
  action. The package's fallback action is one of its own actions, so it is
  excluded with them.
- A turn that fails with a 429, a 5xx or a timeout takes its provider route
  (the worker, its backend endpoint, the provider model there and its
  provider pin) out of the offer for every episode the replica serves, for
  `fallback.cell_exclusion_seconds` (default 30, at most 3600; Phase 1b). A
  turn held on that route decides again. "No endpoints found" does not exclude a route: it is as often about
  what one request asks (a price cap, an image input) as about the route.
  A worker served by several endpoints is not route-excluded: Envoy balances
  its calls across them, so a failure cannot be pinned on one. A config
  reload starts with no route excluded.
  Route exclusions are advice: when they alone would leave nothing, the turn
  is offered as though no route had failed. Each replica keeps its own;
  `rayline_arc_cell_exclusion` logs each.
- Each decision taken with an exclusion in force is logged as
  `rayline_arc_fallback_decision`, with the request id, the exclusions and
  excluded routes, the offered actions, the package's scores and the choice.
- An episode that holds an exclusion is stored as episode-state v4. A
  router that predates v4 refuses such a record, so roll every replica of a
  cell before enabling the fallback on it.

## Open questions

- A context over encoder capacity is refused, never truncated. Should VSR
  serve a fallback action or a 503? It fails closed until the operator decides.
- Where each replica loads packages from, and how the alias registry is
  rolled out.
- Session caps once block features are resident, measured on L4 and RTX PRO
  6000.
