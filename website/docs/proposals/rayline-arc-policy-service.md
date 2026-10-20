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
  and carries its prices. Live prices change a decision only on a package
  whose manifest sets `pricing.live_prices_affect_decisions` (see "Live
  routing prices"); every other package scores its own prices.

`testdata/policy_service/` holds byte-identical copies of pathfinder's wire
fixtures, with `SHA256SUMS`. `policy_service_wire.go` decodes them strictly.
It is the client side the new scorer will use.

## Changes to the #7116 plan

| Issue | Change |
|---|---|
| #88 E2 | The response is the decision, not per-block rows. Block features stay in the service |
| #90 R2 | A thin client scorer behind `raylineARCScorer` replaces the Go head port. The v3 Go head keeps working until it is retired |
| #91 R3 | The ledger feeds `attribution` on every call |
| #93 | The schedule narrows `available_action_ids` or sets `held_action_id`, as pathfinder's learned session does. A two-stage package is the exception: its service holds the model, and VSR does not (#265) |
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

A two-stage package states `stage_one` (v4, v5 and v6): pathfinder's
stage-one part, holding the selector, its boundaries, the offered models and
the proxy arms. The policy service owns and applies it; VSR carries it opaque.
Absent is a single-stage package. When present it must be a JSON object, and
startup refuses an explicit `null` or any other JSON value (#229,
memex-desktop#7291).

The one thing VSR reads from it is that it is there. A two-stage package's
service chooses the model at the package's own stage-one boundaries (a session
or subagent start, a compaction, a typed user message) and holds it between
them from the request's attribution and its episode memory. So when a v5 or v6
manifest states `stage_one`, `model_schedule` holds nothing on a main turn:
every main turn is offered every model, with no `held_action_id`, and keeps
the arm it decided for a retry, as a schedule boundary does. A side call is
still held. Holding by turn count there would hide the typed user message from
the service, which then never decides at it (#265). A single-stage package,
or one served without a manifest (v4), keeps the turn-count hold.

A v6 package requires every bound arm's model card to state `vision`
explicitly, `true` or `false`, and startup refuses one that doesn't, naming
the unmarked arms (#215). An image turn takes a worker off the offer only when
its card sets `vision: false`, and an unmarked card counts as vision-capable,
so under a v6 package that default would send images to a text-only arm. A
text-only worker may still be bound when its card says `vision: false`: it
serves the package's text turns. v4 and v5 packages keep the default, so an
unmarked card there still counts as vision-capable.

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

A Chat or Responses cell without task-fidelity evidence writes nothing after an
image tool result (pathfinder ADR 0129). Such a call's record carries
`thinking_refused: image_tool_tail_task_fidelity`, and its
`thinking_control_sha256` is the control the policy drew, not one in force,
because the training label is the policy's action. What the provider saw is
`thinking_level_in_force`, the level already in force. A consumer reading the
delivered treatment uses `thinking_level_in_force` whenever `thinking_refused`
is set. The same holds on the usage record. A retry of a refused call repeats
the drawn control and the refusal.

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

## Live routing prices

`policy_service.routing_prices` (off when absent) attaches live per-model
rates to every decide request as `routing_prices`, so an eligible package
(factored, with `pricing.live_prices_affect_decisions`) scores current
provider prices instead of its exported ones. Pathfinder's
`arc_serving_contract.md` ("Live routing prices") defines how the service
maps them to arms, fills a null leg from the package, and falls back per arm.

- **Feed.** VSR reads `GET <url>` (schema `arc-routing-prices.v1`, USD per
  million tokens, one entry per `(model, endpoint_tag)`: the OpenRouter id
  and the OpenRouter endpoint the price is read at) and forwards the rates as
  served. It never maps them to arms. A model the feed lists at one endpoint
  is keyed by its bare OpenRouter id, as before endpoint tags; each endpoint
  of a model it lists at several (a standard endpoint and its flex twin) is
  keyed `<model>@<endpoint_tag>`. The feed is decoded strictly: a wrong
  schema, currency or unit, an empty `store_version`, one endpoint listed
  twice, a model at several endpoints with an untagged entry, an empty
  `endpoint_tag`, a model id containing `@`, or an `as_of` or `valid_until`
  that is not RFC 3339 with an offset rejects the whole response, and the
  last good copy stays in use. A VSR from before endpoint keys rejects a feed
  listing one model twice, so this version must serve before the pricing
  card prices a model at two endpoints. A model entry that
  omits a leg (rather than writing it `null`) is not forwarded, so its arms
  keep their package rates, as the service treats such an entry.
- **Auth.** `auth: google_id_token` sends an identity token minted by the
  metadata server for `audience`, the feed's own service URL; `url` must be
  on that host, since a fronting domain does not match the audience the feed
  checks. `auth: bearer` (or empty) sends the static token named by
  `credential_env`; if that variable is unset, live prices stay off.
- **Cache.** A background fetch refreshes the copy `refresh_ahead_seconds`
  before `ttl_seconds` runs out. A request reads it as `fresh`, `stale` (past
  the TTL, sent anyway) or `expired` (past `max_age_seconds`, not sent), or
  `absent` before the first good fetch. Without rates the service scores the
  package's prices, as before.
- **Echo.** The service answers `routing_prices_used` exactly when the
  request carried `routing_prices`. VSR fails the turn closed when the echo
  is missing (`policy_prices_unechoed`), present for a request that sent
  none (`policy_prices_unsent`), or names another `store_version`, `as_of`,
  or a filled rate key the request did not send (`policy_prices_mismatch`).
- **Audit.** Each main-turn ledger entry keeps the package hash, the mode,
  the feed identity, the rates digest, the first few fallback arms and filled
  models with their counts, and the selected action's predicted cost. The
  selection log carries the same and the cache status. Prometheus counts
  fetches by class, lookups by status, and decisions by mode.
- **Settlement.** The call an ARC decision dispatched is settled at the same
  feed, while the cache would still send it (within `max_age_seconds`, by its
  fetch and by its `as_of`). The row is the dispatched OpenRouter model at the
  endpoint its `provider` pin admits: `only`, or `order` with
  `allow_fallbacks: false`, less `ignore`, where an entry admits its own tag
  and that tag's variants (`openai` admits `openai/flex`). A pin naming one
  endpoint that admits one row settles at it; otherwise the response's
  `provider` must name exactly one admitted row. A row with a null or
  negative leg, outside its `valid_from`/`valid_until`, or with a
  `higher_tier` lacking a positive integer `min_prompt_tokens` or any of its
  four legs is not used. A prompt (uncached, cache read and cache write
  together) of `min_prompt_tokens` or more bills every leg of the call at the
  `higher_tier`. Any call the feed cannot price with certainty settles at the
  configured rate card and its usage record says why
  (`pricing_feed_fallback`). Readiness still requires the rate card.

Rollout: enable `routing_prices` only after the policy service accepts the
field (an older service refuses the request), and load a package re-exported
with `live_prices_affect_decisions` for the prices to change decisions. An
ineligible package still echoes `mode: package`, so the audit can be
verified before any decision moves.

## Decisions API v2 request members

`policy_service.decisions_v2` (off when absent) sends a cell's ARC Decisions
API v2 request members (pathfinder#5443). Each member is sent only when set,
so a cell without the block sends the v1 request byte for byte.

```yaml
decisions_v2:
  name: keen                # selection.operating_point.name (else "default")
  cost_weight: 1.0          # selection.operating_point.cost_weight
  tier_points:              # selection.tier_points: 1 to 8, distinct names
    - {name: swift, cost_weight: 2.0}
    - {name: sage, cost_weight: 0.0}
  response_fields: [price_per_mtok, thinking]
```

- **Name.** `name` labels the operating point the cell sends in place of
  `default`: a tier cell names its alias (`swift`, `sage`). It is a record
  label only and does not change scoring, so the echo check does not compare
  it. It must match `^[a-z][a-z0-9_-]{0,31}$`, and the selection log keeps it
  as `policy_operating_point.requested_name`.
- **V.** `cost_weight` is the cell's own V. Tiers are served one cell per
  tier, so a tier cell states its tier's V here and one package can serve
  every tier. The router reads no V from the client request. The service
  honours a V only within the bounds its package declares for the stage that
  chooses the model.
- **Echo.** A cell that sends a V requires the decision's
  `operating_point.cost_weight` to equal it. Another value, or a null, fails
  the turn (`policy_operating_point_mismatch`).
- **Tier picks.** The decision at each tier point is recorded, never served:
  the selection log carries `policy_tier_picks` (action, arm and model per
  configured tier; `policy_tier_picks_missing` names a configured tier the
  response has no pick for, and `policy_tier_picks_invalid` one whose pick
  names an action the cell does not bind or did not offer, or a model its
  binding does not serve) beside `policy_operating_point` (name,
  `point_id`, cost weight, and the configured `requested_name`).
- **Refusals.** `operating_point_out_of_bounds`,
  `operating_point_not_adjustable` and `tier_points_unsupported` are the
  cell's configuration against its package. They fail every turn as
  `policy_service_decisions_v2_refused`, which is `unavailable` to the caller,
  neither held nor undecidable, and the failure log names the reason.

Rollout: enable `decisions_v2` only after the policy service serves Decisions
API v2 (an older one refuses the members, and so every turn), and set a V
only for a package that declares operating bounds: every package before ARC
0.7.2 answers `operating_point_not_adjustable`.

## Stage-1 exploration

`policy_service.exploration` (off when absent) asks the policy service to
explore the Stage-1 model choice (pathfinder#5436). At a task, subagent or
compaction boundary, the service then draws the model from
`softmax(score / tau)` instead of taking the argmax. Effort is still the
argmax within the drawn model.

```yaml
exploration:
  tau: 0.02          # selection.exploration.tau; 0 asks for greedy
  tau_max: 0.05      # the calibration's tau_max; tau may not exceed it
  basis:             # how tau was calibrated
    r: 0.05
    package_sha256: <this cell's package_sha256>
    readout_pin: qwen3.8-27b-arc-encoder@depth8
  salt: vsr-explore-2026-10
eval_cell: false     # an eval or lab cell never explores
```

- **Load checks.** The loader refuses a tau outside `[0, tau_max]`, an `r`
  outside `[0, 1]`, a basis that names another package than the cell's
  `package_sha256`, and an empty or overlong readout pin or salt. Set
  `tau_max` to the service's `ARC_EXPLORE_TAU_MAX`.
- **Eligibility.** The member is sent only on a turn whose user may be
  explored: not in privacy mode, and not opted out of training-data
  collection. The router cannot know either fact itself, so the gateway
  states both: `x-rayline-privacy-mode: false` and
  `x-rayline-training-data: allowed`. They are read only with
  `trust_turn_signal_headers` and never forwarded upstream. Each fact fails
  closed, so an absent, empty or unknown value excludes the turn. An
  `eval_cell` and a side call are excluded too. The selection log names the
  reason (`exploration_excluded`: `privacy_mode`, `privacy_mode_unknown`,
  `training_data_opted_out`, `training_data_unknown`, `eval_cell`,
  `side_call` or `signal_headers_untrusted`).
- **Record.** A turn that sent the member logs `exploration_sent` and, when
  the service returned one, its whole `decision.stage_one.exploration`
  record as `exploration`. `llm_usage` carries `router_propensity`,
  `router_tau`, `router_explored` and `exploration_decision_key` (the
  record's `origin_decision_key` on a held turn). All four are null when the
  decision carried no record.
- **Relaxed held turns.** A relaxed call keeps no episode memory, so the
  service answers a held turn with a null `origin_decision_key`. A relaxed
  episode therefore records its latest Stage-1 boundary itself: the
  boundary's `decision_key` and `chosen` candidate, in the policy state.
  A held turn fills `exploration_decision_key` from that record only when
  the held candidate is the recorded `chosen`. Any other turn clears it: a
  decision without an exploration record (exploration off or the turn
  excluded), a held turn on another candidate, or a turn decided without
  the service. The key then stays null, never an older boundary's. A strict
  episode keeps no such record; its origin is the service's. An episode
  holding the record is stored as episode-state v9, whatever else it
  carries; every other episode keeps its earlier bytes, and a router that
  predates v9 refuses such a record by its schema.
- **Refusals.** `exploration_tau_exceeds_ceiling`,
  `exploration_basis_mismatch` and `exploration_unsupported_by_package` are
  the cell's config against its service and package. They fail every
  eligible turn as `policy_service_exploration_refused`, which is
  `unavailable` to the caller, neither held nor undecidable, and the failure
  log names the reason. The turn is not retried greedy, so a misconfigured
  cell shows up at once.

Rollout: until the gateway sends both eligibility headers, an exploration
cell explores no turn. Enable the block only for a service and package that
serve exploration, with the service's ceiling set.

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
`session_capacity`, also answer 429.

`unsupported_request` answers **400**. It says the service's projection cannot
represent something the request carries, so no arm is dialled: reporting it as
503 would blame a model service that was never called, invite a retry that is
refused identically every time, and -- because a 503 counts as provider
downtime in upstream monitoring -- report an outage that did not happen. A
client fixes it by changing the request, not by waiting.

The 400 holds on every entrypoint. What rides with it differs, because the
decision-only surfaces answer against their own error contract:

- On the inference path (ExtProc) the body is a typed `invalid_request_error`
  in the client's own wire format, with `x-vsr-failure-class: request_shape`.
- On `/v1/routes` and the legacy `/v1/route` it is that surface's own error
  shape and carries no failure class. Those callers get the non-retryable
  status, not the class: a decision-only client cannot currently tell this
  refusal from a malformed body, and should treat any 400 from these
  endpoints as "fix the request, do not retry unchanged".

Other service errors answer 503, `invalid_request` included: the router builds
most of the decision envelope itself, so that code can name a router-side
integration fault rather than anything the caller sent.

Their failure class comes from the contract's error codes only; any other
value is `service_error`.

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
- `x-rayline-privacy-mode: true | false` and
  `x-rayline-training-data: allowed | opted_out`, whether a turn may be
  explored (see "Stage-1 exploration").

A subagent the gateway keyed on its harness agent id (`agent`) is its own
conversation: a main turn of its own episode, with its own ledger, schedule
and decisions. Keyed any other way, or with no key source, it stays a side
call of the episode it arrived on, and its key source is logged as
`unknown`. The parent link is logged (hashed) on `rayline_arc_policy_turn`.
It never routes: it does not select, hold or narrow a subagent's actions
(pathfinder ADR 0117), and no derived hold reads it.

Set the option only when a gateway in front of the router sets these
headers, or strips them from client requests. Otherwise a client could mark
its own turns as side calls, which are not counted, or as compactions, or
claim consent to exploration. A gateway that already sets the turn-signal
headers must also set or strip the two exploration headers before an
exploration cell is enabled behind it. With the setting off, the headers are
ignored. Either way, none of them is
forwarded to a provider.

## Derived hold for two-stage packages

A two-stage package will not decide a mid-conversation turn cold. Such a turn
has an assistant message the episode never attributed and no episode memory
of its model, as when a conversation resumes after its episode cooled or was
evicted, or a subagent is handed its parent's history. The package refuses it
with 422 `selection_refused`, reason `stage_one_held_unknown`. VSR answers
only that refusal: it asks once more, with the offer narrowed to one model's
actions. That is the contract's caller hold, since every offered arm then
belongs to one model. A second refusal fails the turn as undecidable.
Single-stage packages never refuse this way, so their decisions are
unchanged.

The held model is the model that last served this episode (the selected
action's model, not its worker, which may serve several), which the episode
store keeps for seven days, written by each committed policy turn's own
episode commit, so records follow the commits' order. It continues a decision
the policy already made for this conversation, and it is the only source.

There is no fallback model. The parent session's model never holds a subagent
(ADR 0117), and the package manifest's `fallback_action_id`, the listing's
`fallback_model` and a cell's `derived_hold_model` are no longer read; a cell
that sets `derived_hold_model` is refused at load. With no record that names a
model the cell binds, or when the narrowed retry is refused too, the turn is
undecidable: it fails 422 with `x-vsr-failure-class: undecidable` and logs
`rayline_arc_undecidable` with `reason: stage_one_held_unknown` and the
`derived_source` it retried on (empty when none). No model is served in its
place.

A held model's offer is every action of that model, at every level. Exclusions still apply.
If this turn excludes the held model (vision, capability, an operator's
disable, a fallback exclusion), nothing is offered, and the turn fails as
`policy_no_available_action` without a second call.

VSR writes every record, and the gateway owns the episode keys. Nothing in
the request body is read, so a client cannot choose the held model. Each
derived hold logs `rayline_arc_derived_hold` with its source
(`session_record`) and model.

The packages listing (`rayline.arc.policy-packages.v1`) may state
`fallback_action_id` (a sha256 hex action id) and `fallback_model` on a
package, always together and never `null`. A listing that states only one,
or either as `null`, is refused as a decode error. A valid pair is not used.

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
- An episode whose thinking-control placer carries a fold or
  configuration_update rule set, or a refused call (ADR 0129), is stored as
  episode-state v5, with or without exclusions. A router that predates v5
  refuses such a record by its schema, so a rollback to one fails those
  episodes' next turn rather than misreading them.
- An episode keeps up to 64 thinking-control placers and 32 upstream
  records, so a package's bindings may put up to 64 control shapes on their
  workers' formats. One holding more than 16 of either is stored as
  episode-state v6, whatever else it carries; every other episode keeps its
  v3, v4 or v5 bytes. A v6 record may take up to 256 KiB, the others
  64 KiB. A router that predates v6 refuses such a record by its schema or
  its size.

## Open questions

- A context over encoder capacity is refused, never truncated. Should VSR
  serve a fallback action or a 503? It fails closed until the operator decides.
- Where each replica loads packages from, and how the alias registry is
  rolled out.
- Session caps once block features are resident, measured on L4 and RTX PRO
  6000.
