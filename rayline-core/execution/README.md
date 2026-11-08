# Standalone execution

`cmd/rayline-core` is the HTTP transport for the shared VSR implementation.
Raydock's `rld` owns the process, credentials, and harness gateway. The executor owns
protocol conversion, ARC decisions, exact deployment binding, thinking controls,
private append, and completed-execution evidence.

The default `decisions.api_mode: "current"` uses the deployed Decisions API and
keeps execution/cache evidence in this process. It sends no `routing.execution`
field or executed-model attribution, so it cannot establish semantic-boundary
model continuity. `"execution_context_v2"` (below) is the hybrid contract.
`"execution_context_v1"` explicitly selects the earlier advanced contract;
that mode requires the hosted cell to acknowledge the supplied evidence. There
is no automatic change of mode after an unsupported response. The core executes
the API's selected action rather than ranking marginal predictions locally.

## Execution context v2

`"execution_context_v2"` implements surfaces (B) and (C) of the hybrid v2 spec
(pathfinder `docs/plans/2026-10-24-decisions-execution-context-v2.md`). It is
opt-in. The core reports facts and never picks or ranks a model.

- **Policy.** `decisions.routing.policy` is required: `{"preset": ...}` with an
  optional `overrides.schedule` (`model`/`thinking` sets of `cache_boundaries`,
  `user_messages`, `all_turns`) and, for `arc-policy.prefer-local.v1` only,
  `overrides.selection.min_p_pass` in (0, 1). Served presets:
  `arc-policy.hybrid-default.v1`, `arc-policy.prefer-local.v1` (pathfinder
  #5913) and `arc-policy.package-default.v1`. At most 16 executable bindings fit
  `targets[]`. Raydock writes
  `arc-policy.hybrid-default.v1` when its user sets no override; the core never
  sends an absent policy in this mode. Allowed routing keys are `policy`,
  `v_value`, `prices`, `price_profile`, `models`, `excluded_models` and
  `signals`.
- **Registry.** Without an explicit `registry_path`, v2 executes with the
  embedded registry whatever registry the serving package names: controls are
  content-hashed, every bound action's control must be admitted by the
  embedded cells, and the selected control id is checked against the bound
  action. For a packaged profile `registry_sha256` then pins only the serving
  side (taken from the package when omitted); a `derived_actions` profile sets
  none. An explicit `registry_path`, and current/v1 modes,
  keep the exact pin.
- **Package-less profile.** `derived_actions` (v2 only) replaces
  `package_path`, `package_sha256` and `bindings`: each entry
  `{model, worker, deployment_id, bases?}` binds every registry-admitted control
  of that model on that deployment (optionally only these native bases), with
  ARC's v5 action identity (sha256 of JCS `{"control_id","model"}`). With no
  `registry_path`, `registry_sha256` may be omitted and the embedded registry is
  used. Such a profile pins no package: `expect` pins only the registry (and
  any offer), and the response names the package. Every derived action is a
  target, and the service refuses a target its serving package lacks
  (`execution_unknown_action`), so choose `bases` to match the served actions. Raydock builds one by default, so a user needs no profile
  file.
- **Envelope.** Each decision sends `routing.execution` version
  `arc-execution-context.v2`: `turn` (`id`, `attempt`, `epoch`, `retry_of`,
  `reconsult`), `event`, `last_served`, `history`, `hold` (null: the core has no
  current mandatory-hold source), `expect` and one `targets[]` entry per bound
  executable action. `turn.epoch` equals the `cache.context_epoch` string of
  evidence from the current epoch. `cache` is v1's worker evidence without the
  identity members; `capacity` is described below. Unknown is null.
- **Events.** Harnesses are unmodified: the primary source is what they
  already send, read by the core and labelled `basis: marker_inferred`
  (`wire_events.go`; replay tests over real Pi 0.80.2, Claude Code 2.1.280 and
  2.1.309, and Codex 0.154.0 captures in `testdata/wire_events`).
  - `task_start`: no assistant message yet; `subagent_start` instead when
    Claude Code's billing block claims `cc_is_subagent=true` or the request
    opens a new branch of a session that already has another branch.
  - `tool_result` versus `user_message`: the last message's tool results versus
    the user's own words. Injected `<system-reminder>`/project context, command
    expansions, compaction summaries and Codex's `<environment_context>` are not
    typed words (pathfinder's `typed_user_turn` rule), and a subagent's report
    after an `Agent`/`Task` call is tool output.
  - `sidecall`: a harness's own auxiliary call, recognized by its literal
    (Pi's summarization system prompt, Claude Code's summarize directive,
    title and suggestion prompts, Codex's compaction prompt, Hermes'
    iteration-cap summary request). It advances nothing, and one sent without
    a session header (Pi's summary call, Hermes' summary on Messages) runs
    statelessly instead of being refused.
  - `compaction`: a request that rewrites the prefix is compared with the last
    request this branch saw (committed or not). A harness summary literal (Pi,
    Claude Code, Codex) or, without one, a new first user message replacing at
    least two messages and followed by a retained run that ends where the
    previous request ended, is a compaction with `summary_message`,
    `retained_from` (null without a retained run) and `new: true`; the retained
    messages keep their attribution at their new positions. Any other rewrite
    is a new epoch with no boundary.
  - Anything else is `basis: unknown` with a null kind, which holds.

  A harness may optionally state the event itself, in an `X-Rayline-Event`
  JSON header or a top-level `rayline_event` member that the core removes
  before decoding, Decisions and dispatch (both at once are refused). An
  explicit event takes precedence. No default or acceptance path depends on
  it.
- **Continuity.** A prefix change starts a new epoch and clears
  transcript-dependent evidence but keeps the last successfully executed action,
  sent as `last_served` even when that target is not offered or does not fit. Current and v1 modes are unchanged: a prefix
  change clears the held action, and request bytes are untouched: no event channel is read
  in those modes.
- **Capacity.** For each local action whose deployment has a qualified prompt
  counter, the core prepares the request exactly as dispatch would (tools,
  history, private append) and counts it with the runtime's own tokenizer:
  `required` = that count + the prepared output limit + the deployment's
  optional, nonnegative `capacity_margin_tokens`; `available` = its configured
  `context_window`, with `available_source: "configured"` (llama.cpp's
  per-slot allocation is not yet observed); `units` = its `token_units`; `provenance: "local_measured"`. A
  counter that fails reports that target `fits: false` with
  `reason: "runtime_unavailable"` instead of refusing the turn. A **local**
  target without a registered counter is sized from the same prepared request:
  - **Qwen3.8-27B** (`registry_model: "qwen/qwen3.8-27b"`, Chat wire; the
    Raydock default) is rendered with the model's chat template as
    vLLM's Chat path feeds it (`qwen_capacity.go`: arguments and tools
    re-spelt by Python's `tojson`, floats by `repr`; each tool as
    `model_dump()` spells it, declared fields first with null
    `description`/`parameters`; text parts joined with "\n" in the rendered
    text, and a plain concatenation charged too where it costs more) and
    counted in-process with the embedded tokenizer (`tokencount`, pure Go:
    the snapshot's `merges.txt`, its added tokens, NFC and its
    pre-tokenizer), `provenance: "local_measured"`. The rendering takes the
    template's most expensive branch where the request does not pin one and
    adds 16 tokens, 2 per message and 8 per tool. A message the counter can
    only bound by its bytes (text x/text cannot normalize as the reference
    does: a supplementary character in non-NFC text, or one of the combining
    marks the reference's older Unicode tables lack) makes the fact
    `caller_estimated`. This assumes the runtime serves Qwen3.8-27B with that
    template; the core does not verify the served model. An operator serving
    Qwen with a custom template should register a llama.cpp counter instead.
  - **Any other model** gets the UTF-8 byte count of a worst-case
    re-serialization of the prepared request (values re-encoded with
    `indent=4`, Python separators and float `repr`, non-ASCII and `<`, `>`,
    `&` escaped, each tool call's arguments reloaded and re-encoded beside
    it), an upper bound for byte-level BPE (every token covers at least one
    byte), raised by `headroom_ratio` and 512 tokens for a template preamble,
    `provenance: "caller_bounded"`. The policy contract admits a bounded
    `fits: true` as a fit. Two template assumptions are not checked: the
    preamble is at most 512 tokens, and the template renders each part of
    the request at most once. A template that repeats request content (a
    schema key written as an opening and a closing tag, the first tool name
    reused for every tool result) can exceed the bound. The backstop is the
    runtime: on a `"refuses"` deployment an over-window prompt is refused and
    the core consults again; on a `"truncates"` one it is cut.
  - An inline base64 image is charged `media_tokens` instead of its bytes, and
    a fact that includes one is `caller_estimated`. So is any byte bound over
    a body the core does not fully understand: `caller_bounded` needs every
    top-level member, message member, input item, content part and tool type
    to be on the deployment protocol's allowlist (`capacity_bounded.go`:
    text, tool calls and results, reasoning, sampling and output limits).
    Media, content the body only references (`file_id`, `item_reference`,
    `previous_response_id`), a request's own `chat_template` or
    `role_mapping`, a server tool and any unknown or runtime-specific member
    fall to `caller_estimated`.
  - **`context_overflow`** (local deployments, `"refuses"` or
    `"truncates"`, default `"truncates"`) states what the runtime does with a
    prompt over its window. On a `"truncates"` deployment (Ollama,
    `mlx_lm.server`) an estimated or bounded shortfall always stands, so such
    a prompt is never sent to be silently cut; if that leaves every target
    short, the turn gets the compaction 400. On a `"refuses"` deployment
    (llama.cpp, vLLM), when every executable target falls short, its
    estimated and bounded facts are sent as unknown for all its actions: the
    turn dispatches there and the runtime's own context refusal decides.
    A missing `context_overflow` means `"truncates"`; Raydock's CLI
    sets `"refuses"` when it detects llama.cpp. The rule is part of the
    all-short decision itself: an estimated or bounded shortfall on a
    `"refuses"` deployment counts as unknown there, so the turn start, a
    re-consult, the re-consult limit and a service refusal all agree.
  - **A runtime context refusal covers its whole deployment.** It is recorded
    as `provider_reported` for every action bound to that deployment, so no
    sibling control is dispatched to be refused again, and when that leaves no
    target the core answers the compaction 400 at once instead of spending
    re-consults. This is conservative: a near-edge turn is handed off rather
    than retried with a sibling control whose smaller output allowance might
    have fit.
  The policy contract admits `fits: false` from any provenance and reads an
  estimated `fits: true` as unknown, so an over-window local target is excluded
  before anything is sent to it. Facts are recomputed every turn: once the
  history outgrows the local window, local stays excluded. A **cloud** target
  without a counter sends `capacity: null`; its provider refuses an oversized
  prompt itself, and that refusal reaches the caller translated, text intact.
- **Runtime context refusals.** A local runtime's refusal before any output,
  llama.cpp's or an OpenAI-compatible one (`context_length_exceeded` or
  "maximum context length" in the error's code, type or message: vLLM, MLX, a
  custom endpoint), re-consults with `reconsult: "capacity"`.
- **No deployment fits.** When every executable target has a
  `context_capacity` shortfall that stands (see `context_overflow`) and none
  is unknown, the
  core answers `400 context_exceeds_all_deployments` in the caller's wire
  format before Decisions or any runtime is contacted. A target without a fact
  (a cloud target without a counter) is never assumed not to fit, so a default
  hybrid profile is never refused on an estimate. The message starts with
  `prompt is too long:` (Anthropic's wording, on which Claude Code compacts; the
  Messages wire carries no code), gives the request's size and its basis
  (estimated, measured or reported) when that exceeds the largest configured
  "maximum context length", names that window and its model, and asks for the
  conversation to be compacted or shortened. A service refusal `execution_no_feasible_target` (or a selection
  of an infeasible target) under the same condition gets the same answer.
- **Provider context refusals.** A cloud provider's context-length refusal
  (HTTP 400/413 naming `context_length_exceeded`, "maximum context length",
  "prompt is too long" or "input is too long" in the error's code, type or
  message), received before any output, is forwarded translated; on the
  Messages wire its message is prefixed with `prompt is too long:` unless it
  already starts so.
- **Pre-dispatch check and veto.** The final prepared request is checked again
  before dispatch. If it does not fit the selected target, or the local runtime
  refuses it for context before any output, the core re-consults once per cause
  with the same `turn.id` and event, `attempt + 1`, `retry_of` = the vetoed
  route, `reconsult: "veto"` or `"capacity"`, and that target's capacity
  `fits: false`. The service reselects; a service that reselects an infeasible
  target fails visibly (`execution_no_feasible_target`). A refusal recorded for the turn is kept, so a
  client retry of the same request never re-dispatches the refused target.
- **Response.** The selected result must carry a `receipt` naming the selected
  action, control and applied policy, with a known `decided_by` per dimension
  (`decided_by.model` may also be `held_unavailable`, `prefer_local_threshold` or
  `utility_fallback` (prefer-local), or null for a one-stage
  package) and an `excluded`
  entry, with the same reason, for every target the core reported as not
  fitting. `routing.execution` must echo `status: "applied"`, the version and the
  exact turn, and `matched` for each sent identity pin when `pinned` is present;
  `routing.policy` must echo `status: "applied"` and the preset. An
  `unsupported` echo fails visibly; there is no fallback to another mode. Responses add `X-Rayline-Decided-By` and
  `X-Rayline-Event` headers, and logs record the receipt and capacity facts.

## Configuration

Pass `--config /absolute/path/hybrid.json`. Configuration version 1 contains:

| Member | Meaning |
| --- | --- |
| `contract_version` | `1` |
| `package_path`, `package_sha256` | Compatible policy action manifest and SHA-256 of its exact bytes. The manifest is provisioned separately; do not commit private packages. |
| `registry_path`, `registry_sha256` | Optional compiled thinking registry path and mandatory SHA-256. An omitted path selects the embedded registry. |
| `allow_experimental_controls` | Explicit admission of registry controls marked experimental. |
| `decisions` | URL, credential environment name, ARC model, selected result, routing preferences, offer digest and optional resolved-price digest. |
| `deployments` | Concrete local or cloud endpoints and qualified serving identities. |
| `bindings` | Exact package `action_id`, Decisions `worker`, and `deployment_id` triples. |
| `test_alias` | Optional explicit test-only selected-model to execution-action substitution, described below. |
| `max_generation_attempts` | Optional nonnegative process-wide inference-attempt cap; `0` disables it. Failed or partial calls consume an attempt. |

Unknown configuration fields and inconsistent package, control, registry, and
deployment identities are errors. Each action has one deployment in this proof.
The server-side offer must match these bindings; a client configuration cannot
silently narrow a larger hosted offer after selection.

`decisions` requires `url` (the gateway's `/v1/routes`), `auth_env`, `model`,
`result` (`top`, `swift`, `sage`, or `custom`), and explicit `routing`.
`offer_sha256` is optional in current mode and required in advanced mode. The
`routing` block is sent as configured and follows the Routes API V1 contract
(pathfinder#6159): `v_value`, `prices`, `price_profile`, `models`,
`excluded_models`, `signals` and `policy` (required in v2, forwarded in v1,
refused in current mode). There is no `exclusive`
member: the gateway resolves the candidates from `models` and
`excluded_models` (every offered model when both are absent) and prices each
of them leg by leg from inline `prices`, then the profile, then the catalog.
A profile with `exclusive`, `price_profile_id` or `price_profile_version` is
refused at startup in every API mode, naming the V1 replacement; so is any
other member the API does not take (v1 and v2 forward `policy`).

`routing.prices` maps exact model ids to any subset of `input`, `cache_read`,
`cache_write` and `output`, in USD per million tokens. `routing.price_profile`
is `{"id": ..., "version"?: ...}`: an authorized saved profile through the
gateway, combinable with inline `prices` (which win per model and leg). Omit
the version to resolve the current version on each decision, or pin a positive
version for a repeatable run.

**Price acknowledgement.** When `prices` or `price_profile` is sent, the
answer's `routing.prices` must be `applied` and carry the resolved table:
every candidate's four legs with each leg's source (`caller`, `profile` or
`catalog`), the `unused` priced models that are not candidates, and
`price_sha256`. The adapter recomputes that digest, sha256 of the RFC 8785
form of `{"schema": "arc-route-prices.v1", "prices": <table>}` where the
table holds only each candidate's four legs (the `source` members are not
hashed), and requires the echoed digest to equal it. Every caller-priced
model must either be a candidate whose caller legs are resolved at the
caller's price with source `caller`, or be listed in `unused`. The reverse
holds too: a leg may carry source `caller` only where this profile priced
that model and leg, at that value, and source `profile` only when a
`price_profile` was sent; sources are outside the digest and set the
receipt's basis, so an unclaimed source is refused. The selected
model must be one of the priced candidates (and one of `models.resolved`
when `models` was sent), never a model listed `unused`. The selected result's `prices` receipt
must name the table's basis (`catalog`, `caller`, `profile` or `mixed`, over
every leg of the table) and the same digest. A sent `models` or
`excluded_models` must be echoed `applied` with the entries sent, and the
priced candidates must be exactly the resolved ones. With neither `prices` nor
`price_profile`, the receipt must be `catalog` with a null digest and no
`routing.prices` echo. A profile additionally requires an applied
ID/version/content-hash receipt with a known `schema`; profile content hashes
and price digests use different encodings and are not interchangeable. The
resolved profile identity is recorded with each selection.

`price_sha256` is optional. When set, it pins the resolved price world: the
echoed `routing.prices.price_sha256` must equal it. That digest covers every
candidate's four resolved legs, so a client can precompute it only when it
prices every candidate's four legs itself (and confines the candidates with
`models`); a partially priced or catalog-filled table is acknowledged leg by
leg instead, with the pin omitted. The previous contract's
`arc-caller-prices.v1` digest over `{schema, exclusive, prices}` is never
returned to the public caller: a pin equal to that digest of the configured
prices is refused at startup with the migration named, and any other
mismatching pin fails the decision.

A configured
`v_value` requires `result: "custom"`, a returned custom result, and an applied
value acknowledgement; the selected result must also report that value and its
reciprocal cost weight. Missing custom results, mismatched identities or prices,
and unapplied routing keys fail visibly. A harness cannot override these settings
through its generation body.

**Typed signals** (pathfinder `docs/plans/2026-10-25-decisions-topic-signal-contract.md`)
are caller metadata and change no route. In every mode a generation request may
carry `"routing": {"signals": ["topic.operation", ...]}` and nothing else under
`routing`; the core removes that member before dispatch, sends the names to
Decisions unchanged (they replace any configured `routing.signals`), and returns
the response's `signals` block as compact JSON in the `x-vsr-signals` response
header, as cloud VSR does on generation turns. The service owns the vocabulary
and refuses unknown or unadmitted names. A malformed list is `400
invalid_routing_signals`; a request without `routing` is untouched, and no
header is set.

For an isolated development cell, `decisions.cell_tag` sends the gateway's
`X-Rayline-Cell` selection header and requires the same tag in its response.
Use a gateway/model that permits lab selection (for example, the dev gateway's
`rayline-router` with VSR routing enabled). A fixed ARC tier can ignore this
header; the adapter refuses that mismatch. Omitting the tag uses normal gateway
selection. The setting never changes gateway configuration or standing traffic.

The deployed top-only response may provide `policy_action_id`, model and worker
without package or price identity. Current mode accepts that response only with
`result: "top"`, no `offer_sha256` expectation, and a `routing` that is empty
or names candidates only: `models` and `excluded_models` are acknowledged
from their echoes (applied, the entries sent, the selected model among the
resolved candidates), while `prices` or `price_profile` need a price receipt
this response lacks and are refused. It binds the returned action and exact model
aliases to the configured manifest and records `identity_source:
"configured_manifest"`; it does not claim the server attested package, registry,
offer or prices. Responses containing the full identity must match the configured
pins and are recorded with `identity_source: "response"`.

Current preset/custom results also name the action, model and worker without a
package identity. The adapter binds those to the configured manifest and records
`identity_source: "configured_manifest"`; a configured `offer_sha256` still
requires a returned identity. Each selected result's own `prices` must match the
applied routing receipt, independently of the top-level decision. These result
prices contain `basis` and `price_sha256`, the price world described above.
This does not change the requested result or relax the advanced mode's
identity requirements.

Each deployment specifies `id`, `kind` (`local` or `cloud`), full inference `url`,
`protocol` (`anthropic.messages.v1`, `openai.chat.v1`, or `openai.responses.v1`),
`wire_model`, `registry_model`, `registry_provider`, `runtime_identity`,
`max_output_tokens`, `context_window`, `input_modalities`, and `token_units`.
Context and output limits must be positive, with output no larger than context;
input modalities explicitly include `text` and may include `image`. Authenticated
health reports the minimum limits and modality intersection of possible execution
targets, and each limit is enforced against the deployment a turn lands on, not
against the reported minimum. A caller's requested output allowance is not
clamped, except on a deployment that sets `clamp_output_allowance` (optional,
default false), where an allowance over that deployment's `max_output_tokens`
is lowered to it and logged with the stated and sent figures: a harness states
one allowance for a session whose deployments do not share a limit and cannot
be told a smaller one per arm. A turn whose admitted thinking budget leaves no
strict room below the limit is refused rather than clamped, since Messages
counts thinking inside `max_tokens`.
`auth_env` names an optional credential.
`capacity_estimate` (v2, optional, local deployments) tunes the sizing of a
local deployment without a registered counter: `{"headroom_ratio": 0.05,
"media_tokens": 8192}` are the defaults for absent members (limits:
`headroom_ratio` in [0, 10], `media_tokens` >= 0). `headroom_ratio` applies
only to the byte bound; `media_tokens` (a Retina screenshot is about 6.6k
Qwen-VL tokens) to both. Each distinct prepared body is sized once per
deployment and turn, and a configured Qwen3.8 deployment builds its counter
at startup.
Local inference cannot receive the hosted Decisions credential, including under
an alternate environment-variable name. Unknown token units are an empty string;
they cannot support comparable prefix evidence.

A cloud deployment's default `cloud_transport: "exact_dispatch"` requires approved `dispatch_profile_sha256` and
`issuer_sha256` values. OpenRouter deployments require `provider_pin`, for example
`{"only":["moonshotai/mxfp4"],"allow_fallbacks":false}`. Its endpoint must be
qualified for the selected control in the compiled registry and must match the
gateway's approved, priced profile. Optional `data_collection` and `zdr` members
must also match that profile. The gateway verifies provider-ready bytes and
returns identity receipts; it performs no second routing decision.

The explicit `cloud_transport: "model_pin"` uses an existing model router's
OpenRouter Chat or Messages endpoint with its concrete qualified model. It checks
the returned
model and pin receipt, without claiming an exact provider, account or action
receipt. Provider/profile/issuer pins are forbidden in this mode and opaque
reasoning remains unqualified. Only the transport's supported native controls
and registry-admitted private append are accepted; unsupported request shapes
fail before dispatch.

The optional `prompt_counter` qualifies one same-host llama.cpp deployment. Its
model, runtime, template and tokenizer checks are described in the module README.
Other external inference servers can execute without this counter, but their
current prefix counts remain unknown. Compatibility of a model name or HTTP
format does not qualify thinking controls or establish KV residency.
The optional `inband_system: "fold_before_unit_v1"` compatibility rule is
restricted to local Chat with the native default control and a matching runtime
identity suffix; see the [module README](../README.md#qualified-local-token-counting)
for its role semantics and refusal conditions.

An explicit `test_alias` such as `{"selected_model":"gpt-6-luna",
"execution_action_id":"<neutral Qwen package action ID>"}` can exercise local
inference when the live offer does not select Qwen. Both source and target remain
independent package bindings. A source that never executes may omit
`deployment_id` and instead supply an explicit `reported_model`, such as
`"openai/gpt-6-luna"`, in its binding. That selection-only binding still requires
an exact manifest action, worker and registry-known control; it makes no source
transport-admission claim and cannot start without its explicit alias.
The target must be qualified local Qwen 27B with
the native default control and a prompt counter; the source's steering is never
sent to Qwen. This option is off by default and allowed only in current API mode.
Logs, response headers and local cache records distinguish selected from executed
model/action/control identities. It proves execution plumbing, not ARC's Qwen
selection quality. It does not alter caller prices to make the source model look
free, and aliased cache evidence is not sent to Decisions.

## Process contract

Build from the module directory:

```sh
CGO_ENABLED=0 go build -o /absolute/output/rayline-core ./cmd/rayline-core
```

`rld` supplies `RAYLINE_EXEC_AUTH_TOKEN`, invokes `--listen 127.0.0.1:0`, and keeps
stdin open with `RAYLINE_EXEC_PARENT_PIPE=1`. The core validates its configuration
and configured counters before emitting one JSON startup line. All logs go to
stderr. Authenticated `/healthz` reports the contract, instance, and package
identity. Startup failure never publishes readiness.

The parent pipe closing, SIGINT, or SIGTERM cancels active requests and shuts down
the HTTP server. The process neither starts nor stops an external model server.
Credentials live in the environment; startup metadata and command-line arguments
contain no keys. Runtime readiness does not substitute for the live hosted
Decisions and cloud execution acceptance test.

## State and cache contract

The executor serializes a branch through decision, dispatch, and settlement.
Failed requests retain a decision for an identical retry only before output is
observed. Once output was delivered, an identical request is refused rather than
silently regenerated. A successful response commits its placer, provenance,
attribution, held action, usage and completion clock together. Errors, truncated
streams, refusals, and decision-only consults do not mark a worker warm.

Session state is in memory. A new process has new credentials and a new context
epoch. Historical assistant/tool input for an unknown session or branch is
refused because its hidden instruction ledger cannot be reconstructed. Durable
resume and inherited subagent histories require a later state-transfer contract.
At capacity, new sessions fail visibly instead of evicting an active ledger.

A changed instruction, tool definition or historical message starts a new cache
context epoch and clears transcript-dependent placement, attribution and worker
evidence. In current and v1 modes it also clears the held action; only
`execution_context_v2` keeps it, as `last_served`.

Cache evidence describes each previously completed deployment, including workers
that were switched away from. It carries completion age, provider usage, declared
TTL when applicable, context epoch, and qualified current prefix overlap when
available. Token overlap alone does not establish residency. Unknown local
residency and unknown lifetime remain null; inference uptime is not a cache TTL.
In current API mode this cache state remains local. In advanced mode the Decisions
response must acknowledge every supplied worker and report its effective cache
policy before execution proceeds.

Each `execution completed` log includes a prompt-free `cache` object: the count
of completed deployment records, current call usage and completion time, and
retained worker records with their original clocks, declared TTL and qualified
prefix evidence. Missing counts, TTL, prefix measurements and residency remain
null. These record counts do not identify resident caches; failures, refusals and
partial streams emit no completion record.

The live Pi tool-loop proof demonstrates retained completion clocks across model
switches and provider-reported cache reads. Its tool-result-ended requests leave
`current_prefix` null: the local counter currently qualifies a carried boundary
only at a new user turn. The run therefore does not prove tool-turn costing
overlap, retained KV residency, or cache-aware Decisions selection; current API
mode keeps this execution evidence local.

The reusable acceptance script lives in Raydock as
`scripts/test-hybrid-core.py`. It joins the real Rust/Go processes and can include
installed Claude Code and the exact cloud gateway. Its fixture mode deliberately
uses synthetic Decisions and model responses; it is separate from live ARC and
model-serving qualification.

Live acceptance can set a positive `max_generation_attempts` and a runner wall
timeout. The core atomically reserves an attempt immediately before inference;
after exhaustion it returns `generation_limit_reached` without another inference
call. Changing sessions or rewriting history cannot reset the cap. This bounds
calls, not dollar cost; output allowance and provider pricing still matter.
