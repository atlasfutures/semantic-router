# Standalone execution

`cmd/rayline-core` is the HTTP transport for the shared VSR implementation. Rayline
Local's `rld` owns the process, credentials, and harness gateway. The executor owns
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
  `targets[]`. Rayline Local writes
  `arc-policy.hybrid-default.v1` when its user sets no override; the core never
  sends an absent policy in this mode. Allowed routing keys are `policy`,
  `value_usd`, `prices`, `exclusive`, `signals` and the price-profile keys.
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
  (`execution_unknown_action`), so choose `bases` to match the served actions. Rayline Local builds one by default, so a user needs no profile
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
    title and suggestion prompts, Codex's compaction prompt). It advances
    nothing, and one sent without a session header (Pi's summary call) runs
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
  `required` = that count + the prepared output limit + the deployment's optional
  `capacity_margin_tokens`; `available` = its configured `context_window`, with
  `available_source: "configured"` (llama.cpp's per-slot allocation is not yet
  observed); `units` = its `token_units`; `provenance: "local_measured"`. A
  counter that fails reports that target `fits: false` with
  `reason: "runtime_unavailable"` instead of refusing the turn. A **local**
  target without a counter (a custom endpoint, Ollama, MLX) is estimated from the
  same prepared request with `provenance: "caller_estimated"` (never
  `local_measured`) and `units` = its `token_units` or null; see
  `capacity_estimate` below. The policy contract admits an estimated
  `fits: false` as an exclusion and reads an estimated `fits: true` as unknown,
  so an over-window local target is excluded before anything is sent to it.
  Facts are recomputed every turn: once the history outgrows the local window,
  local stays excluded. A **cloud** target without a counter sends
  `capacity: null`; its provider refuses an oversized prompt itself, and that
  refusal reaches the caller translated, text intact.
- **No deployment fits.** Only when every executable target has a measured or
  provider-reported `context_capacity` shortfall does the core answer `400
  context_exceeds_all_deployments` in the caller's wire format, before
  Decisions or any runtime is contacted. Estimated facts never refuse a turn.
  The message starts with `prompt is too long:` (Anthropic's wording, on which
  Claude Code compacts; the Messages wire carries no code), gives the measured
  size and the largest "maximum context length" and its model, and asks for the
  conversation to be compacted or shortened. A service refusal
  `execution_no_feasible_target` (or a selection of an infeasible target) under
  the same condition gets the same answer.
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
| `decisions` | URL, credential environment name, ARC model, selected result, routing preferences, offer digest and optional caller-price digest. |
| `deployments` | Concrete local or cloud endpoints and qualified serving identities. |
| `bindings` | Exact package `action_id`, Decisions `worker`, and `deployment_id` triples. |
| `test_alias` | Optional explicit test-only selected-model to execution-action substitution, described below. |
| `max_generation_attempts` | Optional nonnegative process-wide inference-attempt cap; `0` disables it. Failed or partial calls consume an attempt. |

Unknown configuration fields and inconsistent package, control, registry, and
deployment identities are errors. Each action has one deployment in this proof.
The server-side offer must match these bindings; a client configuration cannot
silently narrow a larger hosted offer after selection.

`decisions` requires `url`, `auth_env`, `model`, `result` (`top`, `swift`, `sage`,
or `custom`), and explicit `routing`. `offer_sha256` is optional in current mode
and required in advanced mode. `routing.prices` requires
`price_sha256`; its returned digest and `exclusive` must match the request. The
current adapter supports `value_usd`, `prices`, `exclusive`, `price_profile_id`,
and `price_profile_version`. A profile ID references an authorized saved profile
through the gateway; it cannot be combined with inline `prices`. Omit the version
to resolve the current version on each decision, or pin a positive version for a
repeatable run. A profile may also pin `price_sha256`. The adapter requires an
applied profile ID/version/content-hash receipt and matches the resolved price
digest to the selected result's price receipt and exclusivity to the routing
price echo. Profile content hashes
and canonical price digests use different encodings and are not interchangeable.
The resolved profile identity is recorded with each selection. A configured
`value_usd` requires `result: "custom"`, a returned custom result, and an applied
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
`result: "top"`, empty `routing`, and no `offer_sha256` expectation. It binds the returned action and exact model
aliases to the configured manifest and records `identity_source:
"configured_manifest"`; it does not claim the server attested package, registry,
offer or prices. Responses containing the full identity must match the configured
pins and are recorded with `identity_source: "response"`.

Current preset/custom results also name the action, model and worker without a
package identity. The adapter binds those to the configured manifest and records
`identity_source: "configured_manifest"`; a configured `offer_sha256` still
requires a returned identity. Each selected result's own `prices` must match the
applied routing receipt, independently of the top-level decision. These result
prices contain `basis` and `price_sha256`; `exclusive` belongs to the routing
echo. This does not change the requested result or relax the advanced mode's
identity requirements.

Each deployment specifies `id`, `kind` (`local` or `cloud`), full inference `url`,
`protocol` (`anthropic.messages.v1`, `openai.chat.v1`, or `openai.responses.v1`),
`wire_model`, `registry_model`, `registry_provider`, `runtime_identity`,
`max_output_tokens`, `context_window`, `input_modalities`, and `token_units`.
Context and output limits must be positive, with output no larger than context;
input modalities explicitly include `text` and may include `image`. Authenticated
health reports the minimum limits and modality intersection of possible execution
targets. No harness gateway clamps the caller's requested output allowance.
`auth_env` names an optional credential.
`capacity_estimate` (v2, optional, local deployments) tunes the estimate used
when the deployment has no prompt counter: `{"headroom_ratio": 0.1,
"media_tokens": 8192}` are the defaults for absent members (limits:
`headroom_ratio` in [0, 10], `media_tokens` >= 0). The estimate charges by
character class over the decoded request text, because byte-level BPE spends
about a token per character on numbers, hashes and base64 but 3.5-4 bytes per
token on prose: an alphanumeric run with a digit, or a long mixed-case run, is
1 token per character; a word is 1 + length/8; a punctuation run is 1 + half
its remaining length; whitespace beyond one space is 1 per 4; a CJK character
is 1, other non-ASCII symbols their UTF-8 byte count; an inline base64 image is
`media_tokens` (a Retina screenshot is about 6.6k Qwen-VL tokens). Against the
Qwen3.8-27B tokenizer (`testdata/capacity`) it never under-counts and
over-counts numeric CSV by 1.11x, float logs 1.17x, hashes 1.22x, base64
1.46x, uv.lock 1.29x, prose 1.49x, code 1.53x, JSON 1.51x and multilingual
text 1.70x (with headroom). Raise `headroom_ratio` for a runtime whose
tokenizer is denser than Qwen's.
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

The reusable acceptance script lives in Rayline Local as
`scripts/test-hybrid-core.py`. It joins the real Rust/Go processes and can include
installed Claude Code and the exact cloud gateway. Its fixture mode deliberately
uses synthetic Decisions and model responses; it is separate from live ARC and
model-serving qualification.

Live acceptance can set a positive `max_generation_attempts` and a runner wall
timeout. The core atomically reserves an attempt immediately before inference;
after exhaustion it returns `generation_limit_reached` without another inference
call. Changing sessions or rewriting history cannot reset the cap. This bounds
calls, not dollar cost; output allowance and provider pricing still matter.
