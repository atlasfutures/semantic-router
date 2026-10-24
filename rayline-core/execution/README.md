# Standalone execution

`cmd/rayline-core` is the HTTP transport for the shared VSR implementation. Rayline
Local's `rld` owns the process, credentials, and harness gateway. The executor owns
protocol conversion, ARC decisions, exact deployment binding, thinking controls,
private append, and completed-execution evidence.

The default `decisions.api_mode: "current"` uses the deployed Decisions API and
keeps execution/cache evidence in this process. It sends no `routing.execution`
field. `"execution_context_v1"` explicitly selects the proposed advanced contract;
that mode requires the hosted cell to acknowledge the supplied evidence. There
is no automatic change of mode after an unsupported response. The core executes
the API's selected action rather than ranking marginal predictions locally.

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
