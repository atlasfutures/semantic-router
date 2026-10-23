# Standalone execution

`cmd/rayline-core` is the HTTP transport for the shared VSR implementation. Rayline
Local's `rld` owns the process, credentials, and harness gateway. The executor owns
protocol conversion, ARC decisions, exact deployment binding, thinking controls,
private append, and completed-execution evidence.

This integration targets the proposed `arc-execution-context.v1` Decisions
contract. A hosted cell must support that contract, enable the required cache
policy, and offer exactly the configured actions. An older cell fails visibly;
this binary does not select models from marginal probabilities or reroute after
an unsupported decision.

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

Unknown configuration fields and inconsistent package, control, registry, and
deployment identities are errors. Each action has one deployment in this proof.
The server-side offer must match these bindings; a client configuration cannot
silently narrow a larger hosted offer after selection.

`decisions` requires `url`, `auth_env`, `model`, `result` (`top`, `swift`, `sage`,
or `custom`), explicit `routing`, and `offer_sha256`. `routing.prices` requires
`price_sha256`; its returned digest and `exclusive` must match the request. The
executor supplies `routing.execution`. A harness cannot override routing through
its generation body.

Each deployment specifies `id`, `kind` (`local` or `cloud`), full inference `url`,
`protocol` (`anthropic.messages.v1`, `openai.chat.v1`, or `openai.responses.v1`),
`wire_model`, `registry_model`, `registry_provider`, `runtime_identity`,
`max_output_tokens`, and `token_units`. `auth_env` names an optional credential.
Local inference cannot receive the hosted Decisions credential, including under
an alternate environment-variable name. Unknown token units are an empty string;
they cannot support comparable prefix evidence.

A cloud deployment additionally requires approved `dispatch_profile_sha256` and
`issuer_sha256` values. OpenRouter deployments require `provider_pin`, for example
`{"only":["moonshotai/mxfp4"],"allow_fallbacks":false}`. Its endpoint must be
qualified for the selected control in the compiled registry and must match the
gateway's approved, priced profile. Optional `data_collection` and `zdr` members
must also match that profile. The gateway verifies provider-ready bytes and
returns identity receipts; it performs no second routing decision.

The optional `prompt_counter` qualifies one same-host llama.cpp deployment. Its
model, runtime, template and tokenizer checks are described in the module README.
Other external inference servers can execute without this counter, but their
current prefix counts remain unknown. Compatibility of a model name or HTTP
format does not qualify thinking controls or establish KV residency.
The optional `inband_system: "fold_before_unit_v1"` compatibility rule is
restricted to local Chat with the native default control and a matching runtime
identity suffix; see the [module README](../README.md#qualified-local-token-counting)
for its role semantics and refusal conditions.

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
The Decisions response must acknowledge every supplied worker and report its
effective cache policy before execution proceeds.

The reusable acceptance script lives in Rayline Local as
`scripts/test-hybrid-core.py`. It joins the real Rust/Go processes and can include
installed Claude Code and the exact cloud gateway. Its fixture mode deliberately
uses synthetic Decisions and model responses; it is separate from live ARC and
model-serving qualification.
