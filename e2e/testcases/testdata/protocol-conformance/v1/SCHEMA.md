# Protocol Conformance Fixture Schema v1

This document freezes the on-disk contract for the protocol-conformance corpus
described by PL-0042, the data-plane protocol-conformance plan, under upstream
Epic vllm-project/semantic-router#1138 and conformance issue #2358.

The reader is `e2e/pkg/conformance`. It parses every file named here, validates
the rules named here, and rejects anything else. A change to this document is a
change to that package and to every authored case.

Scope of v1:

- DPC-102 owns this schema, the loader, and the comparators.
- DPC-101 builds the programmable provider fixture against the replay script in
  [Replay script](#replay-script). That format is complete: DPC-101 does not
  need a schema change.
- DPC-103 wires a `protocol-conformance` profile onto the loader.
- DPC-104 authors the six `seed-*` case directories.

## Version root

```text
e2e/testcases/testdata/protocol-conformance/v1/
  SCHEMA.md
  cases.yaml
  <case-id>/
    client-request.json
    expected-provider-request.json
    provider-response.json | provider-response.sse
    expected-client-response.json | expected-client-response.sse
    compare.yaml                          # optional
    replay.yaml                           # optional until DPC-101 lands
```

A new version is a new sibling directory (`v2/`). v1 is never edited in place
once cases are promoted into a required gate.

## cases.yaml

`cases.yaml` is the frozen DPC-003 inventory: 31 cases, 6 in the `first-six`
tranche and 25 in `firebase-derived-deferred`. It carries `schema_version:
dpc-003-inventory-v1alpha1`, which is the only version the loader accepts.

The file is the contract vocabulary. It declares, per case:

| Field | Meaning |
| --- | --- |
| `id` | Unique case identifier and the case directory name |
| `tranche` | Promotion tranche |
| `contract` | The externally visible behavior in one sentence |
| `client` | Inbound protocol, path, and mode (`buffered`, `streaming`, `sequence`) |
| `provider` | Selected provider wire protocol and dialect profile |
| `mutation_mode` | `patch`, `translate`, or `reject` |
| `features` | Feature axes the case exercises |
| `synthetic_shape` | What a fixture author must generate, in prose |
| `expectation` | Both comparators, allowed patches, invariants, fidelity ledger, loss, dispatch count |
| `provenance` | Origin, normative sources, private behavior reference |
| `ownership` | Primary Workgroup, reviewers, upstream issues |

A case with no directory on disk is contract-only: the loader returns it with
`Case.Loaded() == false`, and the comparators refuse to run on it. That is the
state of all 31 cases until DPC-104 authors fixtures.

## Case directory

A directory that exists must be complete. A partially authored case is a load
error, not a skipped case.

| File | Required when | Contents |
| --- | --- | --- |
| `client-request.json` | Always | The request body sent to the router |
| `expected-provider-request.json` | `provider_request` is not `reject` | The body the provider fixture must observe |
| `provider-response.json` or `.sse` | `provider_request` is not `reject` | What the provider fixture replays |
| `expected-client-response.json` or `.sse` | `client_response` is not `reject` | What the router must return |
| `compare.yaml` | Optional | Comparator tuning the authored bytes cannot express |
| `replay.yaml` | Optional | The provider fixture program for this case |

The extension is the encoding contract: `.json` is a buffered body, `.sse` is a
stream compared as parsed events. Exactly one of the pair may exist. Declaring
both is a load error, because it hides which encoding the case asserts.

A `reject` case never reaches the provider, so it has no provider artifacts and
no `expected-client-response`. Its expected rejection lives in `compare.yaml`.

## compare.yaml

Everything a comparator needs that `cases.yaml` already carries is read from
`cases.yaml`. `compare.yaml` adds only what depends on the authored bytes.

```yaml
# Extra exact-except exclusions, on top of expectation.allowed_patches.
exclude_extra: [/id]

# JSON pointers whose value is nondeterministic under semantic and reject modes.
volatile: [/created, /message/id]

# The expected rejection. Required when a boundary is `reject`.
reject_status: 400
reject_headers: {content-type: application/json}
reject_body:
  error: {type: invalid_request_error, code: unsupported_tool_result}
```

An entry in `exclude_extra` or `allowed_patches` that starts with `/` is an RFC
6901 JSON Pointer into the body. Anything else is a header name, matched
case-insensitively. This is why `allowed_patches: [/model, Authorization]` in
`cases.yaml` works without a second list.

## Replay script

`replay.yaml` is the provider-side program for one case. DPC-101's programmable
fixture executes it; the loader parses and validates it so a malformed script
fails at load, not mid-run.

```yaml
schema_version: protocol-conformance-replay-v1

# Asserted about the inbound provider request before anything is replayed.
# Body identity is asserted by the comparators, not here.
expect:
  method: POST
  path: /v1/messages
  headers:
    anthropic-version: "2023-06-01"

steps:
  - {kind: status, status: 200, headers: {content-type: text/event-stream}}
  - {kind: sse, file: provider-response.sse, chunk_bytes: 17}
  - {kind: delay, millis: 25}
  - {kind: disconnect}
```

| `kind` | Fields | Behavior |
| --- | --- | --- |
| `status` | `status`, `headers` | Write the status line and headers. Must be the first step, and may appear only once |
| `body` | `file` | Write a buffered body from a sibling file, then end the response |
| `sse` | `file`, `chunk_bytes` | Write an SSE stream from a sibling file. `chunk_bytes` splits it at fixed byte boundaries; `0` writes it whole |
| `delay` | `millis` | Sleep before the next step |
| `disconnect` | none | Close the connection with no terminal event |

Rules the loader enforces:

- `schema_version` must be `protocol-conformance-replay-v1`.
- `expect.method` and `expect.path` are required.
- The first step is `status`, and no later step is. The commit point is
  therefore explicit: everything after step 0 is a committed response, which is
  what a post-commit error or truncation case needs to assert.
- `file` must name a sibling artifact in the same case directory, and that file
  must exist. A separator or a parent reference is rejected.
- `chunk_bytes` applies to `sse` only.
- An unknown `kind` or an unknown field is an error.

Composing the fault families PL-0042 requires:

- pre-stream error: one `status` with the error code, then `body`.
- post-commit error: `status: 200`, `sse` for the committed prefix, then a
  second `sse` carrying the provider error event.
- mid-stream truncation: `status: 200`, `sse`, then `disconnect`.
- arbitrary chunk boundaries, including a split inside a UTF-8 sequence: `sse`
  with a `chunk_bytes` that does not align to the event grammar.

## Fidelity tiers

Every ledger entry in `cases.yaml` carries one of seven fidelity actions. Each
action maps onto exactly one of three tiers. The tier is what a test asserts
against; the action is the finer detail a reviewer reads.

| Tier | Actions | What it claims | What a test may assert |
| --- | --- | --- | --- |
| `lossless` | `preserved`, `patched`, `mapped` | The semantics crossed the boundary intact | Round-trip holds. A later turn may replay the value in its original carrier, including A-to-B-to-A |
| `visible-but-not-echoable` | `synthesized`, `coerced` | The value is observable in the response but was not client-authored | Response-side equality holds. The value must not be fed back into a later request as if the client had written it |
| `stateful-or-unsupported` | `omitted`, `rejected` | The semantics did not cross the boundary at all | The case must name the exact loss, or reject before dispatch |

Why each row lands where it does:

- `preserved` is byte or block identity. `patched` is a declared, route-owned
  rewrite such as the model alias: the conversation loses nothing. `mapped` is
  rendered into the target protocol's equivalent carrier. All three keep the
  semantics, so all three are `lossless`.
- `synthesized` is a value the router invented, such as a terminal event the
  provider never sent. `coerced` narrowed a value to what the target accepts,
  such as an adaptive thinking policy collapsed onto named provider controls.
  Neither is provider-authoritative, so replaying either as client input would
  be a fabrication. Both are `visible-but-not-echoable`.
- `omitted` dropped the value. `rejected` refused before dispatch. Neither
  reached the peer, so both are `stateful-or-unsupported`.

The loader enforces one rule from this table, which is the whole point of the
tier: if any entry is `stateful-or-unsupported`, the case must declare a
non-`none` `loss` or carry `outcome: rejected`. Silent loss is not expressible
in v1.

`e2e/pkg/conformance` exposes the mapping as `FidelityAction.Tier()`.

## Comparison modes

| Mode | Status | Headers | Buffered body | Stream |
| --- | --- | --- | --- | --- |
| `exact` | Compared | Every header the expectation names | Raw byte identity | Raw byte identity |
| `exact-except` | Compared | Named headers, minus exclusions | Structural JSON identity, minus excluded pointers | Ordered events; per-event type, id, and data |
| `semantic` | Compared | Not compared | Structural JSON equivalence | Ordered events; per-event type and data |
| `reject` | Must equal `reject_status` | Every header in `reject_headers` | Must equal `reject_body` | n/a |

Shared rules:

- A zero expected status does not constrain the observed status. A captured
  request has no status of its own.
- Only headers the expectation names are constrained. An unnamed header is free.
- Numbers are compared as their literal JSON text, so integer precision is never
  lost in the comparison itself.
- Nondeterministic values are matched by type through `volatile`, never deleted.
  A volatile field that disappears is still a failure. Do not broadly delete
  fields before comparison.
- `exact` admits no exclusions. A boundary that needs one is `exact-except`.

Stream rules, shared by every mode that parses events:

- Comparison runs on dispatched events, not raw bytes, so chunk boundaries,
  CRLF, a missing final newline, comments, and keepalives never fail a stream
  that is otherwise correct.
- A trailing unterminated event is still dispatched, so a truncated stream stays
  observable and a truncation case can assert on what did arrive.
- Event count and order are compared first. A reordered stream fails on the
  event type, not on the payload.
- `data: [DONE]` is a literal sentinel, compared as text.
- `semantic` ignores the SSE `id` field: it is reconnection transport metadata,
  not protocol semantics. `exact` and `exact-except` compare it.

The fidelity ledger is the second half of a `semantic` comparison. Typed
equivalence proves the far side received the right content; the ledger proves
every field that did not survive was declared. The router emits the observed
ledger, so a caller passes it to `Case.CompareFidelity` separately.

## Validation rules

`Load` is strict. Every rule below is an error, and every error is reported in
one pass so an author sees the whole picture:

- `cases.yaml` `schema_version` is not `dpc-003-inventory-v1alpha1`.
- An unknown field anywhere in `cases.yaml`, `compare.yaml`, or `replay.yaml`.
- A duplicate case `id`.
- A missing `contract`, `ownership.primary`, or `provenance.origin`.
- A `provenance.sources` key that `normative_sources` does not declare.
- A comparison mode that is not `exact`, `exact-except`, `semantic`, or
  `reject`, or one that `comparison_modes` does not declare.
- A fidelity action outside the seven declared actions.
- A `stateful-or-unsupported` entry with no declared loss and no rejection.
- An inconsistent rejection: `mutation_mode`, `provider_request`,
  `client_response`, and `dispatch_attempts: 0` must agree.
- A case directory that exists but is missing a required artifact.
- A body that declares both `.json` and `.sse`.
- A `replay.yaml` that breaks any rule in [Replay script](#replay-script).

## Provenance

Every case names its normative public sources, and `normative_sources` resolves
them. Deferred cases additionally carry `private_reference`, which is a
behavioral description only. No private code, payload bytes, prompt, credential,
endpoint, provider response, or model alias may enter this tree. Every authored
fixture is written anew from the public sources the case names. Synthetic media
is generated for the corpus with explicit provenance.
