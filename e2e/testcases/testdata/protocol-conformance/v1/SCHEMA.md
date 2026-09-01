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
| `expected_outcome` | Optional. Marks the case a known failure; see [Expected outcome](#expected-outcome) |
| `provenance` | Origin, normative sources, private behavior reference |
| `ownership` | Primary Workgroup, reviewers, upstream issues |

A case with no directory on disk is contract-only: the loader returns it with
`Case.Loaded() == false`, and the comparators refuse to run on it. That is the
state of all 31 cases until DPC-104 authors fixtures.

Alongside `cases`, the file declares one top-level list:

| Field | Meaning |
| --- | --- |
| `smoke_tier` | Case IDs the pull-request gate runs |

`smoke_tier` is the compact subset CI runs on every pull request. It names the
fewest promoted cases that still reach every ingress protocol the router accepts
and both buffered and streaming client modes. The whole promoted tranche runs
nightly. `Inventory.Smoke()` returns the named cases in declaration order.

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
| `disconnect` | none | Hijack the connection and close it. See the caveat below before using it |

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
- mid-stream truncation: `status: 200`, `sse`, and then simply no more steps.
  The script running out ends the provider body at the HTTP layer with the SSE
  sequence incomplete, which is what a Router must answer with a synthesized
  terminal event.
- arbitrary chunk boundaries, including a split inside a UTF-8 sequence: `sse`
  with a `chunk_bytes` that does not align to the event grammar.

### Do not reach for `disconnect` to model a truncated stream

`disconnect` hijacks the TCP connection and closes it, so no chunked terminator
is written. An in-cluster proxy forwards that as a stream reset and tears the
Router's response path down before anything can be appended, so the client sees
a truncated transfer and no terminal event. The case then measures the proxy
rather than the Router, and it measures a different proxy differently: a managed
edge in front of a provider converts the same backend death into an HTTP-level
end, so the identical case produced a terminal event through Cloud Run and none
through kind while the Router build was the same. One `expected_outcome` cannot
be correct in both.

Ending the body is both the portable shape and the honest one, because a hosted
provider always sits behind a proxy that does the same conversion. Keep
`disconnect` for a case whose subject really is raw-reset handling, and expect
that case's result to be environment-specific.

## Invariants

`expectation.invariants` names the equivalences and properties one case's
contract turns on. The vocabulary is closed: a name outside it is a load error.
Before v1 gained this section an unrecognized name was silently inert.

Each name is held one of three ways.

| Support | Meaning | Who may declare it |
| --- | --- | --- |
| `enforced` | The comparator normalizes both sides for it before diffing | Any case |
| `covered` | The case's authored artifacts already assert it under its declared comparison modes | Any case |
| `deferred` | Recognized, but nothing asserts it yet | Only a case outside the promoted tranche |

The promoted tranche may not declare a `deferred` name. That is the whole point
of the split: a case that gates CI can never carry an invariant that does
nothing. Promoting a deferred case means either implementing its invariant or
proving the authored artifacts already cover it, and moving the name.

### Enforced invariants

Three names describe encodings a public API contract declares interchangeable, so
a structural diff would otherwise report a difference that is not one. They are
opt-in per case, never a global loosening, and they relax the encoding only:
every value is still compared.

| Name | What it accepts | Scope |
| --- | --- | --- |
| `argument-json-equivalence` | A tool-call `arguments` string is compared as parsed JSON, so spacing and key order inside it do not fail | Every OpenAI `function` object carrying a string `arguments`, wherever a request, response, or stream delta puts it |
| `content-encoding-equivalence` | An OpenAI Chat message `content` written as a bare string equals the single text part it expands to | `messages[*].content` and `choices[*].message.content` only, so an Anthropic block list is untouched |
| `null-vs-omitted-equivalence` | A field carrying an explicit `null` equals the same field omitted, in either direction | Any field. Only a `null` is ever reconciled away, so a dropped value is still a failure |

`content-encoding-equivalence` expands rather than collapses: a part carrying
anything beyond `type` and `text`, such as a `cache_control` marker, is not
equivalent to a string and still differs.

An enforced invariant relaxes the structural comparison, so it applies to an
`exact-except` or `semantic` boundary. `exact` asserts byte identity and admits
no relaxation.

`e2e/pkg/conformance/invariant.go` owns the vocabulary and the three rules.

## Expected outcome

A case may declare that it is known to fail against a named router gap. The
promoted tranche carries none today; the shape is kept for the next real gap:

```yaml
expected_outcome:
  status: fail
  reference: vllm-project/semantic-router#0000
  reason: >-
    One sentence naming what the Router does instead.
  signature:
    - 'the exact pointer, path, or event sequence the gap produces'
```

| Field | Meaning |
| --- | --- |
| `status` | Always `fail` in v1. A case expected to pass carries no marker at all |
| `reference` | The gap: an upstream issue, or a stable plan-document token such as `PL-0042#seed-02-anthropic-egress` when no upstream issue covers it |
| `reason` | One sentence on what the router does instead |
| `signature` | Substrings that must **all** appear among the case's failure messages for the failure to count as the known one |

The runner reads the marker over the failures the comparators produced, and there
are three answers:

- **Failed with every signature substring present.** Reported as an expected
  failure. It does not fail the run, and it is counted apart from passes as
  `cases_expected_failures`, so a green run still says what it proved.
- **Passed.** Fails the run. The gap may be fixed, and a stale marker is the one
  thing a reader would wrongly trust.
- **Failed some other way.** Fails the run, with the marker context and every
  actual failure, because a different regression must not hide behind the marker.

A skipped case is never reclassified by its marker. The all-skip guard is
unchanged: a run in which every case skipped asserted nothing and still fails.

The signature is what keeps the marker honest, so make it tight. Name the
pointer, path, or event sequence the gap actually produces, not a phrase any
failure would contain.

### Every marker this corpus has carried was wrong about the Router

Three markers have been retired, and none named a Router defect. seed-02 blamed
the Router for not reaching an Anthropic backend when the profile had never
declared `api_format`. seed-03 blamed it for a missing `anthropic-version` header
that a backend supplies through `extra_headers`, once its provider profile is
complete. seed-06 blamed it for synthesizing no terminal event when the fixture
was resetting a socket the Router never got told about.

So before adding a marker, rule out the corpus itself: the profile's provider
config, the replay script's fault shape, and the environment the run observed.
A marker asserts a Router behaviour, and it is easier to be wrong about that
than the confident tone of a `reason` field suggests. Two of these three
survived a careful reading of the Router source and fell only when run.

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
- An enforced invariant the case declares normalizes both sides before the
  structural diff runs. See [Invariants](#invariants).

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
- An `expectation.invariants` entry outside the closed vocabulary, or a
  `deferred` entry on a case in the promoted tranche.
- An `expected_outcome` whose `status` is not `fail`, or that names no
  `reference`, no `reason`, no `signature`, or an empty signature entry.
- A `stateful-or-unsupported` entry with no declared loss and no rejection.
- An inconsistent rejection: `mutation_mode`, `provider_request`,
  `client_response`, and `dispatch_attempts: 0` must agree.
- A case directory that exists but is missing a required artifact.
- A body that declares both `.json` and `.sse`.
- A `replay.yaml` that breaks any rule in [Replay script](#replay-script).
- A `smoke_tier` entry that is duplicated, names an unknown case, or names a case
  outside the promoted tranche.

## Provenance

Every case names its normative public sources, and `normative_sources` resolves
them. Deferred cases additionally carry `private_reference`, which is a
behavioral description only. No private code, payload bytes, prompt, credential,
endpoint, provider response, or model alias may enter this tree. Every authored
fixture is written anew from the public sources the case names. Synthetic media
is generated for the corpus with explicit provenance.
