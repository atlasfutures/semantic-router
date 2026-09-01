# protocol-conformance profile

## What this profile is for

This profile runs the versioned protocol-conformance corpus
(`e2e/testcases/testdata/protocol-conformance/v1`) against the router and checks
both wire boundaries at once:

- the request the router sent to the provider, and
- the response the router returned to the client.

Every other profile routes at a real model backend, so a test can only assert on
what the router returned. Here the backend is the programmable provider fixture
(`e2e/cmd/conformance-fixture`): a case declares exactly what the provider must
observe and exactly what it replays, which is what makes the outbound half of the
hop assertable.

## What it deploys

| Component | Description |
| --- | --- |
| Envoy Gateway + Envoy AI Gateway | Shared gateway stack (same as `envoy-ai-gateway`). Installed, but the fixture route bypasses the AI Gateway translator -- see the authoring contract below |
| Semantic Router (ExtProc) | Built locally (`e2e-test` image tag) |
| `conformance-fixture` | The provider fixture in the `conformance-fixture-system` namespace |

The fixture image carries the fixture tree as well as the binary. A replay
script's file references resolve on the fixture's own filesystem, so the case
directories have to travel with it; the testcase points `POST /reset` at
`/fixtures/protocol-conformance/v1/<case-id>` inside the container.

## Running the profile

```bash
make e2e-test E2E_PROFILE=protocol-conformance
```

The loop logic also runs without a cluster:

```bash
cd e2e && go test ./testcases/ -run Conformance
```

Those tests drive the same per-case loop against an in-process fixture and a fake
router, so a change to the loop fails fast without a Kind cluster.

## Which cases CI runs

The profile registers two test cases. Both drive the same per-case loop; they
differ only in which corpus selection they run.

| Test case | Selection | Cadence |
| --- | --- | --- |
| `protocol-conformance-smoke` | `smoke_tier` in `cases.yaml` | Pull requests |
| `protocol-conformance-first-six` | The whole `first-six` tranche | Nightly |

`smoke_tier` is the compact all-protocol gate: the fewest promoted cases that
still reach every ingress protocol the router accepts (OpenAI Chat, Anthropic
Messages, OpenAI Responses) and both buffered and streaming client modes. A unit
test in `e2e/pkg/conformance` fails if the tier stops covering that, so the gate
cannot be narrowed into a false pass.

CI never names these test cases in workflow YAML. `e2e/pkg/testmatrix` maps a
profile and cadence onto the subset, and the workflow asks the E2E binary for it:

```bash
./bin/e2e -profile protocol-conformance -list-tests pr
```

## The authoring contract this profile creates

Three things in `values.yaml` are coupled to the corpus. A fixture author has to
keep all of them in view.

**Model aliases and `api_format`.** `providers.models` declares three aliases,
one per provider wire protocol and dialect the seed tranche uses. `api_format` is
what selects the encoding; `backend_refs[].type` is a backend label and selects
nothing, so a model that omits `api_format` is dispatched as OpenAI Chat no
matter what its backend claims to be.

| Alias | `api_format` | Provider wire shape |
| --- | --- | --- |
| `conformance-chat-model` | `openai` | OpenAI Chat Completions |
| `conformance-messages-model` | `anthropic` | Anthropic Messages |
| `conformance-openrouter-model` | `openai` | OpenRouter Chat dialect |

A case's `expected-provider-request.json` names one of these in `/model`. The
route to the fixture is a plain `HTTPRoute`, not an `AIGatewayRoute`: the Router
owns protocol translation and Envoy only carries the bytes, which is what makes
the fixture's recording evidence about the Router rather than about two
translators in series. It is also the only wiring that can carry native Anthropic
egress, because the AI Gateway external processor registers a fixed path set and
answers `/v1/messages` with a 404.

**The Anthropic provider profile.** `conformance-messages-model` declares
`base_url`, `provider`, `auth_header`, a synthetic `api_key` and
`extra_headers: {anthropic-version}`. Anthropic requires that version header and
the Router synthesizes none, so the backend has to supply it -- and a provider
profile is all or nothing: naming `extra_headers` obliges the rest, and a partial
set fails credential resolution and kills dispatch before the provider is
reached. The key is synthetic because the fixture never checks it.

**Routing keywords.** Every seed case sends `"model": "auto"`, so the decision
layer picks the provider. `routing.signals.keywords` selects on a discriminating
word from each case's prompt. A new case whose prompt does not hit the intended
keyword routes to the Chat catch-all and fails at the provider boundary with a
path or model mismatch, which is the symptom to look for first.

## Skipped cases

A case whose directory is not authored yet is reported as skipped with its
reason, in the testcase details under `skipped`. It is never failed and never
dropped from the report. The deferred `import-*` tranche is out of scope for this
profile.

## Fidelity ledger

`expectation.fidelity` is compared only when the router emits a ledger in the
`x-vsr-fidelity-ledger` response header. The router does not emit one yet; the
report counts how many cases were verified that way under
`fidelity_ledgers_checked`, so an unverified ledger is visible rather than
silently treated as satisfied.
