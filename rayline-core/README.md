# Rayline execution core

This Go module contains the shared VSR protocol, thinking-control, cache-evidence,
and reasoning-provenance implementation. Cloud VSR imports these packages through
its local module replacement. The standalone HTTP executor uses the same packages.
The core does not import the cloud router, Envoy, or native inference bindings.

From a checkout of this repository:

```sh
make test-rayline-core
cd rayline-core
CGO_ENABLED=0 go build ./...
```

`protocolcodec` retains the request/response/stream golden corpus and the provider
schema inventory checks. The inventory checks also read the repository's simulator
contracts under `tools/mock-vllm` and `e2e/testing/anthropic-shim`; run them from a
complete repository checkout.

- `llmprotocol` defines the neutral protocol and its validation rules.
- `protocolcodec` translates requests, responses, and incremental stream events.
  `ResponseAccumulator.Response` requires a terminal event; a provider failure is
  represented by the returned response's `Error` field.
- `thinkingcontrol` owns registry admission, private-append rendering, placer
  state, placement identity, and Messages thinking-budget allowance.
- `cachecontext` records completed-call evidence and builds the policy service's
  bounded cache context. A decision consult does not advance the completed-call
  clock. A cache observation is not proof of present KV residency. `MessageChain`
  audits the semantic `messages` prefix, excluding moving cache directives; it
  does not audit Responses input or the full provider/tokenizer prefix.
- `reasoning` owns encrypted and opaque reasoning provenance and issuer identity.
- `diagnostics` emits structured metadata through an optional host callback.
  Cloud VSR installs its existing logging sink; standalone hosts use standard Go
  structured logging by default.

The execution owner stages placer state, issuer history, and call evidence until
successful response completion and downstream delivery. Failed or partial streams
must discard that staged state and must not mark a worker warm. Calling the cache
helpers alone does not implement locking, persistence, retry reconciliation, or
provider residency tracking.

This is a nested module in the VSR repository. All repository consumers carry a
local `replace` so cloud and local builds use this implementation together. A
published module release needs a `rayline-core/vX.Y.Z` tag and consumer version
updates; the current `v0.0.0` repository replacement is not a published release.
The extracted source and fixtures retain the repository's Apache-2.0 license.

## Qualified local token counting

Current-prefix evidence is optional and unknown without an installed counter.
`cachecontext.PromptCounter` consumes the final prepared provider request, including
private append, system instructions and tools. The shared comparison computes a
token-level common prefix only across the same model, template, units and context
epoch. A count-only provider cannot supply token overlap. This evidence never
asserts that a local KV cache is resident.

The command can install a same-host llama.cpp counter through a deployment's
`prompt_counter` member. This implementation is qualified for build
`b11840-36e36e63a`, a text-only model, and the `/v1/chat/completions` provider endpoint.
Run that runtime with `--jinja --no-prefill-assistant`, without mutable `--props`,
and with the configured model alias. This is an external serving runtime; the
execution core does not download a model or launch inference.

```json
{
  "prompt_counter": {
    "model_path": "/absolute/path/model.gguf",
    "model_sha256": "<SHA256 of the complete model file>",
    "build_info": "b11840-36e36e63a",
    "chat_template_sha256": "<SHA256 of the UTF-8 chat_template string>",
    "tool_template_sha256": "<SHA256 of chat_template_tool_use when present>"
  }
}
```

The surrounding deployment must already pin `runtime_identity`, `wire_model`,
and `token_units`. The counter hashes the model file once before readiness, then
checks file identity, size and modification time and `/props` model/build/template
on every measurement. Startup must allow time for the full file hash. Derive the
template pins from the qualified runtime's `/props` response, hashing the decoded
strings directly, without a trailing newline or JSON quoting:

```sh
curl --fail http://127.0.0.1:8080/props > /tmp/llama-props.json
python3 - <<'PY'
import hashlib, json
props = json.load(open('/tmp/llama-props.json'))
print('build_info:', props['build_info'])
for field in ('chat_template', 'chat_template_tool_use'):
    template = props.get(field, '')
    print(field + '_sha256:', hashlib.sha256(template.encode()).hexdigest() if template else '')
PY
shasum -a 256 /absolute/path/model.gguf
```

Omit `tool_template_sha256` only when the runtime exposes no separate tool
template. A changed model, build, template, special-token behavior or failed
counter request produces a visible error. There is no estimated-count fallback.
The counter renders the complete request and its carried history through the
runtime's `/apply-template`, then calls `/tokenize` with `add_special: true` and
`parse_special: true`, matching the pinned inference path. It sends a known
carried-prefix count only when the history's token sequence exactly prefixes the
full prompt; template-dependent history rewrites leave that count unknown.
This boundary is qualified only for a trailing run of Chat `user` messages that
opens a new user turn. A request ending in a `tool` result still stores its exact
full-prompt tokens, but reports no carried-prefix count or overlap. Extending the
costing boundary to tool turns requires separate qualification.
Multimodal requests are refused by this text counter because plain `/tokenize`
does not include the runtime's image/audio processing.

The qualification follows the pinned upstream
[Chat handler and template/tokenization implementation](https://github.com/ggml-org/llama.cpp/blob/36e36e63a54fc8f85b95e9c499a022b597e478dd/tools/server/server-context.cpp)
and [Chat parameter parser](https://github.com/ggml-org/llama.cpp/blob/36e36e63a54fc8f85b95e9c499a022b597e478dd/tools/server/server-common.cpp).
`/props` has no runtime-incarnation or prompt-cache identity. An inference restart
with unchanged pins therefore does not prove residency; local cache TTL and
residency remain unknown. Sidecar restart creates a fresh execution instance and
clears all checkpoints. Transcript rewrites clear the branch epoch and checkpoints.

Claude Code can send a system message after the current user message or tool
result. Qwen's template accepts a system message only at the beginning. A local
Chat deployment may explicitly set `"inband_system": "fold_before_unit_v1"` and
append `:fold_before_unit_v1` to its qualified `runtime_identity`. This reuses
VSR's existing fold rule: the trailing text and cache blocks stay at that turn,
after its input, as user content. Leading instructions are unchanged. It does
change the trailing block's role; it is a qualified compatibility choice, not
lossless preservation of system priority or an admitted thinking capability.
The rule refuses blocks after an answered turn, inside a tool-result run, or
with unsupported fields/content. It is currently allowed only with the native
default control without a steering instruction or budget. Both token counting
and inference consume the identical normalized, rendered body. The action and
control IDs remain unchanged; use a new runtime identity when enabling the rule.
