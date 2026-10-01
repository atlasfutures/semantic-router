//go:build !windows && cgo

package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// responsesWorkersModels adds two Responses workers on OpenAI to the policy
// dispatch config: two targets, each the only reader of its own blobs.
const responsesWorkersModels = `    - name: gpt
      provider_model_id: gpt-5.5
      api_format: responses
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openai-gpt
          base_url: https://api.openai.com/v1
          provider: openai
          api_key_env: POLICY_E2E_PROVIDER_KEY
    - name: gpt-b
      provider_model_id: gpt-5.4
      api_format: responses
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openai-gpt-b
          base_url: https://api.openai.com/v1
          provider: openai
          api_key_env: POLICY_E2E_PROVIDER_KEY

    - name: gpt-or
      provider_model_id: vendor/responses
      api_format: responses
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.1
        cache_write_per_1m: 1.25
        completion_per_1m: 5
      backend_refs:
        - name: openrouter-gpt
          base_url: https://openrouter.ai/api/v1
          provider: openrouter
          api_key_env: POLICY_E2E_PROVIDER_KEY

routing:
  modelCards:
    - name: gpt
      modality: text
    - name: gpt-b
      modality: text
    - name: gpt-or
      modality: text
`

const responsesWorkersModelRefs = `      modelRefs:
        - model: gpt
          use_reasoning: true
        - model: gpt-b
          use_reasoning: true
        - model: gpt-or
          use_reasoning: true
`

// writeResponsesPolicyConfig is the policy dispatch config with the Responses
// workers added beside the Chat and Messages ones.
func writeResponsesPolicyConfig(t *testing.T, policyURL string, actions map[string]config.RaylineARCPolicyBinding, providerDefault bool) string {
	t.Helper()
	path := writePolicyDispatchConfig(t, policyURL, actions)
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	rendered := string(raw)
	for anchor, replacement := range map[string]string{
		"\nrouting:\n  modelCards:\n": "\n" + responsesWorkersModels,
		"      modelRefs:\n":          responsesWorkersModelRefs,
	} {
		if strings.Count(rendered, anchor) != 1 {
			t.Fatalf("the config template changed; update the %q anchor", anchor)
		}
		rendered = strings.Replace(rendered, anchor, replacement, 1)
	}
	if providerDefault {
		anchor := "            package_sha256: " + policyTestPackage + "\n"
		rendered = strings.Replace(rendered, anchor, anchor+"            dispatch_effort: provider_default\n", 1)
	}
	if err := os.WriteFile(path, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// codexTurn runs one captured Codex request through the request phases and
// returns the provider-bound body and the turn's context.
func codexTurn(t *testing.T, router *OpenAIRouter, episode, fixture string) (string, *RequestContext) {
	t.Helper()
	codex, err := os.ReadFile("../protocolcodec/testdata/codex/" + fixture)
	if err != nil {
		t.Fatal(err)
	}
	codex = bytes.Replace(codex, []byte(`"model": "gpt-5-codex"`), []byte(`"model": "auto"`), 1)
	ctx := &RequestContext{Headers: map[string]string{}, RequestID: "responses-e2e-" + episode, StartTime: time.Now(), TraceContext: context.Background()}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/responses"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
		}},
	}}
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: codex, EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		t.Fatalf("request refused: %d %s", immediate.GetStatus().GetCode(), immediate.GetBody())
	}
	return string(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()), ctx
}

func finishCodexTurn(t *testing.T, router *OpenAIRouter, ctx *RequestContext, status string) {
	t.Helper()
	if _, err := router.handleResponseHeaders(arcResponseHeaders(status), ctx); err != nil {
		t.Fatalf("response headers %s: %v", status, err)
	}
	completeTestResponse(t, ctx)
	finalizeSelectionProcessTerminal(ctx)
}

const codexBlob = "gAAAAB-fake-encrypted-content-for-capture"

func responsesPolicyRouter(t *testing.T, providerDefault bool) (*OpenAIRouter, *fakePolicyService, map[string]config.RaylineARCPolicyBinding) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := map[string]config.RaylineARCPolicyBinding{
		"think":  policyAction("think", "none", "think-trained", policyTestEffort("high"), nil, ""),
		"gpt":    policyAction("gpt", "none", "gpt-trained", policyTestEffort("high"), nil, ""),
		"gpt-b":  policyAction("gpt-b", "none", "gpt-b-trained", policyTestEffort("low"), nil, ""),
		"gpt-or": policyAction("gpt-or", "none", "gpt-or-trained", policyTestEffort("low"), nil, ""),
		"off":    policyAction("off", "none", "off-trained", policyTestEffort("none"), nil, ""),
		// Every modelRef serves an action.
		"claude":     policyAction("claude", "none", "claude-opus-5", policyTestEffort("medium"), nil, ""),
		"claude-off": policyAction("claude-off", "none", "claude-opus-5", policyTestEffort("none"), nil, ""),
	}
	catalog := make([]string, 0, len(actions))
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	router, err := NewOpenAIRouter(writeResponsesPolicyConfig(t, fake.URL(), actions, providerDefault))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	return router, fake, actions
}

// A policy action chosen for a Responses worker reaches the provider as
// reasoning.effort, or as no effort under dispatch_effort: provider_default.
func TestPolicyDispatchesACodexTurnToAResponsesWorker(t *testing.T) {
	for _, providerDefault := range []bool{false, true} {
		name := "declared"
		if providerDefault {
			name = "provider_default"
		}
		t.Run(name, func(t *testing.T) {
			router, fake, actions := responsesPolicyRouter(t, providerDefault)
			fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["gpt"].ActionID })
			raw, _ := codexTurn(t, router, "responses-effort-"+name, "turn1-request.json")
			var body map[string]json.RawMessage
			if err := json.Unmarshal([]byte(raw), &body); err != nil {
				t.Fatal(err)
			}
			assertJSONField(t, body, "model", `"gpt-5.5"`)
			var reasoning struct {
				Effort string `json:"effort"`
			}
			_ = json.Unmarshal(body["reasoning"], &reasoning)
			want := "high"
			if providerDefault {
				want = ""
			}
			if reasoning.Effort != want {
				t.Fatalf("reasoning = %s, want effort %q", body["reasoning"], want)
			}
			if _, present := body["input"]; !present {
				t.Fatalf("the Responses body lost its input: %s", raw)
			}
		})
	}
}

// A Codex episode forwards its client's encrypted reasoning only to the target
// that issued it: back to the same worker the blob travels unchanged; once a
// turn has gone to another Responses worker, the client holds blobs of two
// issuers and every later turn drops them.
func TestCodexEncryptedReasoningReachesOnlyItsIssuer(t *testing.T) {
	router, fake, actions := responsesPolicyRouter(t, false)
	route := func(worker string) {
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[worker].ActionID })
	}
	const episode = "codex-issuer"

	route("gpt")
	body, ctx := codexTurn(t, router, episode, "turn1-request.json")
	if strings.Contains(body, codexBlob) {
		t.Fatalf("the first turn holds no blob: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")

	route("gpt")
	body, ctx = codexTurn(t, router, episode, "turn2-request.json")
	if !strings.Contains(body, codexBlob) || !strings.Contains(body, `"type":"reasoning"`) {
		t.Fatalf("the blob did not reach the worker that issued it: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")

	route("gpt-b")
	body, ctx = codexTurn(t, router, episode, "turn2-request.json")
	if strings.Contains(body, codexBlob) || strings.Contains(body, `"type":"reasoning"`) {
		t.Fatalf("the blob, or its item, reached a worker that did not issue it: %s", body)
	}
	if !strings.Contains(body, "call_1") {
		t.Fatalf("the tool turn was lost with the reasoning item: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")

	route("gpt")
	body, ctx = codexTurn(t, router, episode, "turn2-request.json")
	if strings.Contains(body, codexBlob) {
		t.Fatalf("the client now holds blobs of two issuers, yet one was forwarded: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")
}

// The issuer set moves only with a committed turn: a turn to another worker
// that fails leaves the episode's blobs attributed to their issuer.
func TestFailedTurnLeavesTheReasoningIssuer(t *testing.T) {
	router, fake, actions := responsesPolicyRouter(t, false)
	route := func(worker string) {
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[worker].ActionID })
	}
	const episode = "codex-issuer-failed"

	route("gpt")
	_, ctx := codexTurn(t, router, episode, "turn1-request.json")
	finishCodexTurn(t, router, ctx, "200")

	route("gpt-b")
	body, ctx := codexTurn(t, router, episode, "turn2-request.json")
	if strings.Contains(body, codexBlob) {
		t.Fatalf("the blob reached a worker that did not issue it: %s", body)
	}
	finishCodexTurn(t, router, ctx, "500")

	route("gpt")
	body, ctx = codexTurn(t, router, episode, "turn2-request.json")
	if !strings.Contains(body, codexBlob) {
		t.Fatalf("a failed turn moved the issuer record: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")
}

// A turn on a Chat worker drops the blob and issues none, so the blobs the
// client still holds keep their issuer.
func TestChatTurnKeepsTheReasoningIssuer(t *testing.T) {
	router, fake, actions := responsesPolicyRouter(t, false)
	route := func(worker string) {
		fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions[worker].ActionID })
	}
	const episode = "codex-issuer-chat"

	route("gpt")
	_, ctx := codexTurn(t, router, episode, "turn1-request.json")
	finishCodexTurn(t, router, ctx, "200")

	route("think")
	body, ctx := codexTurn(t, router, episode, "turn2-request.json")
	if strings.Contains(body, codexBlob) {
		t.Fatalf("the blob reached a Chat worker: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")

	route("gpt")
	body, ctx = codexTurn(t, router, episode, "turn2-request.json")
	if !strings.Contains(body, codexBlob) {
		t.Fatalf("a Chat turn moved the issuer record: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")
}

// OpenRouter picks the serving provider per request, and providers cannot
// read each other's blobs, so a blob an OpenRouter worker issued is never
// forwarded, not even back to that worker.
func TestOpenRouterResponsesWorkerIsNeverSentEncryptedReasoning(t *testing.T) {
	router, fake, actions := responsesPolicyRouter(t, false)
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["gpt-or"].ActionID })
	const episode = "codex-issuer-openrouter"

	_, ctx := codexTurn(t, router, episode, "turn1-request.json")
	finishCodexTurn(t, router, ctx, "200")
	body, ctx := codexTurn(t, router, episode, "turn2-request.json")
	if strings.Contains(body, codexBlob) {
		t.Fatalf("a blob reached an OpenRouter worker: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")

	// The client's blobs came from OpenRouter, so a direct worker does not
	// get them either.
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return actions["gpt"].ActionID })
	body, ctx = codexTurn(t, router, episode, "turn2-request.json")
	if strings.Contains(body, codexBlob) {
		t.Fatalf("a blob OpenRouter issued reached another worker: %s", body)
	}
	finishCodexTurn(t, router, ctx, "200")
}

// A blob is readable only by the account that issued it, so the credential
// a turn is sent with is part of its issuer: a rotated or per-user key is
// another issuer, and a turn whose credential cannot be resolved forwards
// nothing.
func TestReasoningIssuerFollowsTheCredential(t *testing.T) {
	dispatch := &providerDispatch{
		logicalModel: "gpt", backendName: "openai-gpt", upstreamModel: "gpt-5.5",
		profile: &config.ProviderProfile{Type: "openai", BaseURL: "https://api.openai.com/v1"},
	}
	first := reasoningIssuerFor(dispatch, dispatchCredential{key: "key-one", known: true})
	if first == raylinearc.ReasoningIssuerUnknown || first != reasoningIssuerFor(dispatch, dispatchCredential{key: "key-one", known: true}) {
		t.Fatalf("one target and key named issuer %q inconsistently", first)
	}
	if rotated := reasoningIssuerFor(dispatch, dispatchCredential{key: "key-two", known: true}); rotated == first {
		t.Fatal("a rotated key is the same issuer")
	}
	if unresolved := reasoningIssuerFor(dispatch, dispatchCredential{}); unresolved != raylinearc.ReasoningIssuerUnknown {
		t.Fatalf("an unresolved credential named issuer %q", unresolved)
	}
	if strings.Contains(first, "key-one") {
		t.Fatal("the issuer carries the key")
	}
}
