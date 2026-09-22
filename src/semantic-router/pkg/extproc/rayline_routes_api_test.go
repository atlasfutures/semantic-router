/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func routesRouter(enabled bool) *OpenAIRouter {
	return &OpenAIRouter{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name: "rayline",
			Algorithm: &config.AlgorithmConfig{
				Type:    config.RaylineARCAlgorithmType,
				OnError: "fail_closed",
				RaylineARC: &config.RaylineARCAlgorithmConfig{
					RoutesAPI: config.RaylineARCRoutesAPIConfig{
						Enabled:         enabled,
						CheckpointLabel: "arc-2026-09-12",
					},
				},
			},
		}}},
	}}
}

func routesContext(headers map[string]string) *RequestContext {
	captured := map[string]string{":path": raylineRoutesAPIPath, ":method": "POST"}
	for name, value := range headers {
		captured[name] = value
	}
	return &RequestContext{Headers: captured}
}

// The endpoint stays invisible where it is not configured. A cell that
// answered 405 to a GET would be telling an unauthenticated prober that the
// path exists here and is merely switched off.
func TestRaylineRoutesIsNotFoundWhileDisabled(t *testing.T) {
	t.Parallel()
	router := routesRouter(false)
	for _, method := range []string{"POST", "GET", "DELETE"} {
		response := router.validateRequestHeaders(method, raylineRoutesAPIPath, &RequestContext{Headers: map[string]string{}})
		if response == nil {
			t.Fatalf("validateRequestHeaders(%q, &RequestContext{Headers: map[string]string{}}) = nil, want 404", method)
		}
		if code := immediateStatusCode(t, response); code != 404 {
			t.Fatalf("validateRequestHeaders(%q, &RequestContext{Headers: map[string]string{}}) status = %d, want 404", method, code)
		}
	}
}

func TestRaylineRoutesAdmitsOnlyPost(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	if response := router.validateRequestHeaders("POST", raylineRoutesAPIPath, &RequestContext{Headers: map[string]string{}}); response != nil {
		t.Fatalf("validateRequestHeaders(POST, &RequestContext{Headers: map[string]string{}}) = %v, want nil", response)
	}
	response := router.validateRequestHeaders("GET", raylineRoutesAPIPath, &RequestContext{Headers: map[string]string{}})
	if response == nil {
		t.Fatal("validateRequestHeaders(GET, &RequestContext{Headers: map[string]string{}}) = nil, want 405")
	}
	if code := immediateStatusCode(t, response); code != 405 {
		t.Fatalf("validateRequestHeaders(GET, &RequestContext{Headers: map[string]string{}}) status = %d, want 405", code)
	}
}

// The body's own shape decides the wire format, so the caller sends the same
// bytes here that they were going to send to the executing endpoint.
func TestRaylineRoutesReadsWireFormatFromTheBody(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		body string
		want llmprotocol.WireFormat
	}{
		"anthropic messages": {
			// system is Anthropic's own top-level field. max_tokens alone no
			// longer implies Anthropic: Chat accepts it as a legacy field, so
			// it cannot discriminate.
			body: `{"model":"rayline-router","max_tokens":16,"system":"be terse","messages":[{"role":"user","content":"hi"}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		"openai responses": {
			body: `{"model":"rayline-router","input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]}]}`,
			want: llmprotocol.OpenAIResponsesV1,
		},
		"openai chat completions": {
			body: `{"model":"rayline-router","max_completion_tokens":16,"messages":[{"role":"user","content":"hi"}]}`,
			want: llmprotocol.OpenAIChatV1,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			got, _, detail := readRaylineRoutesBody([]byte(testCase.body), false, "")
			if detail != "" {
				t.Fatalf("readRaylineRoutesBody() detail = %q, want none", detail)
			}
			if got != testCase.want {
				t.Fatalf("readRaylineRoutesBody() = %q, want %q", got, testCase.want)
			}
		})
	}
}

func TestRaylineRoutesRefusesBodiesItCannotRead(t *testing.T) {
	t.Parallel()
	for name, body := range map[string]string{
		"not json":        `not json`,
		"not an object":   `[]`,
		"no turns":        `{"model":"rayline-router"}`,
		"empty messages":  `{"messages":[]}`,
		"messages scalar": `{"messages":"hi"}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if _, _, detail := readRaylineRoutesBody([]byte(body), false, ""); detail == "" {
				t.Fatal("readRaylineRoutesBody() detail = \"\", want a refusal")
			}
		})
	}
}

// A developer who sends tools and gets a plausible route back has no way to
// learn the tools were never encoded. The warning is the only place that
// shows.
// The warning has to describe what this cell did, because the two cells do
// different things with the same body. Telling a caller their tools were
// ignored on a cell that encoded the names would misdescribe the decision.
func TestRaylineRoutesWarnsWhatItDidWithTools(t *testing.T) {
	t.Parallel()
	body := []byte(`{"messages":[{"role":"user","content":"hi"}],"system":"be terse","tools":[{"name":"edit"}]}`)

	_, dropped, detail := readRaylineRoutesBody(body, false, "")
	if detail != "" {
		t.Fatalf("readRaylineRoutesBody() detail = %q, want none", detail)
	}
	if len(dropped) != 1 || !strings.HasPrefix(dropped[0], "tools_not_encoded:") {
		t.Fatalf("warnings = %v, want one tools_not_encoded warning", dropped)
	}

	_, encoded, detail := readRaylineRoutesBody(body, true, "")
	if detail != "" {
		t.Fatalf("readRaylineRoutesBody() detail = %q, want none", detail)
	}
	if len(encoded) != 1 || !strings.HasPrefix(encoded[0], "tool_schemas_not_encoded:") {
		t.Fatalf("warnings = %v, want the names-encoded warning", encoded)
	}
}

// A body that names its dialect is read as that dialect; one that does not
// says so, rather than being routed on a coin flip.
func TestRaylineRoutesDialectMarkers(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		body    string
		want    llmprotocol.WireFormat
		certain bool
	}{
		"anthropic by system": {
			body:    `{"messages":[{"role":"user","content":"hi"}],"system":"be terse"}`,
			want:    llmprotocol.AnthropicMessagesV1,
			certain: true,
		},
		// max_tokens is accepted by BOTH dialects -- required by Anthropic,
		// legacy in Chat -- so it cannot decide, and a body carrying only it
		// is inferred rather than known.
		"shared max_tokens decides nothing": {
			body:    `{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}`,
			want:    llmprotocol.OpenAIChatV1,
			certain: false,
		},
		"chat by system role in messages": {
			body:    `{"messages":[{"role":"system","content":"be terse"},{"role":"user","content":"hi"}]}`,
			want:    llmprotocol.OpenAIChatV1,
			certain: true,
		},
		// The case the old single-field rule got wrong: Chat with no
		// max_completion_tokens read as Anthropic and said nothing.
		"chat by penalty": {
			body:    `{"messages":[{"role":"user","content":"hi"}],"frequency_penalty":0.2}`,
			want:    llmprotocol.OpenAIChatV1,
			certain: true,
		},
		"chat by max_completion_tokens": {
			body:    `{"messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16}`,
			want:    llmprotocol.OpenAIChatV1,
			certain: true,
		},
		// A marker-free body cannot be valid Anthropic, which requires
		// max_tokens, so reading it as Anthropic made every ordinary Chat
		// request fail the codec and answer 400.
		"marker-free reads as chat": {
			body:    `{"model":"m","messages":[{"role":"user","content":"hi"}]}`,
			want:    llmprotocol.OpenAIChatV1,
			certain: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			format, warnings, detail := readRaylineRoutesBody([]byte(testCase.body), false, "")
			if detail != "" {
				t.Fatalf("detail = %q, want none", detail)
			}
			if format != testCase.want {
				t.Fatalf("format = %q, want %q", format, testCase.want)
			}
			inferred := false
			for _, warning := range warnings {
				if strings.HasPrefix(warning, "format_inferred:") {
					inferred = true
				}
			}
			if inferred == testCase.certain {
				t.Fatalf("format_inferred present = %v, want %v", inferred, !testCase.certain)
			}
		})
	}
}

// A route id is a join key. 32 bits collides inside one busy day.
func TestRaylineRouteIDCarriesEnoughEntropy(t *testing.T) {
	t.Parallel()
	minted := raylineRoutesRouteID(routesContext(nil))
	hex := strings.TrimPrefix(minted, raylineRouteIDPrefix)
	if len(hex) < 16 {
		t.Fatalf("route id %q has %d hex characters, want at least 16", minted, len(hex))
	}
}

// Two callers must not be able to compose the same episode identity.
func TestRaylineRoutesEpisodeIdentityIsUnambiguous(t *testing.T) {
	t.Parallel()
	withColonInSession := raylineRoutesEpisodeIdentity(routesContext(map[string]string{
		raylineRoutesSessionHeader: "a:b",
	}))
	withBranch := raylineRoutesEpisodeIdentity(routesContext(map[string]string{
		raylineRoutesSessionHeader: "a",
		raylineRoutesBranchHeader:  "b",
	}))
	if withColonInSession == withBranch {
		t.Fatalf("session %q and session+branch composed to the same identity %q",
			"a:b", withBranch)
	}
}

func TestRaylineRoutesWarnsThatACheckpointPinWasIgnored(t *testing.T) {
	t.Parallel()
	tracked := &config.RaylineARCRoutesAPIConfig{Enabled: true, EpisodeWrites: true}
	ctx := routesContext(map[string]string{raylineRoutesCheckpointHeader: "arc-2026-09-12"})
	warnings := raylineRoutesHeaderWarnings(ctx, tracked)
	if len(warnings) != 1 || !strings.HasPrefix(warnings[0], "checkpoint_not_pinned:") {
		t.Fatalf("warnings = %v, want one checkpoint_not_pinned warning", warnings)
	}
	if extra := raylineRoutesHeaderWarnings(routesContext(nil), tracked); len(extra) != 0 {
		t.Fatalf("warnings without a pin = %v, want none", extra)
	}
}

// A caller who sends a conversation id to a cell that keeps no episodes gets
// a route that looks exactly like a route which used the conversation. Every
// turn silently reads as a first turn, and nothing else in the answer says so.
func TestRaylineRoutesWarnsWhenTheConversationIsNotTracked(t *testing.T) {
	t.Parallel()
	ctx := routesContext(map[string]string{raylineRoutesSessionHeader: "conv-1"})
	untracked := &config.RaylineARCRoutesAPIConfig{Enabled: true}
	warnings := raylineRoutesHeaderWarnings(ctx, untracked)
	if len(warnings) != 1 || !strings.HasPrefix(warnings[0], "episode_not_tracked:") {
		t.Fatalf("warnings = %v, want one episode_not_tracked warning", warnings)
	}
	tracked := &config.RaylineARCRoutesAPIConfig{Enabled: true, EpisodeWrites: true}
	if quiet := raylineRoutesHeaderWarnings(ctx, tracked); len(quiet) != 0 {
		t.Fatalf("warnings with episodes on = %v, want none", quiet)
	}
}

// Two concurrent subagents on one conversation are two trajectories. Folding
// the branch into the identity is what keeps each from reading as the other's
// previous turn.
func TestRaylineRoutesEpisodeIdentitySeparatesBranches(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		headers map[string]string
		want    string
	}{
		"stateless": {headers: nil, want: ""},
		"session only": {
			headers: map[string]string{raylineRoutesSessionHeader: "conv-1"},
			want:    "conv-1",
		},
		"session and branch": {
			headers: map[string]string{
				raylineRoutesSessionHeader: "conv-1",
				raylineRoutesBranchHeader:  "sub-2",
			},
			want: "conv-1:sub-2",
		},
		"branch without a session stays stateless": {
			headers: map[string]string{raylineRoutesBranchHeader: "sub-2"},
			want:    "",
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := raylineRoutesEpisodeIdentity(routesContext(testCase.headers)); got != testCase.want {
				t.Fatalf("raylineRoutesEpisodeIdentity() = %q, want %q", got, testCase.want)
			}
		})
	}
}

func TestRaylineRoutesAdoptsACallerMintedRouteID(t *testing.T) {
	t.Parallel()
	ctx := routesContext(map[string]string{raylineRoutesRouteIDHeader: "rt_abc123"})
	if got := raylineRoutesRouteID(ctx); got != "rt_abc123" {
		t.Fatalf("raylineRoutesRouteID() = %q, want the caller's id", got)
	}
	minted := raylineRoutesRouteID(routesContext(nil))
	if !strings.HasPrefix(minted, raylineRouteIDPrefix) {
		t.Fatalf("raylineRoutesRouteID() = %q, want a %s id", minted, raylineRouteIDPrefix)
	}
	if second := raylineRoutesRouteID(routesContext(nil)); second == minted {
		t.Fatal("raylineRoutesRouteID() minted the same id twice")
	}
}

// The budget is null, not zero, when thinking is off: a caller reading it as
// a number must not be able to mistake "no budget applies" for "a budget of
// zero applies".
func TestRaylineRoutesPayloadOmitsABudgetWhenThinkingIsOff(t *testing.T) {
	t.Parallel()
	payload := raylineRoutesPayload(
		"rte_1234abcd",
		"arc-2026-09-12.c82a1f3e",
		routerruntime.RouteDecision{
			WorkerModel: "deepseek/deepseek-v4-pro",
			Thinking:    routerruntime.RouteThinking{Mode: "off"},
		},
		nil,
		212*time.Millisecond,
	)
	body := marshalRoutesPayload(t, payload)
	if body["thinking"].(map[string]interface{})["budget_tokens"] != nil {
		t.Fatalf("thinking.budget_tokens = %v, want null", body["thinking"])
	}
	if _, present := body["episode"]; present {
		t.Fatal("episode is present on a stateless route, want it omitted")
	}
	if _, present := body["baseline"]; present {
		t.Fatal("baseline is present without a reference worker, want it omitted")
	}
	if body["object"] != "route" {
		t.Fatalf("object = %v, want \"route\"", body["object"])
	}
	if body["latency_ms"] != float64(212) {
		t.Fatalf("latency_ms = %v, want 212", body["latency_ms"])
	}
	// Always present, usually empty. A caller that has to distinguish absent
	// from empty is a caller who will skip the check.
	warnings, ok := body["warnings"].([]interface{})
	if !ok || len(warnings) != 0 {
		t.Fatalf("warnings = %v, want an empty array", body["warnings"])
	}
	alternatives, ok := body["alternatives"].([]interface{})
	if !ok || len(alternatives) != 0 {
		t.Fatalf("alternatives = %v, want an empty array", body["alternatives"])
	}
}

func TestRaylineRoutesPayloadCarriesTheWholeDecision(t *testing.T) {
	t.Parallel()
	payload := raylineRoutesPayload(
		"rte_1234abcd",
		"arc-2026-09-12.c82a1f3e",
		routerruntime.RouteDecision{
			WorkerModel: "deepseek/deepseek-v4-pro",
			Thinking:    routerruntime.RouteThinking{Mode: "on", BudgetTokens: 4096},
			Checkpoint:  "c82a1f3e",
			Alternatives: []routerruntime.RouteAlternative{
				{Model: "z-ai/glm-5.3-flash", Score: 0.62},
			},
			SelectedPricing: routerruntime.RoutePricing{InputPerMTok: 0.435, OutputPerMTok: 0.87},
			Baseline: routerruntime.RouteBaseline{
				Model:   "anthropic/claude-sonnet-5",
				Pricing: routerruntime.RoutePricing{InputPerMTok: 3, OutputPerMTok: 15},
			},
			Episode:  &routerruntime.RouteEpisode{TurnIndex: 4, Stayed: true},
			Usage:    routerruntime.RouteUsage{EncodedInputTokens: 1840, CacheReadTokens: 1200},
			Warnings: []string{"tools_not_encoded: ..."},
		},
		[]string{"checkpoint_not_pinned: ..."},
		212*time.Millisecond,
	)
	body := marshalRoutesPayload(t, payload)
	if body["thinking"].(map[string]interface{})["budget_tokens"] != float64(4096) {
		t.Fatalf("thinking.budget_tokens = %v, want 4096", body["thinking"])
	}
	if body["baseline"].(map[string]interface{})["model"] != "anthropic/claude-sonnet-5" {
		t.Fatalf("baseline = %v, want the reference worker", body["baseline"])
	}
	if body["episode"].(map[string]interface{})["turn_index"] != float64(4) {
		t.Fatalf("episode = %v, want turn_index 4", body["episode"])
	}
	if body["usage"].(map[string]interface{})["encoded_input_tokens"] != float64(1840) {
		t.Fatalf("usage = %v, want 1840 encoded input tokens", body["usage"])
	}
	// Edge warnings and runtime warnings land in one array, because a caller
	// reading it does not care which layer noticed.
	if warnings := body["warnings"].([]interface{}); len(warnings) != 2 {
		t.Fatalf("warnings = %v, want both the header and the body warning", warnings)
	}
}

// The endpoint answers this router's callers, who are already parsing the
// Anthropic envelope from the endpoint they would otherwise have called.
func TestRaylineRoutesErrorsUseTheAnthropicEnvelope(t *testing.T) {
	t.Parallel()
	router := routesRouter(false)
	response := router.handleRaylineRoutesAPI(
		[]byte(`{"messages":[{"role":"user","content":"hi"}]}`),
		routesContext(nil),
	)
	if response == nil {
		t.Fatal("handleRaylineRoutesAPI() = nil on a disabled router, want 404")
	}
	if code := immediateStatusCode(t, response); code != 404 {
		t.Fatalf("status = %d, want 404", code)
	}
	var envelope struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
		t.Fatalf("error body is not JSON: %v", err)
	}
	if envelope.Type != "error" || envelope.Error.Type != "not_found_error" {
		t.Fatalf("error envelope = %+v, want an Anthropic not_found_error", envelope)
	}
}

// Every other path must fall through to routing untouched, or this endpoint
// has quietly become a filter on live traffic.
func TestRaylineRoutesIgnoresEveryOtherPath(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	ctx := &RequestContext{Headers: map[string]string{":path": "/v1/messages", ":method": "POST"}}
	if response := router.handleRaylineRoutesAPI([]byte(`{}`), ctx); response != nil {
		t.Fatalf("handleRaylineRoutesAPI(/v1/messages) = %v, want nil", response)
	}
}

func marshalRoutesPayload(t *testing.T, payload raylineRoutesResponse) map[string]interface{} {
	t.Helper()
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var body map[string]interface{}
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	return body
}

func immediateStatusCode(t *testing.T, response *ext_proc.ProcessingResponse) int {
	t.Helper()
	immediate := response.GetImmediateResponse()
	if immediate == nil || immediate.Status == nil {
		t.Fatalf("response %v carries no immediate status", response)
	}
	return int(immediate.Status.Code)
}

// The body phase must answer this path itself. Reaching the ingress codec
// would mean the request was being prepared for a turn that is never going to
// happen, and the caller would get this router's own error envelope instead
// of the one this endpoint promises.
func TestRaylineRoutesBodyPhaseAnswersBeforeTheIngressCodec(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"model":"rayline-router"}`)},
	}, routesContext(nil))
	if err != nil {
		t.Fatalf("HandleRequestBody() error = %v", err)
	}
	if code := immediateStatusCode(t, response); code != 400 {
		t.Fatalf("status = %d, want 400", code)
	}
	var envelope map[string]interface{}
	if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
		t.Fatalf("error body is not JSON: %v", err)
	}
	if envelope["type"] != "error" {
		t.Fatalf("error body = %v, want the Anthropic envelope this endpoint promises", envelope)
	}
}

// The mirror of the case above, and the one that matters more: a routed
// request on a routes-enabled cell must be untouched by any of this.
func TestRaylineRoutesLeavesRoutedTrafficAlone(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	ctx := &RequestContext{Headers: map[string]string{":path": "/v1/messages", ":method": "POST"}}
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"model":"rayline-router"}`)},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody() error = %v", err)
	}
	body := response.GetImmediateResponse().Body
	if strings.Contains(string(body), `"type":"error"`) {
		t.Fatalf("routed request answered with the routes envelope: %s", body)
	}
	if !strings.Contains(string(body), "invalid inference request") {
		t.Fatalf("routed request answered with %s, want the ingress refusal", body)
	}
}

// A constant is worse than an absent field: it reads as measured. Neither
// confidence nor reason had a real source, so neither is published.
func TestRaylineRoutesPublishesNoUnsourcedFields(t *testing.T) {
	t.Parallel()
	payload := raylineRoutesPayload(
		"rte_1234abcd",
		"arc-2026-09-12.c82a1f3e",
		routerruntime.RouteDecision{WorkerModel: "deepseek/deepseek-v4-pro"},
		nil,
		time.Millisecond,
	)
	body := marshalRoutesPayload(t, payload)
	for _, field := range []string{"confidence", "reason"} {
		if _, present := body[field]; present {
			t.Fatalf("%s is published, but this router has no source for it", field)
		}
	}
}

// The hash pins the artifact and the label says which release it is. Neither
// alone does both, so the pair is the identity and either half alone is a
// degraded form of it rather than a different format.
func TestRaylineRoutesCheckpointPairsLabelAndHash(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		label string
		hash  string
		want  string
	}{
		"both":     {label: "arc-2026-09-12", hash: "c82a1f3e", want: "arc-2026-09-12.c82a1f3e"},
		"no label": {label: "", hash: "c82a1f3e", want: "c82a1f3e"},
		"no hash":  {label: "arc-2026-09-12", hash: "", want: "arc-2026-09-12"},
		"neither":  {label: "", hash: "", want: ""},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if got := raylineRoutesCheckpoint(testCase.label, testCase.hash); got != testCase.want {
				t.Fatalf("raylineRoutesCheckpoint() = %q, want %q", got, testCase.want)
			}
		})
	}
}

// The encoder is configured for a dispatched turn that streams for minutes
// and wraps whatever context it is handed, so the lookup has to impose its
// own bound or inherit one measured in minutes.
func TestRaylineRoutesDeadlineDefaultsToTheAuthorityBudget(t *testing.T) {
	t.Parallel()
	settings := config.RaylineARCRoutesAPIConfig{Enabled: true}
	if got := settings.EffectiveDeadline(); got != 1500*time.Millisecond {
		t.Fatalf("EffectiveDeadline() = %v, want the 1.5s authority budget", got)
	}
	settings.DeadlineMS = 800
	if got := settings.EffectiveDeadline(); got != 800*time.Millisecond {
		t.Fatalf("EffectiveDeadline() = %v, want 800ms", got)
	}
}

// Expiry is read from the lookup's own context because the selector converts
// every failure to a bounded class that does not unwrap to a context error.
// A 503 would send the caller into fallback and read as an outage; running
// out of budget is neither.
func TestRaylineRoutesReportsAnExpiredLookupAsATimeout(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	expired, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	response := router.raylineRoutesFailure(
		nil,
		expired,
		"rte_1234abcd",
		errors.New("rayline ARC selection failed: encoder"),
	)
	if code := immediateStatusCode(t, response); code != 504 {
		t.Fatalf("status = %d, want 504", code)
	}
	live, liveCancel := context.WithCancel(context.Background())
	defer liveCancel()
	response = router.raylineRoutesFailure(
		nil,
		live,
		"rte_1234abcd",
		errors.New("rayline ARC selection failed: encoder"),
	)
	if code := immediateStatusCode(t, response); code != 503 {
		t.Fatalf("status = %d, want 503 while the budget still holds", code)
	}
}

// The opt-out header is honoured for traffic this router forwards. A lookup
// is not forwarded, so honouring it would hand the body to the upstream and
// execute the turn this endpoint promises not to execute.
func TestRaylineRoutesIgnoresTheProcessingOptOut(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	ctx := routesContext(nil)
	ctx.SkipProcessing = true
	response, err := router.HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"model":"rayline-router"}`)},
	}, ctx)
	if err != nil {
		t.Fatalf("HandleRequestBody() error = %v", err)
	}
	if response.GetImmediateResponse() == nil {
		t.Fatal("a skip-processing lookup was forwarded upstream, want it answered here")
	}
	if code := immediateStatusCode(t, response); code != 400 {
		t.Fatalf("status = %d, want the endpoint's own 400", code)
	}
}

// Header-phase refusals are produced before the body-phase error producer
// runs, so without claiming them they are rewritten into the source format's
// error shape and a caller sees two different contracts from one endpoint.
func TestRaylineRoutesHeaderErrorsUseTheRoutesEnvelope(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		enabled bool
		method  string
		status  int
		errType string
	}{
		"disabled is not found": {enabled: false, method: "POST", status: 404, errType: "not_found_error"},
		"wrong method":          {enabled: true, method: "GET", status: 405, errType: "invalid_request_error"},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			ctx := routesContext(nil)
			response := routesRouter(testCase.enabled).
				validateRequestHeaders(testCase.method, raylineRoutesAPIPath, ctx)
			if response == nil {
				t.Fatalf("validateRequestHeaders() = nil, want %d", testCase.status)
			}
			if code := immediateStatusCode(t, response); code != testCase.status {
				t.Fatalf("status = %d, want %d", code, testCase.status)
			}
			if !ctx.ImmediateResponseEncoded {
				t.Fatal("the header-phase refusal was not claimed, so it will be re-encoded")
			}
			var envelope struct {
				Type  string `json:"type"`
				Error struct {
					Type string `json:"type"`
				} `json:"error"`
			}
			if err := json.Unmarshal(response.GetImmediateResponse().Body, &envelope); err != nil {
				t.Fatalf("error body is not JSON: %v", err)
			}
			if envelope.Type != "error" || envelope.Error.Type != testCase.errType {
				t.Fatalf("envelope = %+v, want an Anthropic %s", envelope, testCase.errType)
			}
		})
	}
}

// A caller can state the dialect instead of having it inferred. The body
// stays exactly what they were going to send; the one thing it cannot always
// say about itself is which of two overlapping dialects it is written in.
func TestRaylineRoutesFormatHeaderWins(t *testing.T) {
	t.Parallel()
	// A body that would otherwise infer as Chat.
	body := []byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}`)
	for hint, want := range map[string]llmprotocol.WireFormat{
		"anthropic": llmprotocol.AnthropicMessagesV1,
		"chat":      llmprotocol.OpenAIChatV1,
		"responses": llmprotocol.OpenAIResponsesV1,
	} {
		format, _, detail := readRaylineRoutesBody(body, false, hint)
		if detail != "" {
			t.Fatalf("hint %q detail = %q, want none", hint, detail)
		}
		if format != want {
			t.Fatalf("hint %q gave %q, want %q", hint, format, want)
		}
	}
	// An unrecognised hint falls back to inference rather than refusing a
	// well-formed request over a header typo.
	format, _, detail := readRaylineRoutesBody(body, false, "klingon")
	if detail != "" || format != llmprotocol.OpenAIChatV1 {
		t.Fatalf("bad hint gave (%q, %q), want inference", format, detail)
	}
}

// Two arms can serve one model through different providers at different
// prices, so the model alone cannot be executed faithfully.
func TestRaylineRoutesPublishesTheSelectedArm(t *testing.T) {
	t.Parallel()
	payload := raylineRoutesPayload(
		"rte_1234abcd5678efab",
		"arc-2026-09-12.c82a1f3e",
		routerruntime.RouteDecision{
			WorkerModel:    "deepseek/deepseek-v4-pro",
			SelectedWorker: "deepseek-v4-pro@thinking-off",
			Provider:       "deepinfra",
		},
		nil,
		time.Millisecond,
	)
	body := marshalRoutesPayload(t, payload)
	if body["worker"] != "deepseek-v4-pro@thinking-off" {
		t.Fatalf("worker = %v, want the selected arm", body["worker"])
	}
	if body["provider"] != "deepinfra" {
		t.Fatalf("provider = %v, want the arm's provider", body["provider"])
	}
}

// An Anthropic agentic turn names its dialect only through its tools: the
// roles are ordinary, the content is blocks, and max_tokens is shared. Read
// as Chat, every such request is refused by a codec that has no tool_use
// block and no input_schema.
func TestRaylineRoutesReadsToolShapesAsDialectMarkers(t *testing.T) {
	t.Parallel()
	for name, testCase := range map[string]struct {
		body string
		want llmprotocol.WireFormat
	}{
		"anthropic by tool input_schema": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"name":"search","input_schema":{"type":"object"}}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		"anthropic by tool_use block": {
			body: `{"max_tokens":16,"messages":[{"role":"assistant","content":` +
				`[{"type":"tool_use","id":"t1","name":"search","input":{}}]}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		"anthropic by tool_result block": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":` +
				`[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		"anthropic by thinking block": {
			body: `{"max_tokens":16,"messages":[{"role":"assistant","content":` +
				`[{"type":"thinking","thinking":"...","signature":"s"}]}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		// A server tool declares a type and nothing else -- no name, no
		// schema -- so it is the one tool shape with no member to key on.
		// Read as Chat, the commonest web-search request is refused by a
		// codec that has no such tool, while the selector is told the tool
		// was there.
		"anthropic by schema-less server tool": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"type":"web_search_20250305","name":"web_search"}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		"anthropic by server tool with no name at all": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"type":"code_execution_20250522"}]}`,
			want: llmprotocol.AnthropicMessagesV1,
		},
		// The mirror image: Chat nests the callable under function, and a
		// body whose only marker is that must not read as Anthropic.
		"chat by tool function": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"type":"function","function":{"name":"search"}}]}`,
			want: llmprotocol.OpenAIChatV1,
		},
		// Chat's own tool spellings are not Anthropic server tools. `custom`
		// is Chat's second tool type, and reading it as a server tool would
		// push an ordinary Chat body onto the Anthropic codec.
		"chat custom tool is not a server tool": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
				`"tools":[{"type":"custom","custom":{"name":"grep"}}]}`,
			want: llmprotocol.OpenAIChatV1,
		},
		// Both dialects carry plain text blocks, so they decide nothing.
		"text blocks decide nothing": {
			body: `{"max_tokens":16,"messages":[{"role":"user","content":` +
				`[{"type":"text","text":"hi"}]}]}`,
			want: llmprotocol.OpenAIChatV1,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			format, warnings, detail := readRaylineRoutesBody([]byte(testCase.body), false, "")
			if detail != "" {
				t.Fatalf("detail = %q, want none", detail)
			}
			if format != testCase.want {
				t.Fatalf("format = %q, want %q", format, testCase.want)
			}
			// A marker the body actually carries is knowledge, not a guess,
			// so nothing may claim it was inferred. The text-block case is
			// the deliberate exception: it carries no marker at all.
			inferred := false
			for _, warning := range warnings {
				if strings.HasPrefix(warning, "format_inferred:") {
					inferred = true
				}
			}
			markerFree := name == "text blocks decide nothing" ||
				name == "chat custom tool is not a server tool"
			if wantInferred := markerFree; inferred != wantInferred {
				t.Fatalf("format_inferred present = %v, want %v", inferred, wantInferred)
			}
		})
	}
}

// An override some body shapes silently outrank is not an override. The
// Responses inference used to run before the header was read, so a caller
// stating the dialect on an input body was ignored with no way to tell.
func TestRaylineRoutesFormatHeaderOutranksTheResponsesInference(t *testing.T) {
	t.Parallel()
	body := []byte(`{"input":[{"role":"user","content":"hi"}]}`)
	format, _, detail := readRaylineRoutesBody(body, false, "chat")
	if detail != "" {
		t.Fatalf("detail = %q, want none", detail)
	}
	if format != llmprotocol.OpenAIChatV1 {
		t.Fatalf("format = %q, want the declared chat", format)
	}
	// With no header the body's own shape still decides.
	if format, _, _ := readRaylineRoutesBody(body, false, ""); format != llmprotocol.OpenAIResponsesV1 {
		t.Fatalf("format = %q, want the inferred responses", format)
	}
}

// A 429 with no Retry-After is read by most clients as "retry now", onto the
// same lease or encoder queue that produced the contention.
func TestRaylineRoutesContentionSaysWhenToRetry(t *testing.T) {
	t.Parallel()
	ctx := routesContext(nil)
	response := routesRouter(true).raylineRoutesFailure(
		ctx,
		context.Background(),
		"rte_1234abcd5678efab",
		routerruntime.ErrRouteDecisionContended,
	)
	if code := immediateStatusCode(t, response); code != 429 {
		t.Fatalf("status = %d, want 429", code)
	}
	found := ""
	for _, header := range response.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if strings.EqualFold(header.GetHeader().GetKey(), "retry-after") {
			found = string(header.GetHeader().GetRawValue())
		}
	}
	if found == "" {
		t.Fatal("the contended 429 carries no Retry-After, so clients will retry immediately")
	}
	if found != "1" {
		t.Fatalf("Retry-After = %q, want the one-turn delay", found)
	}
	// Content-type must survive the append rather than be replaced by it.
	contentType := false
	for _, header := range response.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if strings.EqualFold(header.GetHeader().GetKey(), "content-type") {
			contentType = true
		}
	}
	if !contentType {
		t.Fatal("appending Retry-After dropped content-type")
	}
}

// Envoy sends no body callback for a header message that ended the stream, so
// a bodyless POST finds no body phase to answer it and is forwarded to an
// upstream that has never heard of this path.
func TestRaylineRoutesRefusesABodylessPost(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	ctx := routesContext(nil)
	response, err := router.handleRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers:     routesHeaderMap("POST", raylineRoutesAPIPath),
			EndOfStream: true,
		},
	}, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	if response.GetImmediateResponse() == nil {
		t.Fatal("a bodyless route lookup was continued upstream, want it refused here")
	}
	if code := immediateStatusCode(t, response); code != 400 {
		t.Fatalf("status = %d, want 400", code)
	}
	if !ctx.ImmediateResponseEncoded {
		t.Fatal("the refusal was not claimed, so it will be re-encoded into another contract")
	}
}

// The same message with a body to follow must still reach the body phase.
func TestRaylineRoutesAdmitsAPostThatCarriesABody(t *testing.T) {
	t.Parallel()
	ctx := routesContext(nil)
	response, err := routesRouter(true).handleRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers:     routesHeaderMap("POST", raylineRoutesAPIPath),
			EndOfStream: false,
		},
	}, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	if response.GetImmediateResponse() != nil {
		t.Fatalf("a POST with a body was refused at the header phase: %d", immediateStatusCode(t, response))
	}
}

// A branch names a lane inside a conversation, so alone it names a lane in no
// conversation. Dropped silently, the caller sees a plausible route and loses
// every branch's continuity with no symptom to read.
func TestRaylineRoutesWarnsThatALoneBranchWasDropped(t *testing.T) {
	t.Parallel()
	settings := &config.RaylineARCRoutesAPIConfig{Enabled: true, EpisodeWrites: true}
	warnings := raylineRoutesHeaderWarnings(
		routesContext(map[string]string{raylineRoutesBranchHeader: "subagent-2"}),
		settings,
	)
	if !routesWarns(warnings, "branch_ignored:") {
		t.Fatalf("warnings = %v, want branch_ignored", warnings)
	}
	// With the conversation present the branch is honoured, so there is
	// nothing to warn about.
	warnings = raylineRoutesHeaderWarnings(
		routesContext(map[string]string{
			raylineRoutesSessionHeader: "conv-1",
			raylineRoutesBranchHeader:  "subagent-2",
		}),
		settings,
	)
	if routesWarns(warnings, "branch_ignored:") {
		t.Fatalf("warnings = %v, want no branch_ignored", warnings)
	}
}

func routesWarns(warnings []string, prefix string) bool {
	for _, warning := range warnings {
		if strings.HasPrefix(warning, prefix) {
			return true
		}
	}
	return false
}

// routesHeaderMap builds the header message Envoy sends for a request line.
func routesHeaderMap(method, path string) *core.HeaderMap {
	return &core.HeaderMap{
		Headers: []*core.HeaderValue{
			{Key: ":method", RawValue: []byte(method)},
			{Key: ":path", RawValue: []byte(path)},
		},
	}
}

// Both consult surfaces share one RouteDecision method, and only /v1/routes
// has a routes_api block to read. Letting that block disambiguate for the
// legacy consult moved it off its fail-closed ambiguity the moment some
// unrelated decision turned the new endpoint on -- onto a policy its own
// caller never selected.
func TestDecisionOnlyRoutingTargetHonoursTheSurface(t *testing.T) {
	t.Parallel()
	router := twoARCDecisionRouter()

	decision, _, err := router.decisionOnlyRoutingTarget(routerruntime.RouteDecisionSurfaceRoutes)
	if err != nil {
		t.Fatalf("routes surface: err = %v, want the claiming decision", err)
	}
	if decision.Name != "claims-routes" {
		t.Fatalf("routes surface picked %q, want the decision that enabled routes_api", decision.Name)
	}

	if _, _, err := router.decisionOnlyRoutingTarget(routerruntime.RouteDecisionSurfaceConsult); err == nil {
		t.Fatal("the legacy consult resolved a target from another surface's claim, want its own ambiguity error")
	} else if !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("consult err = %v, want the ambiguity refusal", err)
	}

	// One ARC decision is unambiguous for both, claim or no claim.
	if _, _, err := routesRouter(true).decisionOnlyRoutingTarget(routerruntime.RouteDecisionSurfaceConsult); err != nil {
		t.Fatalf("single-decision consult: err = %v, want the only decision", err)
	}
}

// The adapter must name its surface. An unset one reads as the legacy
// consult, which resolves a different decision -- and a Surface field that
// exists, is documented, and is never set looks exactly like one that works.
func TestRaylineRoutesNamesItsSurface(t *testing.T) {
	t.Parallel()
	if routerruntime.RouteDecisionSurfaceRoutes == routerruntime.RouteDecisionSurfaceConsult {
		t.Fatal("the two surfaces are indistinguishable, so naming one decides nothing")
	}
	consult := raylineRoutesConsult(
		[]byte(`{"messages":[]}`),
		"rte_1234abcd5678efab",
		"conv-1",
		false,
		llmprotocol.AnthropicMessagesV1,
	)
	if consult.Surface != routerruntime.RouteDecisionSurfaceRoutes {
		t.Fatalf("Surface = %q, want %q", consult.Surface, routerruntime.RouteDecisionSurfaceRoutes)
	}
}

func twoARCDecisionRouter() *OpenAIRouter {
	arc := func(name string, routes bool) config.Decision {
		return config.Decision{
			Name: name,
			Algorithm: &config.AlgorithmConfig{
				Type:    config.RaylineARCAlgorithmType,
				OnError: "fail_closed",
				RaylineARC: &config.RaylineARCAlgorithmConfig{
					RoutesAPI: config.RaylineARCRoutesAPIConfig{Enabled: routes},
				},
			},
		}
	}
	return &OpenAIRouter{Config: &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{
			arc("plain-arc", false),
			arc("claims-routes", true),
		}},
	}}
}

// slowCommitTransaction blocks CommitOnHeaders until released, so a lookup
// whose budget expires mid-commit can be observed without a real episode
// store.
type slowCommitTransaction struct {
	release chan struct{}
	// outcome records how the blocked commit ended: nil for released, the
	// context's error if the commit was cancelled while it waited. Recorded
	// inside the call, because the context is cancelled as soon as the call
	// returns and reading it afterwards says nothing.
	outcome chan error
	aborted chan string
}

func (t *slowCommitTransaction) ValidateDispatch(context.Context) error { return nil }

func (t *slowCommitTransaction) CommitOnHeaders(ctx context.Context, _ int) error {
	select {
	case <-t.release:
		t.outcome <- nil
		return nil
	case <-ctx.Done():
		t.outcome <- ctx.Err()
		return ctx.Err()
	}
}

func (t *slowCommitTransaction) Abort(_ context.Context, class string) error {
	t.aborted <- class
	return nil
}

func (t *slowCommitTransaction) Settle(context.Context, selectionActualOutcome) error { return nil }

// The commit runs detached so an expired lookup does not strand the lease,
// but the CALLER must stop waiting at its own deadline. Detaching the run and
// the wait together let a slow commit add its whole finalization timeout on
// top of deadline_ms and still answer 200.
func TestTrackedLookupStopsWaitingAtItsDeadline(t *testing.T) {
	t.Parallel()
	transaction := &slowCommitTransaction{
		release: make(chan struct{}),
		outcome: make(chan error, 1),
		aborted: make(chan string, 1),
	}
	requestContext := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner("test", transaction),
	}
	lookupContext, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()

	started := time.Now()
	err := commitDecisionOnlyEpisode(lookupContext, requestContext)
	waited := time.Since(started)

	if err == nil {
		t.Fatal("a lookup that ran out of budget mid-commit reported success")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want it to carry the deadline", err)
	}
	// The whole point: bounded by deadline_ms, not by episodeFinalizeTimeout.
	if waited >= episodeFinalizeTimeout {
		t.Fatalf("waited %v, want well under the finalization timeout %v", waited, episodeFinalizeTimeout)
	}

	// ...and the episode is RELEASED rather than committed. The caller got a
	// 504 and no route, so committing would record the selected arm as the
	// previous arm of a turn nobody ran, and the retry would then be scored
	// against it. The lease still has to be resolved, which is why the abort
	// runs at all.
	select {
	case class := <-transaction.aborted:
		if class != raylineARCRoutesDeadlineAbortClass {
			t.Fatalf("abort class = %q, want %q", class, raylineARCRoutesDeadlineAbortClass)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the timed-out lookup neither committed nor released its episode")
	}
	// The commit itself was cancelled, not left running to land later.
	select {
	case outcome := <-transaction.outcome:
		if outcome == nil {
			t.Fatal("the commit landed after the lookup reported a timeout, advancing the episode")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the cancelled commit never returned")
	}
}

// Inside the budget nothing changes: the commit is awaited, because the
// episode turn index this endpoint reports is the one the commit stores.
func TestTrackedLookupAwaitsACommitInsideItsBudget(t *testing.T) {
	t.Parallel()
	transaction := &slowCommitTransaction{
		release: make(chan struct{}),
		outcome: make(chan error, 1),
		aborted: make(chan string, 1),
	}
	close(transaction.release)
	requestContext := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner("test", transaction),
	}
	lookupContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := commitDecisionOnlyEpisode(lookupContext, requestContext); err != nil {
		t.Fatalf("commitDecisionOnlyEpisode() = %v, want the commit to be awaited", err)
	}
	select {
	case outcome := <-transaction.outcome:
		if outcome != nil {
			t.Fatalf("commit outcome = %v, want a clean commit", outcome)
		}
	default:
		t.Fatal("the function returned before the commit ran")
	}
}

// `"tools": []` is how clients commonly serialise an absent tool collection.
// Warning on it claims the route was shaped by tools the projection never
// saw -- and on a cell that encodes names, claims they influenced a decision
// they took no part in.
func TestRaylineRoutesSaysNothingAboutAnEmptyToolList(t *testing.T) {
	t.Parallel()
	for name, body := range map[string]string{
		"empty list": `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":[]}`,
		"absent":     `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`,
		"not a list": `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],"tools":{}}`,
	} {
		for _, encodesNames := range []bool{false, true} {
			t.Run(name, func(t *testing.T) {
				_, warnings, detail := readRaylineRoutesBody([]byte(body), encodesNames, "")
				if detail != "" {
					t.Fatalf("detail = %q, want none", detail)
				}
				if routesWarns(warnings, "tools_not_encoded:") ||
					routesWarns(warnings, "tool_schemas_not_encoded:") {
					t.Fatalf("warnings = %v, want nothing said about tools", warnings)
				}
			})
		}
	}
	// A body that does declare a tool still explains what happened to it.
	withTools := `{"max_tokens":16,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"name":"search","input_schema":{"type":"object"}}]}`
	_, warnings, _ := readRaylineRoutesBody([]byte(withTools), false, "")
	if !routesWarns(warnings, "tools_not_encoded:") {
		t.Fatalf("warnings = %v, want the dropped-tools warning", warnings)
	}
}

// A decode failure has to name the codec that actually ran. The message is
// the only thing telling a caller which contract their body was read under,
// and naming the wrong one sends them to fix the wrong end.
func TestRaylineRoutesDecodeFailureNamesTheDetectedFormat(t *testing.T) {
	t.Parallel()
	ctx := routesContext(map[string]string{raylineRoutesFormatHeader: "anthropic"})
	// The path-derived default before the body phase runs.
	ctx.SourceFormat = llmprotocol.OpenAIChatV1

	response := routesRouter(true).handleRaylineRoutesAPI(
		[]byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}`),
		ctx,
	)
	// Selection is unavailable on this bare router, so the lookup fails after
	// the body was read -- which is exactly the point: by then the format the
	// body was read under must be the one an error would name.
	if response == nil {
		t.Fatal("handleRaylineRoutesAPI() = nil, want an answer")
	}
	if wireFormatOf(ctx) != llmprotocol.AnthropicMessagesV1 {
		t.Fatalf("recorded format = %q, want the declared anthropic", wireFormatOf(ctx))
	}
}

// directCloseEncoder has the bare EncoderClient's close signature: one
// session, no replica list.
type directCloseEncoder struct {
	closed []string
	err    error
}

func (e *directCloseEncoder) CloseSession(_ context.Context, episodeIDHash string) error {
	e.closed = append(e.closed, episodeIDHash)
	return e.err
}

// poolCloseEncoder has the pool's.
type poolCloseEncoder struct{ closed []string }

func (e *poolCloseEncoder) CloseSession(
	_ context.Context,
	episodeIDHash string,
	_ []string,
) (raylinearc.EncoderCloseReport, error) {
	e.closed = append(e.closed, episodeIDHash)
	return raylinearc.EncoderCloseReport{Attempted: 1, Closed: 1}, nil
}

func retainedARCConfig(retained bool) *config.RaylineARCAlgorithmConfig {
	capabilities := []string{"chunked_causal_mean"}
	if retained {
		capabilities = append(capabilities, config.RaylineARCCapabilityResumableMean)
	}
	return &config.RaylineARCAlgorithmConfig{
		Encoder: config.RaylineARCEncoderConfig{RequiredCapabilities: capabilities},
	}
}

// The supported single-base_url deployment builds a bare EncoderClient, whose
// close takes no replica list. Matching only the pool's shape left the close
// unwired there, so every ephemeral lookup stranded one retained session --
// and sustained lookup traffic then evicted the live conversation prefixes it
// shares the encoder with.
func TestSessionCloseIsWiredForTheDirectEncoderClient(t *testing.T) {
	t.Parallel()
	encoder := &directCloseEncoder{}
	closeSession := raylineARCCloseSessionFor(encoder, retainedARCConfig(true))
	if closeSession == nil {
		t.Fatal("no close wired for a retained single-client encoder, so its sessions leak")
	}
	report, err := closeSession(context.Background(), "episode-hash", nil)
	if err != nil {
		t.Fatalf("closeSession() error = %v", err)
	}
	if len(encoder.closed) != 1 || encoder.closed[0] != "episode-hash" {
		t.Fatalf("closed = %v, want the episode closed once", encoder.closed)
	}
	// One client is one replica.
	if report.Attempted != 1 || report.Closed != 1 {
		t.Fatalf("report = %+v, want one attempted and one closed", report)
	}
}

func TestSessionCloseStillPrefersThePoolShape(t *testing.T) {
	t.Parallel()
	encoder := &poolCloseEncoder{}
	closeSession := raylineARCCloseSessionFor(encoder, retainedARCConfig(true))
	if closeSession == nil {
		t.Fatal("no close wired for a pool encoder")
	}
	if _, err := closeSession(context.Background(), "episode-hash", []string{"a"}); err != nil {
		t.Fatalf("closeSession() error = %v", err)
	}
	if len(encoder.closed) != 1 {
		t.Fatalf("closed = %v, want the pool's own close to run", encoder.closed)
	}
}

// An encoder that retains nothing refuses the close outright, so wiring one
// would log a failure per lookup for housekeeping that was never needed.
func TestSessionCloseIsUnwiredWhenNothingIsRetained(t *testing.T) {
	t.Parallel()
	if closeSession := raylineARCCloseSessionFor(&directCloseEncoder{}, retainedARCConfig(false)); closeSession != nil {
		t.Fatal("a close was wired for an encoder that retains no session")
	}
}

// uncancellableCommitTransaction ignores its context, as MemoryEpisodeStore's
// Commit does. Cancelling such a commit does not stop it, so a lookup that
// only cancelled and walked away could not know whether the episode advanced.
type uncancellableCommitTransaction struct {
	release   chan struct{}
	committed chan struct{}
	aborted   chan string
}

func (t *uncancellableCommitTransaction) ValidateDispatch(context.Context) error { return nil }

func (t *uncancellableCommitTransaction) CommitOnHeaders(_ context.Context, _ int) error {
	<-t.release
	t.committed <- struct{}{}
	return nil
}

func (t *uncancellableCommitTransaction) Abort(_ context.Context, class string) error {
	t.aborted <- class
	return nil
}

func (t *uncancellableCommitTransaction) Settle(context.Context, selectionActualOutcome) error {
	return nil
}

// Cancelling establishes nothing on a store that ignores cancellation. The
// timeout path has to WAIT for the outcome, or it issues an abort that
// silently does nothing against a transaction that has already committed --
// and reports a released episode that in fact advanced.
func TestTimedOutLookupLearnsWhetherTheCommitLanded(t *testing.T) {
	t.Parallel()
	transaction := &uncancellableCommitTransaction{
		release:   make(chan struct{}),
		committed: make(chan struct{}, 1),
		aborted:   make(chan string, 1),
	}
	requestContext := &RequestContext{
		SelectionTransaction: newSelectionTransactionOwner("test", transaction),
	}
	lookupContext, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	// Let the commit land right as the deadline fires.
	go func() {
		time.Sleep(40 * time.Millisecond)
		close(transaction.release)
	}()

	err := commitDecisionOnlyEpisode(lookupContext, requestContext)
	if err == nil || !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want the deadline", err)
	}
	select {
	case <-transaction.committed:
	case <-time.After(2 * time.Second):
		t.Fatal("the uncancellable commit never ran")
	}
	// No abort was issued, because there was nothing left to release: the
	// commit had already landed and an abort would have been a no-op reported
	// as a release.
	select {
	case class := <-transaction.aborted:
		t.Fatalf("aborted with %q after the commit had already landed", class)
	case <-time.After(100 * time.Millisecond):
	}
}
