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

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
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
		response := router.validateRequestHeaders(method, raylineRoutesAPIPath)
		if response == nil {
			t.Fatalf("validateRequestHeaders(%q) = nil, want 404", method)
		}
		if code := immediateStatusCode(t, response); code != 404 {
			t.Fatalf("validateRequestHeaders(%q) status = %d, want 404", method, code)
		}
	}
}

func TestRaylineRoutesAdmitsOnlyPost(t *testing.T) {
	t.Parallel()
	router := routesRouter(true)
	if response := router.validateRequestHeaders("POST", raylineRoutesAPIPath); response != nil {
		t.Fatalf("validateRequestHeaders(POST) = %v, want nil", response)
	}
	response := router.validateRequestHeaders("GET", raylineRoutesAPIPath)
	if response == nil {
		t.Fatal("validateRequestHeaders(GET) = nil, want 405")
	}
	if code := immediateStatusCode(t, response); code != 405 {
		t.Fatalf("validateRequestHeaders(GET) status = %d, want 405", code)
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
			body: `{"model":"rayline-router","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`,
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
			got, _, detail := readRaylineRoutesBody([]byte(testCase.body), false)
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
			if _, _, detail := readRaylineRoutesBody([]byte(body), false); detail == "" {
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
	body := []byte(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":16,"tools":[{"name":"edit"}]}`)

	_, dropped, detail := readRaylineRoutesBody(body, false)
	if detail != "" {
		t.Fatalf("readRaylineRoutesBody() detail = %q, want none", detail)
	}
	if len(dropped) != 1 || !strings.HasPrefix(dropped[0], "tools_not_encoded:") {
		t.Fatalf("warnings = %v, want one tools_not_encoded warning", dropped)
	}

	_, encoded, detail := readRaylineRoutesBody(body, true)
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
		"anthropic by required max_tokens": {
			body:    `{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}`,
			want:    llmprotocol.AnthropicMessagesV1,
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
		"ambiguous": {
			body:    `{"messages":[{"role":"user","content":"hi"}]}`,
			want:    llmprotocol.AnthropicMessagesV1,
			certain: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			format, warnings, detail := readRaylineRoutesBody([]byte(testCase.body), false)
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
