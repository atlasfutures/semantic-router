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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// wireSignalFixture is a request body captured from a released harness
// binary (see the provenance in each file).
func wireSignalFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "selection", "raylinearc", "testdata", "wire_signals", name))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Body json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture.Body
}

// Each captured harness call is read as what it is: the title and
// summarization calls are side calls, the first request of a compacted
// context is a compaction, and a main turn is left unclassified.
func TestRaylineARCPolicyTurnSignalsReadCapturedHarnessCalls(t *testing.T) {
	cases := []struct {
		fixture    string
		format     string
		kind       raylinearc.PolicyCallKind
		source     string
		compaction bool
	}{
		{"claude_code_2.1.280_main_turn.json", policyFormatAnthropic, raylinearc.PolicyCallUnknown, "", false},
		{"claude_code_2.1.280_title.json", policyFormatAnthropic, raylinearc.PolicyCallSide, "claude_code_title", false},
		{"claude_code_2.1.280_compaction_request.json", policyFormatAnthropic, raylinearc.PolicyCallSide, "claude_code_compaction_request", false},
		{"claude_code_2.1.280_post_compaction.json", policyFormatAnthropic, raylinearc.PolicyCallUnknown, "", true},
		{"codex_0.154.0_main_turn.json", policyFormatResponses, raylinearc.PolicyCallUnknown, "", false},
		{"codex_0.154.0_compaction_request.json", policyFormatResponses, raylinearc.PolicyCallSide, "codex_compaction_request", false},
		{"codex_0.154.0_post_compaction.json", policyFormatResponses, raylinearc.PolicyCallUnknown, "", true},
		{"codex_0.154.0_post_compaction_next.json", policyFormatResponses, raylinearc.PolicyCallUnknown, "", true},
	}
	for _, c := range cases {
		var body struct {
			System   json.RawMessage   `json:"system"`
			Messages []json.RawMessage `json:"messages"`
			Input    []json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(wireSignalFixture(t, c.fixture), &body); err != nil {
			t.Fatal(err)
		}
		request := raylinearc.PolicyClientRequest{System: body.System, Input: body.Input}
		messages := body.Messages
		if c.format == policyFormatResponses {
			messages = body.Input
		}
		signals := raylineARCPolicyTurnSignals(nil, c.format, request, messages, "episode")
		compaction := signals.Compaction != nil && signals.Compaction.SummaryDigest != ""
		if signals.CallKind != c.kind || signals.CallKindSource != c.source || compaction != c.compaction {
			t.Errorf("%s: kind %q source %q compaction %v; want %q %q %v",
				c.fixture, signals.CallKind, signals.CallKindSource, compaction, c.kind, c.source, c.compaction)
		}
	}
}

// policyLeaseRouter is a strict policy-service cell on an in-memory episode
// store, its turns arriving as OpenAI chat.
func policyLeaseRouter(t *testing.T) (*OpenAIRouter, *config.AlgorithmConfig) {
	t.Helper()
	router, _, algorithm := missingSessionRequestContext(t, "")
	algorithm.RaylineARC.PolicyService = &config.RaylineARCPolicyServiceConfig{TrustTurnSignalHeaders: true}
	return router, algorithm
}

func policyLeaseTurn(router *OpenAIRouter, algorithm *config.AlgorithmConfig, body string, headers map[string]string) (*RequestContext, string) {
	ctx := coalesceRequestContext(body)
	for name, value := range headers {
		ctx.Headers[name] = value
	}
	arc := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
	return ctx, arc.PreparationFailure
}

// A side call commits nothing, so it must not wait on the lease a main turn
// holds for the whole of its stream: it is served at once and leaves that
// lease where it was. Two main turns still contend as before.
func TestRaylineARCSideCallDoesNotWaitOnTheMainTurnLease(t *testing.T) {
	sideCalls := []struct {
		name string
		turn func(*testing.T, *OpenAIRouter, *config.AlgorithmConfig) (*RequestContext, string)
	}{
		{"call-kind header", func(_ *testing.T, router *OpenAIRouter, algorithm *config.AlgorithmConfig) (*RequestContext, string) {
			return policyLeaseTurn(router, algorithm, `{"turn":"title"}`, map[string]string{raylineARCCallKindHeader: "side"})
		}},
		{"captured Claude Code title call", func(t *testing.T, router *OpenAIRouter, algorithm *config.AlgorithmConfig) (*RequestContext, string) {
			ctx := coalesceRequestContext("")
			ctx.SourceFormat = llmprotocol.AnthropicMessagesV1
			ctx.RaylineARCRawBody = wireSignalFixture(t, "claude_code_2.1.280_title.json")
			arc := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
			return ctx, arc.PreparationFailure
		}},
	}
	for _, sideCall := range sideCalls {
		t.Run(sideCall.name, func(t *testing.T) {
			router, algorithm := policyLeaseRouter(t)
			main, failure := policyLeaseTurn(router, algorithm, `{"turn":"main"}`, nil)
			if failure != "" {
				t.Fatalf("main turn failure = %q", failure)
			}
			// The main turn is mid-stream: its lease is held.
			started := time.Now()
			side, failure := sideCall.turn(t, router, algorithm)
			if failure != "" {
				t.Fatalf("a side call during a main turn's stream failed with %q", failure)
			}
			if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
				t.Fatalf("the side call waited %v for the main turn's lease", elapsed)
			}
			side.RaylineARCTransaction.markSelection(1, 5)
			side.RaylineARCTransaction.markPolicyState(nil, true)
			if err := side.RaylineARCTransaction.commit(context.Background(), side); err != nil {
				t.Fatalf("side call commit = %v", err)
			}
			// The side call released nothing: another main turn still contends.
			if _, contended := policyLeaseTurn(router, algorithm, `{"turn":"other"}`, nil); contended != "episode_timeout" {
				t.Fatalf("a second main turn during the first = %q, want episode_timeout", contended)
			}
			main.RaylineARCTransaction.markSelection(0, 5)
			if err := main.RaylineARCTransaction.commit(context.Background(), main); err != nil {
				t.Fatalf("the main turn lost its lease to the side call: %v", err)
			}
			probe, failure := policyLeaseTurn(router, algorithm, `{"turn":"probe"}`, nil)
			if failure != "" {
				t.Fatalf("probe failure = %q", failure)
			}
			if turn := probe.RaylineARCTransaction.state.TurnIndex; turn != 1 {
				t.Fatalf("turn index = %d, want 1: the side call counted or the main turn did not", turn)
			}
			_ = probe.RaylineARCTransaction.abort(context.Background(), "test")
		})
	}
}

// busyFixture is a policy selector whose episode acquire timeout is one
// second, the bound on how long a decision waits out session_busy.
func busyFixture(t *testing.T) *policySelectorFixture {
	t.Helper()
	fixture := newPolicySelectorFixture(t, "")
	fixture.decision.Algorithm.RaylineARC.Episode.AcquireTimeoutSeconds = 1
	fixture.selector.arm(&raylineARCArmedComponents{
		scorer:    newPolicyServiceScorer(&config.RouterConfig{}, fixture.decision),
		admission: raylinearc.NewAdmissionGate(0),
		policy:    raylinearc.NewPolicyServiceClient(raylinearc.PolicyServiceConfig{BaseURL: fixture.fake.URL(), TotalTimeout: 5 * time.Second}),
	})
	fixture.fake.chooseWith(func(request raylinearc.PolicyDecisionRequest) string {
		return request.Selection.AvailableActionIDs[0]
	})
	return fixture
}

// A side call takes no episode lease, so its decision can overlap a main
// turn's at the service, which answers the later one session_busy. That
// request waits the other decision out instead of being refused.
func TestPolicyDecisionWaitsOutAnOverlappingDecision(t *testing.T) {
	fixture := busyFixture(t)
	fixture.fake.failFirstCalls("session_busy", 2)
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	if _, err := fixture.selectOn(t, state, body); err != nil {
		t.Fatalf("a decision overlapping another was refused: %v", err)
	}
	if calls := len(fixture.fake.received()); calls != 3 {
		t.Fatalf("decide calls = %d, want 3 (two busy, one served)", calls)
	}
}

// A service that stays busy past the episode's acquire timeout is still
// refused as session_busy.
func TestPolicyDecisionStillRefusedWhenBusyOutlastsTheTimeout(t *testing.T) {
	fixture := busyFixture(t)
	fixture.fake.failNext("session_busy")
	state, _ := raylinearc.NewEpisodeState(2)
	body := policyTestRequest(t, map[string]any{"role": "user", "content": "fix the bug"})
	started := time.Now()
	_, err := fixture.selectOn(t, state, body)
	var failure *raylineARCSelectionFailure
	if !errors.As(err, &failure) || failure.class != "policy_service_session_busy" {
		t.Fatalf("persistently busy service = %v, want policy_service_session_busy", err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("waited %v on a busy service, beyond the 1s acquire timeout", elapsed)
	}
}

// A Responses request may send its input as one string; codex's compaction
// directive sent that way is still its summarization side call.
func TestRaylineARCPolicyCallKindReadsResponsesStringInput(t *testing.T) {
	var fixture struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(wireSignalFixture(t, "codex_0.154.0_compaction_request.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	var directive struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
	}
	if err := json.Unmarshal(fixture.Input[len(fixture.Input)-1], &directive); err != nil {
		t.Fatal(err)
	}
	body, err := json.Marshal(map[string]any{"model": "auto", "input": directive.Content[0].Text})
	if err != nil {
		t.Fatal(err)
	}
	kind, source := raylineARCPolicyCallKindOfBody(nil, policyFormatResponses, body, "episode")
	if kind != raylinearc.PolicyCallSide || source != callKindSourceCodexCompactionRq {
		t.Fatalf("string-input compaction directive = %q %q, want a codex side call", kind, source)
	}
	plain, _ := json.Marshal(map[string]any{"model": "auto", "input": "fix the bug"})
	if kind, _ := raylineARCPolicyCallKindOfBody(nil, policyFormatResponses, plain, "episode"); kind != raylinearc.PolicyCallUnknown {
		t.Fatalf("plain string input = %q, want unknown", kind)
	}
}

// An artifact-mode turn commits at its 2xx response headers and releases its
// strict lease there, as before: a long stream must not hold the episode, so
// a second request on it (a parallel subagent sharing the session) is served
// while the first is still streaming.
func TestRaylineARCArtifactTurnReleasesTheLeaseAtTheHeaders(t *testing.T) {
	router, _, algorithm := missingSessionRequestContext(t, "")
	first := coalesceRequestContext(`{"turn":"long stream"}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, first, missingSessionModelRefs(), raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("first turn failure = %q", failure)
	}
	first.RaylineARCTransaction.markSelection(0, 5)
	sendHeaders(t, &OpenAIRouter{}, first, streamingResponseHeaders("200"))
	// The stream has started and has not ended.
	started := time.Now()
	second := coalesceRequestContext(`{"turn":"parallel subagent"}`)
	if failure := router.buildRaylineARCSelectionContext(algorithm, second, missingSessionModelRefs(), raylineARCEpisodeRequired).PreparationFailure; failure != "" {
		t.Fatalf("a second request during an artifact-mode stream failed with %q", failure)
	}
	if elapsed := time.Since(started); elapsed > 500*time.Millisecond {
		t.Fatalf("the second request waited %v for the streaming turn's lease", elapsed)
	}
	if turn := second.RaylineARCTransaction.state.TurnIndex; turn != 1 {
		t.Fatalf("turn index = %d, want 1: the first turn commits at its headers", turn)
	}
	_ = second.RaylineARCTransaction.abort(context.Background(), "test")
	completeStream(&OpenAIRouter{}, first)
	finalizeSelectionProcessTerminal(first)
}

// The turn-signal headers are a gateway's to set. Unless the cell says a
// gateway sets them, a client's own headers are ignored and the body decides,
// so a client cannot mark its turns as side calls or compactions.
func TestRaylineARCTurnSignalHeadersAreIgnoredUnlessTrusted(t *testing.T) {
	spoofed := map[string]string{raylineARCCallKindHeader: "side", raylineARCCompactionHeader: "3"}
	for _, trusted := range []bool{false, true} {
		router, algorithm := policyLeaseRouter(t)
		algorithm.RaylineARC.PolicyService.TrustTurnSignalHeaders = trusted
		ctx := coalesceRequestContext(`{"turn":"main"}`)
		for name, value := range spoofed {
			ctx.Headers[name] = value
		}
		arc := router.buildRaylineARCSelectionContext(algorithm, ctx, missingSessionModelRefs(), raylineARCEpisodeRequired)
		if arc.PreparationFailure != "" {
			t.Fatalf("trusted=%v: failure %q", trusted, arc.PreparationFailure)
		}
		wantKind := raylinearc.PolicyCallUnknown
		if trusted {
			wantKind = raylinearc.PolicyCallSide
		}
		if arc.PolicyCallKind != wantKind || (arc.PolicySignalHeaders[raylineARCCompactionHeader] == "3") != trusted {
			t.Fatalf("trusted=%v: kind %q signal headers %v", trusted, arc.PolicyCallKind, arc.PolicySignalHeaders)
		}
		_ = ctx.RaylineARCTransaction.abort(context.Background(), "test")
	}
}

// Trusted or not, the turn-signal headers speak to this router and are never
// forwarded to a provider.
func TestRaylineARCTurnSignalHeadersAreNotForwarded(t *testing.T) {
	removed := strings.Join(faultInjectionHeadersForRemoval(), ",")
	for _, name := range []string{raylineARCCallKindHeader, raylineARCCompactionHeader} {
		if !strings.Contains(removed, name) {
			t.Fatalf("routed removal list %q does not strip %s", removed, name)
		}
	}
}
