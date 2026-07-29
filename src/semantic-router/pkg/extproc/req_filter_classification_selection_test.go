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
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

type selectionResultSelector struct {
	result *selection.SelectionResult
	err    error
}

func (s selectionResultSelector) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	return s.result, s.err
}

func (s selectionResultSelector) Method() selection.SelectionMethod {
	return selection.MethodStatic
}

func (s selectionResultSelector) UpdateFeedback(ctx context.Context, feedback *selection.Feedback) error {
	return nil
}

func (s selectionResultSelector) Tier() selection.AlgorithmTier {
	return selection.TierSupported
}

func (s selectionResultSelector) ExternalDependencies() []selection.Dependency {
	return nil
}

func TestSelectModelFromCandidatesUsesDefaultCandidateOnInvalidSelectionResult(t *testing.T) {
	for _, tc := range []struct {
		name   string
		result *selection.SelectionResult
	}{
		{
			name: "nil result",
		},
		{
			name:   "non candidate result",
			result: &selection.SelectionResult{SelectedModel: "model-c"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			registry := selection.NewRegistry()
			registry.Register(selection.MethodStatic, selectionResultSelector{result: tc.result})

			router := &OpenAIRouter{ModelSelector: registry}
			selected, method, err := router.selectModelFromCandidates(&selection.SelectionContext{
				CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
			}, nil, nil)
			if err != nil {
				t.Fatal(err)
			}

			if selected == nil || selected.Model != "model-a" {
				t.Fatalf("expected default candidate model-a, got %#v", selected)
			}
			if method != string(selection.MethodStatic) {
				t.Fatalf("expected static method, got %q", method)
			}
		})
	}
}

func TestSelectModelFromCandidatesUsesFirstValidDefaultCandidateOnInvalidContext(t *testing.T) {
	router := &OpenAIRouter{}
	selected, method, err := router.selectModelFromCandidates(&selection.SelectionContext{
		CandidateModels: []config.ModelRef{{Model: " "}, {Model: "model-b"}},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	if selected == nil || selected.Model != "model-b" {
		t.Fatalf("expected default candidate model-b, got %#v", selected)
	}
	if method != "" {
		t.Fatalf("expected empty method for invalid context default, got %q", method)
	}
}

func TestSelectModelFromCandidatesFailsClosedForRaylineARCWithoutSessionMutation(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	registry := selection.NewRegistry()
	registry.Register(
		selection.MethodRaylineARC,
		selectionResultSelector{err: fmt.Errorf("private encoder detail")},
	)
	router := &OpenAIRouter{ModelSelector: registry}
	algorithm := &config.AlgorithmConfig{
		Type:    config.RaylineARCAlgorithmType,
		OnError: "fail_closed",
	}
	selected, method, err := router.selectModelFromCandidates(
		&selection.SelectionContext{
			SessionID:       "arc-session",
			CandidateModels: []config.ModelRef{{Model: "a"}, {Model: "b"}},
		},
		algorithm,
		&RequestContext{SessionID: "arc-session"},
	)
	if selected != nil || method != string(selection.MethodRaylineARC) {
		t.Fatalf("selected=%#v method=%q", selected, method)
	}
	var failure *modelSelectionFailure
	if !errors.As(err, &failure) || failure.class != "selector" {
		t.Fatalf("error = %#v, want bounded selector failure", err)
	}
	if _, ok := sessiontelemetry.GetRouterSessionSnapshot(
		"arc-session",
		time.Now(),
	); ok {
		t.Fatal("ARC failure recorded a successful session decision")
	}
}

func TestSelectModelFromCandidatesBindsPrivateARCDispatchContract(t *testing.T) {
	state, err := raylinearc.NewEpisodeState(2)
	if err != nil {
		t.Fatal(err)
	}
	scorer := &fakeARCScorer{
		workerIDs: []string{"worker-a", "worker-b"},
		decision: raylinearc.Decision{
			SelectedArm:     1,
			SelectedWorker:  "worker-b",
			RawScores:       []float32{0.1, 0.9},
			AdjustedScores:  []float32{0.1, 0.9},
			SwitchCostUSD:   []float64{0, 0},
			CacheMissTokens: []int{0, 0},
		},
	}
	encoder := &fakeARCEncoder{
		result: &raylinearc.EncoderResult{
			Embedding:         make([]float32, 1024),
			SerializedTokens:  10,
			FullHistoryTokens: 10,
			ModelRevision:     config.RaylineARCEncoderModelRevision,
		},
	}
	registry := selection.NewRegistry()
	registry.Register(
		selection.MethodRaylineARC,
		newRaylineARCSelector(scorer, encoder, "revision"),
	)
	router := &OpenAIRouter{ModelSelector: registry}
	requestContext := &RequestContext{}
	selected, method, err := router.selectModelFromCandidates(
		validARCSelectionContext(state),
		&config.AlgorithmConfig{
			Type:    config.RaylineARCAlgorithmType,
			OnError: "fail_closed",
		},
		requestContext,
	)
	if err != nil {
		t.Fatal(err)
	}
	if selected == nil || selected.Model != "worker-b" ||
		method != string(selection.MethodRaylineARC) {
		t.Fatalf("selection = %#v, method = %q", selected, method)
	}
	if requestContext.RaylineARCDispatch == nil ||
		requestContext.RaylineARCDispatch.ID != "worker-b" ||
		requestContext.RaylineARCDispatch.Model != "provider/worker-b" {
		t.Fatalf(
			"dispatch contract = %#v",
			requestContext.RaylineARCDispatch,
		)
	}
	if requestContext.VSRRaylineARC == nil {
		t.Fatal("privacy-safe ARC trace was not retained")
	}
}

func TestBuildSelectionContextHashesARCIdentityAndNormalizesOriginalBody(t *testing.T) {
	rawEpisodeID := "private-episode"
	algorithm := &config.AlgorithmConfig{
		Type: config.RaylineARCAlgorithmType,
		RaylineARC: &config.RaylineARCAlgorithmConfig{
			Episode: config.RaylineARCEpisodeConfig{
				IDHeader:              "x-rayline-episode-id",
				AcquireTimeoutSeconds: 1,
				LeaseTTLSeconds:       60,
			},
		},
	}
	episodeStore, err := raylinearc.NewMemoryEpisodeStore(
		raylinearc.MemoryEpisodeStoreConfig{
			MaxEpisodes: 4,
			IdleTTL:     time.Minute,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{
		Config:                 &config.RouterConfig{},
		RaylineARCEpisodeStore: episodeStore,
	}
	requestContext := &RequestContext{
		Headers: map[string]string{
			"x-rayline-episode-id": rawEpisodeID,
		},
		OriginalRequestBody: []byte(
			`{"model":"auto","messages":[{"role":"user","content":"hello"}]}`,
		),
	}
	t.Cleanup(func() {
		router.finalizeRaylineARCAbort(requestContext, "test_cleanup")
	})
	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "a"}, {Model: "b"}},
		"arc",
		"query",
		algorithm,
		"",
		nil,
		requestContext,
	)
	if selCtx.RaylineARC == nil {
		t.Fatal("missing ARC selection context")
	}
	if selCtx.RaylineARC.EpisodeIDHash != raylinearc.HashEpisodeID(rawEpisodeID) {
		t.Fatalf("episode hash = %q", selCtx.RaylineARC.EpisodeIDHash)
	}
	if len(selCtx.RaylineARC.Turns) != 1 ||
		selCtx.RaylineARC.Turns[0].Text != "hello" {
		t.Fatalf("turns = %#v", selCtx.RaylineARC.Turns)
	}
	if strings.Contains(selCtx.RaylineARC.EpisodeIDHash, rawEpisodeID) {
		t.Fatal("raw episode ID retained in ARC context")
	}
}

func TestRaylineARCSelectionFailureReturnsPrivacySafe503(t *testing.T) {
	router := &OpenAIRouter{}
	response, handled := router.decisionEvaluationErrorResponse(
		&modelSelectionFailure{class: "encoder_contract"},
		&RequestContext{RequestID: "request-id"},
	)
	if !handled || response == nil || response.GetImmediateResponse() == nil {
		t.Fatal("ARC failure was not handled as an immediate response")
	}
	immediate := response.GetImmediateResponse()
	if got := int(immediate.GetStatus().GetCode()); got != 503 {
		t.Fatalf("status = %d, want 503", got)
	}
	body := string(immediate.Body)
	if strings.Contains(body, "encoder_contract") ||
		!strings.Contains(body, "Rayline ARC routing unavailable") {
		t.Fatalf("unexpected response body: %s", body)
	}
}

func TestSelectModelFromCandidatesRecordsSingleCandidateInRouterMemory(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)

	router := &OpenAIRouter{}
	reqCtx := &RequestContext{SessionID: "single-candidate-session"}
	selected, method, err := router.selectModelFromCandidates(&selection.SelectionContext{
		SessionID:       "single-candidate-session",
		DecisionName:    "warmup",
		CandidateModels: []config.ModelRef{{Model: "model-a"}},
	}, nil, reqCtx)
	if err != nil {
		t.Fatal(err)
	}

	if selected == nil || selected.Model != "model-a" {
		t.Fatalf("expected model-a, got %#v", selected)
	}
	if method != "single" {
		t.Fatalf("expected single method, got %q", method)
	}

	snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot("single-candidate-session", time.Now())
	if !ok {
		t.Fatal("expected router memory snapshot for single-candidate selection")
	}
	if snapshot.CurrentModel != "model-a" {
		t.Fatalf("expected current model model-a, got %q", snapshot.CurrentModel)
	}
}

func TestSelectorForDecisionMethodBuildsDecisionScopedHybridSelector(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.BackendModels.ModelConfig = map[string]config.ModelParams{
		"current":  {Description: "general chat"},
		"frontier": {Description: "coding specialist"},
	}

	modelSelectionCfg := buildModelSelectionConfig(&cfg)
	registry := selection.NewFactory(modelSelectionCfg).
		WithModelConfig(cfg.BackendModels.ModelConfig).
		WithEmbeddingFunc(func(text string) ([]float32, error) {
			lower := strings.ToLower(text)
			switch {
			case strings.Contains(lower, "coding"):
				return []float32{1, 0}, nil
			case strings.Contains(lower, "general"):
				return []float32{0, 1}, nil
			default:
				return []float32{0.5, 0.5}, nil
			}
		}).
		CreateAll()

	router := &OpenAIRouter{
		Config:        &cfg,
		ModelSelector: registry,
	}

	selector := router.selectorForDecisionMethod(selection.MethodHybrid, &config.AlgorithmConfig{
		Type: "hybrid",
		Hybrid: &config.HybridSelectionConfig{
			ExperienceWeight: 0.6,
			RouterDCWeight:   0.4,
		},
	})

	result, err := selector.Select(context.Background(), &selection.SelectionContext{
		Query:           "need help with coding",
		DecisionName:    "hybrid_route",
		CandidateModels: []config.ModelRef{{Model: "current"}, {Model: "frontier"}},
	})
	if err != nil {
		t.Fatalf("Select returned error: %v", err)
	}
	wantWeights := fmt.Sprintf("weights=[elo:%.2f, dc:%.2f, am:%.2f, cost:%.2f]", 0.6, 0.4, 0.2, 0.2)
	if !strings.Contains(result.Reasoning, wantWeights) {
		t.Fatalf("expected decision-scoped hybrid weights in reasoning, got %q", result.Reasoning)
	}
}

func TestBuildSelectionContextUsesPinnedSessionIDAndToolLoopFacts(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{
			ModelConfig: map[string]config.ModelParams{
				"model-a": {ContextWindowSize: 8192},
			},
		},
	}}
	reqCtx := &RequestContext{
		SessionID:            "pinned-session",
		PreviousModel:        "model-a",
		TurnIndex:            2,
		HistoryTokenCount:    1024,
		VSRContextTokenCount: 2048,
		SessionIdleSeconds:   12,
		SessionIdleKnown:     true,
		VSRConversationFacts: classification.ConversationFacts{
			AssistantToolCallCount: 1,
			ToolResultCount:        1,
			LastMessageRole:        "tool",
			LastMessageToolResult:  true,
		},
	}

	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "model-a"}},
		"agentic",
		"query",
		nil,
		"",
		nil,
		reqCtx,
	)

	if selCtx.SessionID != "pinned-session" {
		t.Fatalf("expected pinned session ID, got %q", selCtx.SessionID)
	}
	if selCtx.AgenticSession == nil || !selCtx.AgenticSession.ActiveToolLoop {
		t.Fatalf("expected active tool loop in agentic session context: %#v", selCtx.AgenticSession)
	}
	if got := selCtx.AgenticSession.ModelContextWindows["model-a"]; got != 8192 {
		t.Fatalf("expected model context window 8192, got %d", got)
	}
}

func TestBuildSelectionContextMarksUserAfterToolResultAsToolLoop(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	reqCtx := &RequestContext{
		SessionID:     "tool-continuation-session",
		PreviousModel: "model-a",
		VSRConversationFacts: classification.ConversationFacts{
			AssistantToolCallCount:  1,
			ToolResultCount:         1,
			LastMessageRole:         "user",
			LastUserAfterToolResult: true,
		},
	}

	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		"agentic",
		"continue after tool output",
		nil,
		"",
		nil,
		reqCtx,
	)

	if selCtx.AgenticSession == nil || !selCtx.AgenticSession.ActiveToolLoop {
		t.Fatalf("expected user-after-tool continuation to be an active tool loop: %#v", selCtx.AgenticSession)
	}
	if selCtx.AgenticSession.Phase != selection.AgenticPhaseToolLoop {
		t.Fatalf("expected tool-loop phase, got %q", selCtx.AgenticSession.Phase)
	}
}

func TestBuildSelectionContextMarksPreviousResponseIDAsNonPortableContext(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	reqCtx := &RequestContext{
		SessionID:          "response-api-session",
		PreviousModel:      "model-a",
		PreviousResponseID: "resp_123",
	}

	selCtx := router.buildSelectionContext(
		[]config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		"agentic",
		"continue response",
		nil,
		"",
		nil,
		reqCtx,
	)

	if selCtx.AgenticSession == nil || !selCtx.AgenticSession.HasNonPortableContext {
		t.Fatalf("expected previous_response_id to mark non-portable context: %#v", selCtx.AgenticSession)
	}
	if got := selCtx.AgenticSession.NonPortableContextReason; got != "previous_response_id" {
		t.Fatalf("expected previous_response_id reason, got %q", got)
	}
	if got := selCtx.AgenticSession.Phase; got != selection.AgenticPhaseProviderState {
		t.Fatalf("expected provider-state phase for previous_response_id, got %q", got)
	}
}
