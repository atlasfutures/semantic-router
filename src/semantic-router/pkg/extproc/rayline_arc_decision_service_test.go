package extproc

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

const decisionServiceEpisodeHeader = "x-rayline-episode-id"

// stubARCSelector stands in for the artifact-backed selector. It records every
// selection context it is handed, so a test can assert what the decision-only
// path did and did not put into the routing inputs.
type stubARCSelector struct {
	contexts []*selection.SelectionContext
	arm      int
	workers  []raylinearc.WorkerManifest
	err      error
}

func (s *stubARCSelector) Select(
	_ context.Context,
	selCtx *selection.SelectionContext,
) (*selection.SelectionResult, error) {
	s.contexts = append(s.contexts, selCtx)
	if s.err != nil {
		return nil, s.err
	}
	return &selection.SelectionResult{
		SelectedModel: selCtx.CandidateModels[s.arm].Model,
		Method:        selection.MethodRaylineARC,
		Tier:          selection.TierSupported,
		RaylineARC: &selection.RaylineARCTrace{
			SelectedArm:      s.arm,
			SerializedTokens: 11,
		},
	}, nil
}

func (s *stubARCSelector) Method() selection.SelectionMethod { return selection.MethodRaylineARC }
func (s *stubARCSelector) Tier() selection.AlgorithmTier     { return selection.TierSupported }
func (s *stubARCSelector) ExternalDependencies() []selection.Dependency {
	return nil
}

func (s *stubARCSelector) UpdateFeedback(context.Context, *selection.Feedback) error {
	return nil
}

func (s *stubARCSelector) Worker(index int) (raylinearc.WorkerManifest, bool) {
	if index < 0 || index >= len(s.workers) {
		return raylinearc.WorkerManifest{}, false
	}
	return s.workers[index], true
}

func decisionServiceWorkers() []raylinearc.WorkerManifest {
	return []raylinearc.WorkerManifest{
		{ID: "worker-a", Model: "vendor/model-a", OpenRouterProviderSlug: "vendor-a"},
		{ID: "worker-b", Model: "vendor/model-b"},
	}
}

func decisionServiceConfig() *config.RouterConfig {
	return &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{
				{
					Name: "arc-decision",
					ModelRefs: []config.ModelRef{
						{Model: "worker-a"},
						{Model: "worker-b"},
					},
					Algorithm: &config.AlgorithmConfig{
						Type:    config.RaylineARCAlgorithmType,
						OnError: "fail_closed",
						RaylineARC: &config.RaylineARCAlgorithmConfig{
							Episode: config.RaylineARCEpisodeConfig{
								IDHeader:              decisionServiceEpisodeHeader,
								AcquireTimeoutSeconds: 5,
								LeaseTTLSeconds:       30,
							},
						},
					},
				},
			},
		},
	}
}

func decisionServiceRouter(t *testing.T, selector *stubARCSelector) *OpenAIRouter {
	t.Helper()
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{
		MaxEpisodes: 8,
		IdleTTL:     time.Minute,
	})
	if err != nil {
		t.Fatalf("NewMemoryEpisodeStore() error = %v", err)
	}
	registry := selection.NewRegistry()
	registry.Register(selection.MethodRaylineARC, selector)
	return &OpenAIRouter{
		Config:                 decisionServiceConfig(),
		ModelSelector:          registry,
		RaylineARCEpisodeStore: store,
	}
}

func anthropicBody(text string) []byte {
	return []byte(`{"model":"claude","messages":[{"role":"user","content":"` + text + `"}]}`)
}

func TestRouteDecisionResolvesWorkerManifestFacts(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)

	decision, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{
			Body:       anthropicBody("hello"),
			DecisionID: "rt_0a1b",
			SessionID:  "session-1",
		},
	)
	if err != nil {
		t.Fatalf("RouteDecision() error = %v, want nil", err)
	}
	if decision.SelectedWorker != "worker-a" {
		t.Fatalf("SelectedWorker = %q, want worker-a", decision.SelectedWorker)
	}
	if decision.WorkerModel != "vendor/model-a" {
		t.Fatalf("WorkerModel = %q, want vendor/model-a", decision.WorkerModel)
	}
	if decision.Provider != "vendor-a" {
		t.Fatalf("Provider = %q, want vendor-a", decision.Provider)
	}
}

// A worker with no declared provider slug must produce an empty Provider, so
// the adapter can omit the field instead of publishing an invented one.
func TestRouteDecisionLeavesProviderEmptyWhenUndeclared(t *testing.T) {
	selector := &stubARCSelector{arm: 1, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)

	decision, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-1"},
	)
	if err != nil {
		t.Fatalf("RouteDecision() error = %v, want nil", err)
	}
	if decision.Provider != "" {
		t.Fatalf("Provider = %q, want empty when the manifest declares none", decision.Provider)
	}
}

func TestRouteDecisionKeysEpisodesOnTheCallerSession(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)

	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-1"},
	); err != nil {
		t.Fatalf("RouteDecision() error = %v, want nil", err)
	}

	got := selector.contexts[0].RaylineARC.EpisodeIDHash
	if want := raylinearc.HashEpisodeID("session-1"); got != want {
		t.Fatalf("EpisodeIDHash = %q, want the hash of the caller session %q", got, want)
	}
}

// Without a session the consult still routes, on a fresh episode of its own.
// Reusing one shared episode would silently braid unrelated callers together.
func TestRouteDecisionRoutesUnsessionedConsultsOnFreshEpisodes(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	service := router.routeDecisionRuntimeState()

	for _, decisionID := range []string{"rt_0a1b", "rt_0c2d"} {
		if _, err := service.RouteDecision(
			context.Background(),
			routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), DecisionID: decisionID},
		); err != nil {
			t.Fatalf("RouteDecision() error = %v, want nil", err)
		}
	}

	first := selector.contexts[0].RaylineARC
	second := selector.contexts[1].RaylineARC
	if first.EpisodeIDHash == second.EpisodeIDHash {
		t.Fatalf("unsessioned consults shared episode %q", first.EpisodeIDHash)
	}
	if second.State.PreviousArm != nil {
		t.Fatalf("fresh episode carried PreviousArm = %v, want nil", *second.State.PreviousArm)
	}
}

// A decision-only consult has no dispatch phase, so the episode must advance at
// decision time. Otherwise the next turn on the same session would score
// against a stale trajectory and the lease would leak.
func TestRouteDecisionCommitsTheEpisodeAtDecisionTime(t *testing.T) {
	selector := &stubARCSelector{arm: 1, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	service := router.routeDecisionRuntimeState()

	for range 2 {
		if _, err := service.RouteDecision(
			context.Background(),
			routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-1"},
		); err != nil {
			t.Fatalf("RouteDecision() error = %v, want nil", err)
		}
	}

	state := selector.contexts[1].RaylineARC.State
	if state.TurnIndex != 1 {
		t.Fatalf("TurnIndex = %d on the second consult, want 1", state.TurnIndex)
	}
	if state.PreviousArm == nil || *state.PreviousArm != 1 {
		t.Fatalf("PreviousArm = %v, want the arm the first consult chose", state.PreviousArm)
	}
}

// Negative control for the record-only rule. The executed model names the arm
// the caller did NOT get; if it reached episode state, the second consult would
// see a PreviousArm derived from it instead of from this router's own choice.
func TestRouteDecisionKeepsExecutedModelOutOfEpisodeState(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	service := router.routeDecisionRuntimeState()

	for range 2 {
		if _, err := service.RouteDecision(
			context.Background(),
			routerruntime.RouteDecisionRequest{
				Body:          anthropicBody("hello"),
				SessionID:     "session-1",
				ExecutedModel: "worker-b",
			},
		); err != nil {
			t.Fatalf("RouteDecision() error = %v, want nil", err)
		}
	}

	state := selector.contexts[1].RaylineARC.State
	if state.PreviousArm == nil || *state.PreviousArm != 0 {
		t.Fatalf("PreviousArm = %v, want arm 0 (this router's choice), not the executed model's arm",
			state.PreviousArm)
	}
	got := selector.contexts[1].RaylineARC.EpisodeIDHash
	if want := raylinearc.HashEpisodeID("session-1"); got != want {
		t.Fatalf("EpisodeIDHash = %q, want %q: the executed model must not shift episode identity either",
			got, want)
	}
}

// The state selection reads is built here, and this is the only place the
// executed model could leak into it. Assert it does not appear anywhere in
// that state, so the record-only rule holds structurally rather than by habit.
func TestDecisionOnlyRequestContextExcludesExecutedModel(t *testing.T) {
	algorithm := decisionServiceConfig().Decisions[0].Algorithm
	const executed = "executed-model-marker"

	requestContext := decisionOnlyRequestContext(
		context.Background(),
		algorithm,
		routerruntime.RouteDecisionRequest{
			Body:          anthropicBody("hello"),
			DecisionID:    "rt_0a1b",
			SessionID:     "session-1",
			ExecutedModel: executed,
		},
	)

	for name, value := range map[string]string{
		"episode header":  requestContext.Headers[decisionServiceEpisodeHeader],
		"request id":      requestContext.RequestID,
		"request body":    string(requestContext.OriginalRequestBody),
		"client protocol": requestContext.ClientProtocol,
	} {
		if strings.Contains(value, executed) {
			t.Fatalf("%s carries the executed model: %q", name, value)
		}
	}
	if len(requestContext.Headers) != 1 {
		t.Fatalf("Headers = %v, want only the configured episode header", requestContext.Headers)
	}
}

func TestRouteDecisionFailsClosedWhenSelectionFails(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers(), err: errors.New("encoder down")}
	router := decisionServiceRouter(t, selector)

	decision, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-1"},
	)
	if err == nil {
		t.Fatalf("RouteDecision() = %+v, want an error rather than a fallback worker", decision)
	}
	if decision.SelectedWorker != "" {
		t.Fatalf("failed consult still named worker %q", decision.SelectedWorker)
	}

	// The lease must be released, not left pending: a later consult on the
	// same session has to be able to acquire the episode.
	selector.err = nil
	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-1"},
	); err != nil {
		t.Fatalf("consult after a failed one error = %v, want the lease released", err)
	}
	state := selector.contexts[1].RaylineARC.State
	if state.TurnIndex != 0 {
		t.Fatalf("TurnIndex = %d, want 0: a failed consult must not advance the episode", state.TurnIndex)
	}
}

func TestRouteDecisionFailsClosedWithoutAConfiguredDecision(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	router.Config = &config.RouterConfig{}

	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello")},
	); err == nil {
		t.Fatal("RouteDecision() error = nil, want a failure when no decision-only algorithm is configured")
	}
}

// A decision can name the algorithm without carrying its configuration. Config
// validation normally catches that, but the management listener answers over
// the network and must not turn a bad config into a crashed router.
func TestRouteDecisionFailsClosedOnAlgorithmWithoutConfiguration(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	router.Config.Decisions[0].Algorithm.RaylineARC = nil

	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello")},
	); err == nil {
		t.Fatal("RouteDecision() error = nil, want a failure when the algorithm carries no configuration")
	}
}

// Two candidate decisions make the choice ambiguous. Guessing one would route
// live traffic through a policy nobody selected.
func TestRouteDecisionFailsClosedOnAmbiguousDecisions(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)
	second := router.Config.Decisions[0]
	second.Name = "arc-decision-2"
	router.Config.Decisions = append(router.Config.Decisions, second)

	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello")},
	); err == nil {
		t.Fatal("RouteDecision() error = nil, want a failure when two decisions could serve the consult")
	}
}

// The transport layer accepts this body: messages is a non-empty list of
// objects. The algorithm still cannot read it, and must fail closed rather
// than route on an empty conversation.
func TestRouteDecisionRejectsUnnormalizableBodies(t *testing.T) {
	selector := &stubARCSelector{arm: 0, workers: decisionServiceWorkers()}
	router := decisionServiceRouter(t, selector)

	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{
			Body:      []byte(`{"messages":[{"role":"user","content":7}]}`),
			SessionID: "session-1",
		},
	); err == nil {
		t.Fatal("RouteDecision() error = nil, want a failure on a body the algorithm cannot normalize")
	}
	if len(selector.contexts) != 0 {
		t.Fatalf("selector ran on an unnormalizable body: %+v", selector.contexts)
	}
}

// An admission shed is deliberate back-pressure, not breakage: the gate's own
// documentation calls it "a deliberate capacity decision, never an encoder
// error". It must classify as contended so the adapter answers 429. It escaped
// the contended classification because it surfaces as a selection failure
// rather than an episode-preparation one, and a 503 here sends callers into
// fallback during exactly the bursts the gate exists to absorb.
func TestRouteDecisionClassifiesAdmissionShedAsContended(t *testing.T) {
	selector := &stubARCSelector{
		arm:     0,
		workers: decisionServiceWorkers(),
		// Exactly the error the real selector returns when the gate sheds,
		// built by the same function, so the classifier and the producer
		// cannot drift apart without this test noticing.
		err: boundedARCEncoderFailure(&raylinearc.EncoderFailure{
			Class: raylinearc.EncoderFailureAdmission,
		}),
	}
	router := decisionServiceRouter(t, selector)

	_, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-shed"},
	)
	if err == nil {
		t.Fatal("RouteDecision() succeeded, want an admission-shed error")
	}
	if !errors.Is(err, routerruntime.ErrRouteDecisionContended) {
		t.Fatalf("shed error = %v, want it to classify as contended (429), not 503", err)
	}

	// The shed must not strand the lease: the whole point of answering 429 is
	// that the caller can come back, so coming back has to work.
	selector.err = nil
	if _, err := router.routeDecisionRuntimeState().RouteDecision(
		context.Background(),
		routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-shed"},
	); err != nil {
		t.Fatalf("consult after a shed error = %v, want the lease released", err)
	}
}

// Only the shed converts. Every other encoder failure class means waiting will
// not help, so it must keep reading as 503 rather than inviting a retry storm
// against a broken encoder.
func TestRouteDecisionKeepsEncoderBreakageUncontended(t *testing.T) {
	for name, cause := range map[string]error{
		"transport": &raylinearc.EncoderFailure{Class: raylinearc.EncoderFailureTransport},
		"timeout":   &raylinearc.EncoderFailure{Class: raylinearc.EncoderFailureTimeout},
		"unclassed": errors.New("encoder fell over"),
	} {
		t.Run(name, func(t *testing.T) {
			selector := &stubARCSelector{
				arm:     0,
				workers: decisionServiceWorkers(),
				err:     boundedARCEncoderFailure(cause),
			}
			router := decisionServiceRouter(t, selector)

			_, err := router.routeDecisionRuntimeState().RouteDecision(
				context.Background(),
				routerruntime.RouteDecisionRequest{Body: anthropicBody("hello"), SessionID: "session-broken"},
			)
			if err == nil {
				t.Fatal("RouteDecision() succeeded, want an encoder failure")
			}
			if errors.Is(err, routerruntime.ErrRouteDecisionContended) {
				t.Fatalf("encoder breakage %v classified as contended; a 429 would invite retries against a broken encoder", err)
			}
		})
	}
}
