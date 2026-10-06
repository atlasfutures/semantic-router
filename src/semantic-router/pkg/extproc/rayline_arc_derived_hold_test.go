//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A history whose assistant message the episode never attributed: a
// conversation resumed after its episode cooled, or begun elsewhere.
const resumedHistory = `{"model":"auto","max_tokens":1024,"messages":[` +
	`{"role":"user","content":"fix the failing test"},` +
	`{"role":"assistant","content":"Looking at it."},` +
	`{"role":"user","content":"go on"}]}`

// v5ActionModels are the v5 fixture's actions by trained model.
var v5ActionModels = map[string]string{
	v5OpusAction: "claude-opus-5", v5GLMNone: "glm-5.3-flash", v5GLMUp: "glm-5.3-flash",
}

// refuseColdTwoStage is a two-stage package: it refuses a turn whose history
// holds an unattributed assistant message unless every offered action
// belongs to one model (pathfinder arc_serving_contract.md, "Stage one").
func refuseColdTwoStage(request raylinearc.PolicyDecisionRequest) bool {
	models := map[string]bool{}
	for _, action := range request.Selection.AvailableActionIDs {
		models[v5ActionModels[action]] = true
	}
	var history struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	_ = json.Unmarshal(request.Request.Messages, &history.Messages)
	assistant := slices.ContainsFunc(history.Messages, func(m struct {
		Role string `json:"role"`
	},
	) bool {
		return m.Role == "assistant"
	})
	return request.Selection.HeldActionID == nil && len(models) > 1 && assistant && len(request.Attribution) == 0
}

// heldTurnStatus runs one Messages body through the request phase and
// returns the provider-bound model, or the refusal's status.
func heldTurnStatus(t *testing.T, router *OpenAIRouter, episode, client string, extra ...*core.HeaderValue) (model string, status int) {
	t.Helper()
	ctx := &RequestContext{
		Headers: map[string]string{}, RequestID: fmt.Sprintf("hold-%s-%d", episode, time.Now().UnixNano()),
		StartTime: time.Now(), TraceContext: context.Background(),
	}
	headers := &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
			{Key: ":method", Value: "POST"},
			{Key: ":path", Value: "/v1/messages"},
			{Key: "content-type", Value: "application/json"},
			{Key: "x-rayline-session", Value: episode},
		}},
	}}
	headers.RequestHeaders.Headers.Headers = append(headers.RequestHeaders.Headers.Headers, extra...)
	if response, err := router.handleRequestHeaders(headers, ctx); err != nil || response.GetImmediateResponse() != nil {
		t.Fatalf("request headers: err=%v immediate=%v", err, response.GetImmediateResponse())
	}
	response, err := router.handleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(client), EndOfStream: true},
	}, ctx)
	if err != nil {
		t.Fatalf("request body: %v", err)
	}
	if immediate := response.GetImmediateResponse(); immediate != nil {
		return "", int(immediate.GetStatus().GetCode())
	}
	var body struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatal(err)
	}
	return body.Model, 200
}

func servedStore(t *testing.T, router *OpenAIRouter) raylinearc.ServedWorkerStore {
	t.Helper()
	store, ok := router.raylineARCEpisodeStoreFor(&RequestContext{}).(raylinearc.ServedWorkerStore)
	if !ok {
		t.Fatal("the episode store keeps no served-worker records")
	}
	return store
}

// chooseFirstOffered picks the first offered action.
func chooseFirstOffered(request raylinearc.PolicyDecisionRequest) string {
	return request.Selection.AvailableActionIDs[0]
}

// A committed turn records its worker past the episode, and a resumed
// conversation whose episode is gone is held on that worker's model when a
// two-stage package refuses it cold.
func TestDerivedHoldUsesTheLastServedWorker(t *testing.T) {
	router, fake := v5Router(t, "openai")
	v5Turn(t, router, fake, v5GLMUp, "episode-served", `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	store := servedStore(t, router)
	worker, err := store.LastServedWorker(context.Background(), raylinearc.HashEpisodeID("episode-served"))
	if err != nil || worker != "glm" {
		t.Fatalf("recorded worker = %q, %v; want glm", worker, err)
	}

	// The record outlives the episode: a cooled episode has none of its own
	// state, only this.
	if err := store.RecordServedWorker(context.Background(), raylinearc.HashEpisodeID("episode-cooled"), "glm", time.Now()); err != nil {
		t.Fatal(err)
	}
	fake.refuseCold = refuseColdTwoStage
	fake.chooseWith(chooseFirstOffered)
	before := len(fake.received())
	model, status := heldTurnStatus(t, router, "episode-cooled", resumedHistory)
	if status != 200 || model != "z-ai/glm-5.3-flash" {
		t.Fatalf("dispatched %q (status %d), want the held GLM", model, status)
	}
	calls := fake.received()[before:]
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{v5GLMNone, v5GLMUp}) {
		t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
	}
}

// With no record, the hold is the package's declared fallback action's model.
func TestDerivedHoldFallsBackToThePackageFallback(t *testing.T) {
	router, fake := v5Router(t, "openai")
	fake.refuseCold = refuseColdTwoStage
	fake.chooseWith(chooseFirstOffered)
	model, status := heldTurnStatus(t, router, "episode-unknown", resumedHistory)
	if status != 200 || model != "anthropic/claude-opus-5" {
		t.Fatalf("dispatched %q (status %d), want the package fallback's Opus", model, status)
	}
	calls := fake.received()
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{v5OpusAction}) {
		t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
	}
}

// A single-stage package decides the same turn freely: one call, the whole
// offer, nothing narrowed -- even with a served-worker record present.
func TestDerivedHoldLeavesSingleStagePackagesAlone(t *testing.T) {
	router, fake := v5Router(t, "openai")
	if err := servedStore(t, router).RecordServedWorker(context.Background(), raylinearc.HashEpisodeID("episode-single"), "glm", time.Now()); err != nil {
		t.Fatal(err)
	}
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return v5OpusAction })
	model, status := heldTurnStatus(t, router, "episode-single", resumedHistory)
	if status != 200 || model != "anthropic/claude-opus-5" {
		t.Fatalf("dispatched %q (status %d)", model, status)
	}
	calls := fake.received()
	if len(calls) != 1 || len(calls[0].Selection.AvailableActionIDs) != 3 {
		t.Fatalf("decide calls = %d, offer %v", len(calls), calls[0].Selection.AvailableActionIDs)
	}
}

// A refusal of the held retry fails the turn as before: one retry, then the
// selection failure's 503.
func TestDerivedHoldRetriesOnce(t *testing.T) {
	router, fake := v5Router(t, "openai")
	fake.refuseCold = func(raylinearc.PolicyDecisionRequest) bool { return true }
	fake.chooseWith(chooseFirstOffered)
	_, status := heldTurnStatus(t, router, "episode-refused", resumedHistory)
	if status != 503 {
		t.Fatalf("status %d, want 503", status)
	}
	if calls := fake.received(); len(calls) != 2 {
		t.Fatalf("decide calls = %d, want the first and one retry", len(calls))
	}
}

// A subagent handed its parent's history, with the gateway's turn-signal
// headers trusted, is held on the model that last served the parent session.
func TestDerivedHoldUsesTheParentSession(t *testing.T) {
	router, fake := v5RouterWith(t, "openai", func(config string) string {
		return strings.Replace(config, "            allow_experimental_controls: true\n",
			"            allow_experimental_controls: true\n            trust_turn_signal_headers: true\n", 1)
	})
	if err := servedStore(t, router).RecordServedWorker(context.Background(), raylinearc.HashEpisodeID("episode-parent"), "glm", time.Now()); err != nil {
		t.Fatal(err)
	}
	fake.refuseCold = refuseColdTwoStage
	fake.chooseWith(chooseFirstOffered)
	model, status := heldTurnStatus(t, router, "episode-subagent", resumedHistory,
		&core.HeaderValue{Key: "x-rayline-parent-session", Value: "episode-parent"},
		&core.HeaderValue{Key: "x-rayline-agent-key-source", Value: "agent"})
	if status != 200 || model != "z-ai/glm-5.3-flash" {
		t.Fatalf("dispatched %q (status %d), want the parent's GLM", model, status)
	}
	calls := fake.received()
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{v5GLMNone, v5GLMUp}) {
		t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
	}
}

// The derived model is held whatever the turn excludes, as the contract holds
// a model: here an image turn, whose vision gate takes the text-only GLM out
// of the offer, is derived onto GLM. Nothing of it is offered, so the turn
// fails as an empty offer does -- policy_no_available_action -- without a
// second, empty, decide.
func TestDerivedHoldOnAnExcludedModelFailsWithoutRetrying(t *testing.T) {
	router, fake := v5RouterWith(t, "openai", func(config string) string {
		return strings.Replace(config, "    - name: glm\n      modality: text\n",
			"    - name: glm\n      modality: text\n      vision: false\n", 1)
	})
	if err := servedStore(t, router).RecordServedWorker(context.Background(), raylinearc.HashEpisodeID("episode-image"), "glm", time.Now()); err != nil {
		t.Fatal(err)
	}
	// The service refuses even Opus alone, so the derivation runs.
	fake.refuseCold = func(raylinearc.PolicyDecisionRequest) bool { return true }
	fake.chooseWith(chooseFirstOffered)
	image := `{"model":"auto","max_tokens":1024,"messages":[` +
		`{"role":"user","content":"look at this"},{"role":"assistant","content":"Sure."},` +
		`{"role":"user","content":[{"type":"text","text":"what is it?"},` +
		`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + userImagePNG + `"}}]}]}`
	failures := func() float64 {
		return testutil.ToFloat64(metrics.RaylineARCSelectionFailures.WithLabelValues("policy_no_available_action"))
	}
	before := failures()
	_, status := heldTurnStatus(t, router, "episode-image", image)
	if status != 503 {
		t.Fatalf("status %d, want 503", status)
	}
	calls := fake.received()
	if len(calls) != 1 || slices.Contains(calls[0].Selection.AvailableActionIDs, v5GLMNone) {
		t.Fatalf("decide calls = %d (offer %v), want one, without GLM", len(calls), calls[0].Selection.AvailableActionIDs)
	}
	if failures()-before != 1 {
		t.Fatal("the turn did not fail as policy_no_available_action")
	}
}

// Two turns' served-worker writes landing out of order leave the record on
// the later turn's worker: each is stamped before its episode commit, while
// the lease still orders the turns.
func TestServedWorkerRecordsLandingOutOfOrder(t *testing.T) {
	store, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	episode := raylinearc.HashEpisodeID(t.Name())
	turn := func(worker string) *raylineARCEpisodeTransaction {
		lease, state, err := store.Prepare(context.Background(), episode, 2)
		if err != nil {
			t.Fatal(err)
		}
		transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
		transaction.markSelection(0, 10)
		transaction.markServedWorker(worker)
		if err := transaction.commit(context.Background(), &RequestContext{}); err != nil {
			t.Fatal(err)
		}
		return transaction
	}
	first := turn("glm")
	turn("opus")
	// The first turn's write arrives again, after the second's.
	first.recordServedWorker(context.Background())
	if worker, err := store.LastServedWorker(context.Background(), episode); err != nil || worker != "opus" {
		t.Fatalf("record = %q, %v; want the later turn's opus", worker, err)
	}
}
