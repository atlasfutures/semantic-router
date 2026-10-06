//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
	model, status, _ = heldTurn(t, router, episode, client, extra...)
	return model, status
}

// heldTurn is heldTurnStatus with the request's context, for a caller that
// goes on to commit the turn.
func heldTurn(t *testing.T, router *OpenAIRouter, episode, client string, extra ...*core.HeaderValue) (model string, status int, ctx *RequestContext) {
	t.Helper()
	ctx = &RequestContext{
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
		return "", int(immediate.GetStatus().GetCode()), ctx
	}
	var body struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatal(err)
	}
	return body.Model, 200, ctx
}

func servedStore(t *testing.T, router *OpenAIRouter) raylinearc.ServedModelStore {
	t.Helper()
	store, ok := router.raylineARCEpisodeStoreFor(&RequestContext{}).(raylinearc.ServedModelStore)
	if !ok {
		t.Fatal("the episode store keeps no served-worker records")
	}
	return store
}

// chooseFirstOffered picks the first offered action.
func chooseFirstOffered(request raylinearc.PolicyDecisionRequest) string {
	return request.Selection.AvailableActionIDs[0]
}

// A committed turn records its action's model past the episode, and a
// resumed conversation whose episode is gone is held on that model when a
// two-stage package refuses it cold.
func TestDerivedHoldUsesTheLastServedModel(t *testing.T) {
	router, fake := v5Router(t, "openai")
	v5Turn(t, router, fake, v5GLMUp, "episode-served", `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	store := servedStore(t, router)
	recorded, err := store.LastServedModel(context.Background(), raylinearc.HashEpisodeID("episode-served"))
	if err != nil || recorded != "glm-5.3-flash" {
		t.Fatalf("recorded %q, %v; want the action's model glm-5.3-flash", recorded, err)
	}

	// The record outlives the episode: a cooled episode has none of its own
	// state, only this.
	for name, record := range map[string]string{
		"model record":                   recorded,
		"worker record from before this": "glm",
	} {
		t.Run(name, func(t *testing.T) {
			episode := "episode-cooled-" + strings.ReplaceAll(name, " ", "-")
			seedServed(t, router, episode, record)
			fake.refuseCold = refuseColdTwoStage
			fake.chooseWith(chooseFirstOffered)
			before := len(fake.received())
			model, status := heldTurnStatus(t, router, episode, resumedHistory)
			if status != 200 || model != "z-ai/glm-5.3-flash" {
				t.Fatalf("dispatched %q (status %d), want the held GLM", model, status)
			}
			calls := fake.received()[before:]
			if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{v5GLMNone, v5GLMUp}) {
				t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
			}
		})
	}
}

// A manifest-less package may bind two trained models to one worker. The
// record names the model that served, not the worker, so a cold resume holds
// that model -- not the worker's first bound one.
func TestDerivedHoldHoldsTheServedModelOfAWorkerWithTwo(t *testing.T) {
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	actions["off-b"] = policyAction("off", "none", "off-b-trained", policyTestEffort("none"), nil, "")
	catalog := make([]string, 0, len(actions))
	models := map[string]string{}
	for _, action := range actions {
		catalog = append(catalog, action.ActionID)
		models[action.ActionID] = action.Model
	}
	fake := newFakePolicyService(t, policyTestAlias, policyTestPackage, catalog)
	router, err := NewOpenAIRouter(writePolicyDispatchConfig(t, fake.URL(), actions))
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	second := actions["off-b"].ActionID

	// The served turn chooses the worker's second model.
	fake.chooseWith(func(raylinearc.PolicyDecisionRequest) string { return second })
	model, status, ctx := heldTurn(t, router, "episode-two-models", `{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	if status != 200 || model != "vendor/off" {
		t.Fatalf("served %q (status %d)", model, status)
	}
	if _, commitErr := router.handleResponseHeaders(arcResponseHeaders("200"), ctx); commitErr != nil {
		t.Fatalf("commit: %v", commitErr)
	}
	completeTestResponse(t, ctx)
	finalizeSelectionProcessTerminal(ctx)
	recorded, err := servedStore(t, router).LastServedModel(context.Background(), raylinearc.HashEpisodeID("episode-two-models"))
	if err != nil || recorded != "off-b-trained" {
		t.Fatalf("recorded %q, %v; want the served off-b-trained", recorded, err)
	}

	seedServed(t, router, "episode-two-models-cold", recorded)
	fake.refuseCold = func(request raylinearc.PolicyDecisionRequest) bool {
		offered := map[string]bool{}
		for _, action := range request.Selection.AvailableActionIDs {
			offered[models[action]] = true
		}
		return len(offered) > 1 && len(request.Attribution) == 0
	}
	fake.chooseWith(chooseFirstOffered)
	before := len(fake.received())
	if _, status := heldTurnStatus(t, router, "episode-two-models-cold", resumedHistory); status != 200 {
		t.Fatalf("status %d", status)
	}
	calls := fake.received()[before:]
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{second}) {
		t.Fatalf("decide calls = %d, retry offer %v; want only the served model's action", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
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
	seedServed(t, router, "episode-single", "glm")
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
	seedServed(t, router, "episode-parent", "glm")
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
	seedServed(t, router, "episode-image", "glm")
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

// seedServed records a served worker for an episode with no episode state,
// as a cooled episode leaves it.
func seedServed(t *testing.T, router *OpenAIRouter, episode, worker string) {
	t.Helper()
	store := router.raylineARCEpisodeStoreFor(&RequestContext{})
	if unready, ok := store.(unreadyRaylineARCEpisodeStore); ok {
		store = unready.EpisodeStore
	}
	memory, ok := store.(*raylinearc.MemoryEpisodeStore)
	if !ok {
		t.Fatalf("episode store %T is not the memory store", store)
	}
	if err := memory.SeedServedModel(raylinearc.HashEpisodeID(episode), worker); err != nil {
		t.Fatal(err)
	}
}

// countingEpisodeStore counts the calls a commit makes on its store.
type countingEpisodeStore struct {
	*raylinearc.MemoryEpisodeStore
	calls atomic.Int32
}

func (store *countingEpisodeStore) Commit(ctx context.Context, lease raylinearc.Lease, version uint64, state *raylinearc.EpisodeState) error {
	store.calls.Add(1)
	return store.MemoryEpisodeStore.Commit(ctx, lease, version, state)
}

func (store *countingEpisodeStore) LastServedModel(ctx context.Context, episodeIDHash string) (string, error) {
	store.calls.Add(1)
	return store.MemoryEpisodeStore.LastServedModel(ctx, episodeIDHash)
}

// A turn's commit records its served worker in the commit itself: one store
// call, after which the record is there. Nothing else is written on the
// response path, so a record cannot hold a commit up.
func TestCommitRecordsTheServedModelInOneStoreCall(t *testing.T) {
	memory, err := raylinearc.NewMemoryEpisodeStore(raylinearc.MemoryEpisodeStoreConfig{MaxEpisodes: 4, IdleTTL: time.Minute})
	if err != nil {
		t.Fatal(err)
	}
	store := &countingEpisodeStore{MemoryEpisodeStore: memory}
	episode := raylinearc.HashEpisodeID(t.Name())
	lease, state, err := store.Prepare(context.Background(), episode, 2)
	if err != nil {
		t.Fatal(err)
	}
	transaction := newRaylineARCEpisodeTransaction(store, lease, state, episode, time.Minute, nil)
	transaction.markSelection(0, 10)
	transaction.markServedModel("glm")
	if err := transaction.commit(context.Background(), &RequestContext{}); err != nil {
		t.Fatal(err)
	}
	if calls := store.calls.Load(); calls != 1 {
		t.Fatalf("the commit made %d store calls, want one", calls)
	}
	if worker, _ := memory.LastServedModel(context.Background(), episode); worker != "glm" {
		t.Fatalf("record %q after the commit", worker)
	}
}

// stalledServedStore never answers a served-worker read until its context
// ends.
type stalledServedStore struct{}

func (stalledServedStore) LastServedModel(ctx context.Context, _ string) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

// A stalled store costs a turn's served-worker read at most
// servedModelReadTimeout, after which the derivation falls to the package.
func TestServedModelReadIsBounded(t *testing.T) {
	keys := []servedModelKey{
		{source: derivedHoldSessionRecord, episodeIDHash: raylinearc.HashEpisodeID("a")},
		{source: derivedHoldParentSession, episodeIDHash: raylinearc.HashEpisodeID("b")},
	}
	started := time.Now()
	served := readServedModels(context.Background(), stalledServedStore{}, keys)
	if elapsed := time.Since(started); elapsed > servedModelReadTimeout+250*time.Millisecond || len(served) != 0 {
		t.Fatalf("read took %v and returned %v", elapsed, served)
	}
}

// v4ColdRouter is the relaxed policy cell serving a v4 package (bindings
// declare their dispatch, no package_manifest), with derivedHold as its
// derived_hold_model when set, behind a two-stage service that refuses a
// cold turn whose offer spans more than one model.
func v4ColdRouter(t *testing.T, derivedHold string) (*OpenAIRouter, *fakePolicyService, map[string]config.RaylineARCPolicyBinding) {
	t.Helper()
	return v4ColdRouterListing(t, derivedHold, "", "")
}

// v4ColdRouterListing is v4ColdRouter behind a service whose packages
// listing names listedActionID and listedModel as the package's fallback pair
// (none when listedModel is empty).
func v4ColdRouterListing(t *testing.T, derivedHold, listedActionID, listedModel string) (*OpenAIRouter, *fakePolicyService, map[string]config.RaylineARCPolicyBinding) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	if listedModel != "" {
		fake.listFallback(listedActionID, listedModel)
	}
	path := writeConsistentPolicyConfig(t, fake.URL(), "strict")
	if derivedHold != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		const anchor = "            bindings:\n"
		if !strings.Contains(string(raw), anchor) {
			t.Fatal("the policy config no longer carries its bindings anchor")
		}
		edited := strings.Replace(string(raw), anchor, "            derived_hold_model: "+derivedHold+"\n"+anchor, 1)
		if err := os.WriteFile(path, []byte(edited), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	router, err := NewOpenAIRouter(path)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)
	models := map[string]string{}
	for _, action := range actions {
		models[action.ActionID] = action.Model
	}
	fake.refuseCold = func(request raylinearc.PolicyDecisionRequest) bool {
		offered := map[string]bool{}
		for _, action := range request.Selection.AvailableActionIDs {
			offered[models[action]] = true
		}
		return len(offered) > 1 && len(request.Attribution) == 0
	}
	fake.chooseWith(chooseFirstOffered)
	return router, fake, actions
}

// A v4 package has no manifest, and when the service's package listing names
// no fallback either, a cold turn with no record is held on the cell's
// configured derived_hold_model.
func TestDerivedHoldUsesTheConfiguredModelForAManifestlessPackage(t *testing.T) {
	router, fake, actions := v4ColdRouter(t, "off-trained")
	model, status := heldTurnStatus(t, router, "episode-v4-cold", resumedHistory)
	if status != 200 || model != "vendor/off" {
		t.Fatalf("dispatched %q (status %d), want the configured off-trained", model, status)
	}
	calls := fake.received()
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{actions["off"].ActionID}) {
		t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
	}
}

// Without it, a manifest-less package has no fallback: the refusal stands.
func TestDerivedHoldWithoutAFallbackLeavesTheRefusal(t *testing.T) {
	router, fake, _ := v4ColdRouter(t, "")
	if _, status := heldTurnStatus(t, router, "episode-v4-none", resumedHistory); status != 503 {
		t.Fatalf("status %d, want 503", status)
	}
	if calls := fake.received(); len(calls) != 1 {
		t.Fatalf("decide calls = %d, want one", len(calls))
	}
}

// A v4 package from the store has no manifest, but a service from
// pathfinder#3677 on names its fallback in the packages listing: with no
// derived_hold_model configured, a cold turn with no record is held on it.
func TestDerivedHoldUsesTheListedFallbackForAManifestlessPackage(t *testing.T) {
	router, fake, actions := v4ColdRouterListing(t, "", relaxedPolicyActions()["off"].ActionID, "off-trained")
	model, status := heldTurnStatus(t, router, "episode-v4-listed", resumedHistory)
	if status != 200 || model != "vendor/off" {
		t.Fatalf("dispatched %q (status %d), want the listed off-trained", model, status)
	}
	calls := fake.received()
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{actions["off"].ActionID}) {
		t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
	}
}

// The listing is the service's statement of the package it serves, so it
// ranks above the cell's configured derived_hold_model.
func TestDerivedHoldPrefersTheListedFallbackToTheConfiguredModel(t *testing.T) {
	router, fake, actions := v4ColdRouterListing(t, "off-trained", relaxedPolicyActions()["claude"].ActionID, "claude-opus-5")
	if _, status := heldTurnStatus(t, router, "episode-v4-listed-first", resumedHistory); status != 200 {
		t.Fatalf("status %d, want 200", status)
	}
	calls := fake.received()
	want := []string{actions["claude"].ActionID, actions["claude-off"].ActionID}
	got := slices.Clone(calls[len(calls)-1].Selection.AvailableActionIDs)
	slices.Sort(got)
	slices.Sort(want)
	if len(calls) != 2 || !slices.Equal(got, want) {
		t.Fatalf("decide calls = %d, retry offer %v, want claude-opus-5's %v", len(calls), got, want)
	}
}

// A listed fallback the bindings do not dispatch cannot hold a turn: it is
// skipped, and the configured derived_hold_model holds it instead.
func TestDerivedHoldSkipsAnUnboundListedFallback(t *testing.T) {
	router, fake, actions := v4ColdRouterListing(t, "off-trained", strings.Repeat("a", 64), "anthropic/claude-opus-5")
	model, status := heldTurnStatus(t, router, "episode-v4-listed-unbound", resumedHistory)
	if status != 200 || model != "vendor/off" {
		t.Fatalf("dispatched %q (status %d), want the configured off-trained", model, status)
	}
	calls := fake.received()
	if len(calls) != 2 || !slices.Equal(calls[1].Selection.AvailableActionIDs, []string{actions["off"].ActionID}) {
		t.Fatalf("decide calls = %d, retry offer %v", len(calls), calls[len(calls)-1].Selection.AvailableActionIDs)
	}
}

// The listed pair must agree with the cell: the fallback action, resolved
// through the bindings, must be the listed model. A pair whose action the
// cell binds to another model is not used, so with no derived_hold_model the
// refusal stands.
func TestDerivedHoldRefusesAMismatchedListedPair(t *testing.T) {
	router, fake, _ := v4ColdRouterListing(t, "", relaxedPolicyActions()["claude"].ActionID, "off-trained")
	if _, status := heldTurnStatus(t, router, "episode-v4-listed-mismatch", resumedHistory); status != 503 {
		t.Fatalf("status %d, want 503: a mismatched listed pair held the turn", status)
	}
	if calls := fake.received(); len(calls) != 1 {
		t.Fatalf("decide calls = %d, want one", len(calls))
	}
}

// A listed fallback action this cell does not bind is the package's own
// fallback, which the cell does not serve: the pair is skipped even when its
// model is bound.
func TestDerivedHoldSkipsAListedPairWhoseActionIsUnbound(t *testing.T) {
	router, fake, _ := v4ColdRouterListing(t, "", strings.Repeat("a", 64), "off-trained")
	if _, status := heldTurnStatus(t, router, "episode-v4-listed-unbound-action", resumedHistory); status != 503 {
		t.Fatalf("status %d, want 503: an unbound listed action held the turn", status)
	}
	if calls := fake.received(); len(calls) != 1 {
		t.Fatalf("decide calls = %d, want one", len(calls))
	}
}
