//go:build !windows && cgo

package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"go.uber.org/zap/zaptest/observer"

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
	model, ctx, immediate := heldTurnResponse(t, router, episode, client, extra...)
	if immediate != nil {
		return "", int(immediate.GetStatus().GetCode()), ctx
	}
	return model, 200, ctx
}

// heldTurnResponse runs one Messages body through the request phase and
// returns the provider-bound model, or the immediate refusal.
func heldTurnResponse(t *testing.T, router *OpenAIRouter, episode, client string, extra ...*core.HeaderValue) (model string, ctx *RequestContext, refusal *ext_proc.ImmediateResponse) {
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
		return "", ctx, immediate
	}
	var body struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(response.GetRequestBody().GetResponse().GetBodyMutation().GetBody(), &body); err != nil {
		t.Fatal(err)
	}
	return body.Model, ctx, nil
}

// immediateHeader is a header an immediate refusal carries, or "".
func immediateHeader(refusal *ext_proc.ImmediateResponse, key string) string {
	for _, header := range refusal.GetHeaders().GetSetHeaders() {
		if header.GetHeader().GetKey() == key {
			if value := header.GetHeader().GetValue(); value != "" {
				return value
			}
			return string(header.GetHeader().GetRawValue())
		}
	}
	return ""
}

// assertUndecidable checks a turn failed as undecidable: a 422 with the
// public class, the policy_undecidable selection failure, and a
// rayline_arc_undecidable line naming the policy's refusal and the record a
// hold was retried on (derivedSource, "" when none).
func assertUndecidable(t *testing.T, refusal *ext_proc.ImmediateResponse, logs *observer.ObservedLogs, failuresBefore float64, derivedSource string) {
	t.Helper()
	if refusal == nil {
		t.Fatal("the turn was served; want it refused as undecidable")
	}
	if code := refusal.GetStatus().GetCode(); int(code) != 422 {
		t.Fatalf("status %d, want 422", code)
	}
	if class := immediateHeader(refusal, "x-vsr-failure-class"); class != "undecidable" {
		t.Fatalf("x-vsr-failure-class = %q, want undecidable", class)
	}
	if undecidableFailures()-failuresBefore != 1 {
		t.Fatal("the turn did not fail as policy_undecidable")
	}
	fields := findLogEvent(t, logs, "rayline_arc_undecidable")
	if fields["reason"] != "stage_one_held_unknown" || fields["derived_source"] != derivedSource {
		t.Fatalf("rayline_arc_undecidable reason=%v derived_source=%v, want stage_one_held_unknown and %q",
			fields["reason"], fields["derived_source"], derivedSource)
	}
}

func undecidableFailures() float64 {
	return testutil.ToFloat64(metrics.RaylineARCSelectionFailures.WithLabelValues("policy_undecidable"))
}

// noLogEvent fails when event was logged.
func noLogEvent(t *testing.T, logs *observer.ObservedLogs, event string) {
	t.Helper()
	for _, entry := range logs.All() {
		if name, _ := entry.ContextMap()["event"].(string); name == event {
			t.Fatalf("%s was logged: %v", event, entry.ContextMap())
		}
	}
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
// resumed conversation whose episode is gone is held on that model when the
// package refuses it cold. This fixture's manifest declares no stage one, so
// VSR does not know it as two-stage and the record rides on the re-ask, not
// the first ask (a declared two-stage package is covered in
// rayline_arc_two_stage_schedule_test.go).
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
			if len(calls) != 2 || calls[0].Selection.AttestedHeldActionID != nil || len(calls[0].Selection.AvailableActionIDs) != 3 ||
				!slices.Equal(calls[1].Selection.AvailableActionIDs, []string{v5GLMNone, v5GLMUp}) {
				t.Fatalf("decide calls = %d, offers %v; want the plain ask refused, then one narrowed to the recorded model", len(calls), offersOf(calls))
			}
			// a package whose manifest does not declare a model-unit stage one
			// is never attested: its retry is the narrowed hold, as before
			if calls[1].Selection.AttestedHeldActionID != nil {
				t.Fatal("a package without a model-unit stage one was attested")
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

// With no record of this episode's own decisions, the first ask carries
// nothing, and a turn the two-stage policy refuses is undecidable: it fails
// 422, and the package's declared fallback action (Opus here) never serves it.
func TestDerivedHoldWithoutARecordIsUndecidable(t *testing.T) {
	logs := captureLogs(t)
	router, fake := v5Router(t, "openai")
	fake.refuseCold = refuseColdTwoStage
	fake.chooseWith(chooseFirstOffered)
	before := undecidableFailures()
	model, _, refusal := heldTurnResponse(t, router, "episode-unknown", resumedHistory)
	if model != "" {
		t.Fatalf("dispatched %q, want no model", model)
	}
	assertUndecidable(t, refusal, logs, before, "")
	noLogEvent(t, logs, "rayline_arc_derived_hold")
	if calls := fake.received(); len(calls) != 1 || calls[0].Selection.AttestedHeldActionID != nil || len(calls[0].Selection.AvailableActionIDs) != 3 {
		t.Fatalf("decide calls = %d, want the one plain refused call and no retry", len(calls))
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

// A refusal of the held retry fails the turn as undecidable: one retry on
// the recorded model, then 422.
func TestDerivedHoldRetriesOnce(t *testing.T) {
	logs := captureLogs(t)
	router, fake := v5Router(t, "openai")
	seedServed(t, router, "episode-refused", "glm")
	fake.refuseCold = func(raylinearc.PolicyDecisionRequest) bool { return true }
	fake.chooseWith(chooseFirstOffered)
	before := undecidableFailures()
	_, _, refusal := heldTurnResponse(t, router, "episode-refused", resumedHistory)
	assertUndecidable(t, refusal, logs, before, "session_record")
	if calls := fake.received(); len(calls) != 2 {
		t.Fatalf("decide calls = %d, want the first and one retry", len(calls))
	}
}

// A subagent handed its parent's history is its own episode (ADR 0117): the
// model that last served the parent never holds it, even with the gateway's
// turn-signal headers trusted. With no record of its own, it is undecidable.
func TestDerivedHoldNeverUsesTheParentSession(t *testing.T) {
	logs := captureLogs(t)
	router, fake := v5RouterWith(t, "openai", func(config string) string {
		return strings.Replace(config, "            allow_experimental_controls: true\n",
			"            allow_experimental_controls: true\n            trust_turn_signal_headers: true\n", 1)
	})
	seedServed(t, router, "episode-parent", "glm")
	fake.refuseCold = refuseColdTwoStage
	fake.chooseWith(chooseFirstOffered)
	before := undecidableFailures()
	_, _, refusal := heldTurnResponse(t, router, "episode-subagent", resumedHistory,
		&core.HeaderValue{Key: "x-rayline-parent-session", Value: "episode-parent"},
		&core.HeaderValue{Key: "x-rayline-agent-key-source", Value: "agent"})
	assertUndecidable(t, refusal, logs, before, "")
	if calls := fake.received(); len(calls) != 1 {
		t.Fatalf("decide calls = %d, want one", len(calls))
	}
}

// The derived model is held whatever the turn excludes, as the contract holds
// a model: here an image turn, whose vision gate takes the text-only GLM out
// of the offer, is derived onto GLM. Nothing of it is offered, so the turn
// fails as an empty offer does -- policy_no_available_action -- without a
// second, empty, decide.
func TestDerivedHoldOnAnExcludedModelFailsWithoutRetrying(t *testing.T) {
	logs := captureLogs(t)
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
	// An excluded model has no action to attest or narrow to: the one ask
	// goes out plain, and its refusal fails the turn.
	if calls[0].Selection.AttestedHeldActionID != nil {
		t.Fatal("the recorded model is excluded this turn but was attested")
	}
	if failures()-before != 1 {
		t.Fatal("the turn did not fail as policy_no_available_action")
	}
	fields := findLogEvent(t, logs, "rayline_arc_empty_offer")
	if fields["empty_at"] != "derived_hold" {
		t.Fatalf("empty offer empty_at = %v, want derived_hold", fields["empty_at"])
	}
	// No arm is held on this path, so the event names the derived model whose
	// exclusion emptied the offer.
	if model, _ := fields["derived_model"].(string); model == "" || fields["derived_source"] == "" {
		t.Fatalf("empty offer names no derived hold: model=%v source=%v", fields["derived_model"], fields["derived_source"])
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
// servedModelReadTimeout, after which the turn has no record.
func TestServedModelReadIsBounded(t *testing.T) {
	keys := []servedModelKey{
		{source: derivedHoldSessionRecord, episodeIDHash: raylinearc.HashEpisodeID("a")},
		{source: derivedHoldSessionRecord, episodeIDHash: raylinearc.HashEpisodeID("b")},
	}
	started := time.Now()
	served := readServedModels(context.Background(), stalledServedStore{}, keys)
	if elapsed := time.Since(started); elapsed > servedModelReadTimeout+250*time.Millisecond || len(served) != 0 {
		t.Fatalf("read took %v and returned %v", elapsed, served)
	}
}

// v4ColdRouter is the relaxed policy cell serving a v4 package (bindings
// declare their dispatch, no package_manifest) behind a two-stage service that
// refuses a cold turn whose offer spans more than one model, and whose
// packages listing names listedActionID and listedModel as the package's
// fallback pair (none when listedModel is empty).
func v4ColdRouter(t *testing.T, listedActionID, listedModel string) (*OpenAIRouter, *fakePolicyService) {
	t.Helper()
	t.Setenv("POLICY_E2E_PROVIDER_KEY", "public-e2e-provider-key")
	actions := relaxedPolicyActions()
	fake := newRelaxedPolicyFake(t)
	if listedModel != "" {
		fake.listFallback(listedActionID, listedModel)
	}
	router, err := NewOpenAIRouter(writeConsistentPolicyConfig(t, fake.URL(), "strict"))
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
	return router, fake
}

// A manifest-less package with no record is undecidable like any other: the
// refusal fails the turn 422 after the one call.
func TestDerivedHoldManifestlessWithoutARecordIsUndecidable(t *testing.T) {
	logs := captureLogs(t)
	router, fake := v4ColdRouter(t, "", "")
	before := undecidableFailures()
	_, _, refusal := heldTurnResponse(t, router, "episode-v4-none", resumedHistory)
	assertUndecidable(t, refusal, logs, before, "")
	if calls := fake.received(); len(calls) != 1 {
		t.Fatalf("decide calls = %d, want one", len(calls))
	}
}

// A service whose packages listing names a fallback pair (pathfinder#3677)
// still never holds a cold turn on it: the listing is decoded and ignored,
// so the turn is undecidable even when the cell binds the listed action.
func TestDerivedHoldIgnoresTheListedFallback(t *testing.T) {
	logs := captureLogs(t)
	router, fake := v4ColdRouter(t, relaxedPolicyActions()["off"].ActionID, "off-trained")
	before := undecidableFailures()
	_, _, refusal := heldTurnResponse(t, router, "episode-v4-listed", resumedHistory)
	assertUndecidable(t, refusal, logs, before, "")
	noLogEvent(t, logs, "rayline_arc_derived_hold")
	if calls := fake.received(); len(calls) != 1 {
		t.Fatalf("decide calls = %d, want one", len(calls))
	}
}

// A held turn the policy refuses for another reason -- here the service finding
// no supported selection, where it used to serve its package's fallback action
// -- is undecidable too: 422, and the reason is logged.
func TestSelectionRefusedIsUndecidable(t *testing.T) {
	logs := captureLogs(t)
	router, fake := v5Router(t, "openai")
	fake.failWith = "selection_refused"
	fake.failDetail = map[string]any{"reason": "no_supported_selection"}
	before := undecidableFailures()
	_, _, refusal := heldTurnResponse(t, router, "episode-unsupported",
		`{"model":"auto","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	if refusal == nil || int(refusal.GetStatus().GetCode()) != 422 ||
		immediateHeader(refusal, "x-vsr-failure-class") != "undecidable" {
		t.Fatalf("refusal %v, want 422 undecidable", refusal.GetStatus())
	}
	if undecidableFailures()-before != 1 {
		t.Fatal("the turn did not fail as policy_undecidable")
	}
	if fields := findLogEvent(t, logs, "rayline_arc_undecidable"); fields["reason"] != "no_supported_selection" {
		t.Fatalf("rayline_arc_undecidable reason = %v, want no_supported_selection", fields["reason"])
	}
}
