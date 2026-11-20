package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func reportContext(headers map[string]string) *RequestContext {
	captured := map[string]string{":path": raylineReportAPIPath, ":method": "POST"}
	for name, value := range headers {
		captured[name] = value
	}
	return &RequestContext{Headers: captured}
}

// reportAnswer is a /v1/report answer as a caller reads it.
type reportAnswer struct {
	Object     string  `json:"object"`
	Status     string  `json:"status"`
	Reason     string  `json:"reason"`
	RouteID    *string `json:"route_id"`
	TurnIndex  *uint64 `json:"turn_index"`
	Attributed *bool   `json:"attributed"`
}

// postReport posts one report on conversation and returns its status and
// raw body.
func (fixture *pendingRoutesFixture) postReport(t *testing.T, conversation, source string, report map[string]any) (int, []byte) {
	t.Helper()
	headers := map[string]string{}
	if conversation != "" {
		headers[raylineRoutesSessionHeader] = conversation
	}
	if source != "" {
		headers[raylineReportSourceHeader] = source
	}
	body, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	response := fixture.router.handleRaylineReportAPI(body, reportContext(headers))
	if response == nil {
		t.Fatal("the report was not answered by the router")
	}
	return immediateStatusCode(t, response), response.GetImmediateResponse().Body
}

// report posts a report expected to be answered 200 and returns the answer.
func (fixture *pendingRoutesFixture) report(t *testing.T, conversation, source string, report map[string]any) reportAnswer {
	t.Helper()
	code, body := fixture.postReport(t, conversation, source, report)
	if code != 200 {
		t.Fatalf("report status %d: %s", code, body)
	}
	var answer reportAnswer
	if err := json.Unmarshal(body, &answer); err != nil {
		t.Fatal(err)
	}
	if answer.Object != "route.report" {
		t.Fatalf("answer %s", body)
	}
	return answer
}

// executedReport is a report body of model, finished now, with usage.
func executedReport(routeID, model string, extra map[string]any) map[string]any {
	executed := map[string]any{
		"model":       model,
		"finished_at": time.Now().UTC().Format(time.RFC3339Nano),
		"usage": map[string]any{
			"prompt_tokens": 1200, "cache_read_tokens": 1000, "cache_write_tokens": 100, "cache_ttl_seconds": 300,
		},
	}
	for name, value := range extra {
		executed[name] = value
	}
	body := map[string]any{"executed": executed}
	if routeID != "" {
		body["route_id"] = routeID
	}
	return body
}

// A lookup, its report of the decided model, and the next turn's lookup: the
// pending decision becomes a reported turn with the reported usage, and the
// next decide attributes the reply to the reported action.
func TestRoutesReportOfTheDecidedModel(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.picks = []string{fixture.actions[2]}
	const conversation = "conversation-report"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))

	answer := fixture.report(t, conversation, "", executedReport(first.RouteID, first.Model, nil))
	if answer.Status != "recorded" || answer.Reason != raylinearc.ReportPendingResolved || *answer.RouteID != first.RouteID ||
		*answer.TurnIndex != 1 || !*answer.Attributed {
		t.Fatalf("answer %+v", answer)
	}
	state := fixture.stored(t, conversation)
	if state.Pending != nil || state.TurnIndex != 1 || state.PreviousArm == nil || *state.PreviousArm != 1 {
		t.Fatalf("record %+v", state)
	}
	entry := state.Policy.Ledger[0]
	if entry.Message != 1 || entry.ActionID != fixture.actions[2] || entry.Basis != raylinearc.PolicyLedgerBasisReported {
		t.Fatalf("entry %+v", entry)
	}
	call := state.Warmth[1].Call
	if call == nil || *call.PromptTokens != 1200 || *call.CacheReadTokens != 1000 || *call.CacheWriteTokens != 100 ||
		call.TTLSeconds != 300 || call.TTLSource != raylinearc.ReportedTTLSource || call.ActionID != fixture.actions[2] {
		t.Fatalf("call %+v", call)
	}
	if state.LastReport == nil || state.LastReport.Source != raylinearc.ReportSourceClient {
		t.Fatalf("last report %+v", state.LastReport)
	}

	next := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	sent := fixture.policy.fake.received()
	attribution := sent[len(sent)-1].Attribution
	if len(attribution) != 1 || attribution[0].Message != 1 || attribution[0].ActionID != fixture.actions[2] {
		t.Fatalf("next decide attribution %+v", attribution)
	}
	if next.Episode == nil || next.Episode.TurnIndex != 2 {
		t.Fatalf("next episode %+v", next.Episode)
	}
}

// A report of another model than the decided one records that model's arm
// and action, leaves the decided arm's warmth as it was, and the next decide
// attributes the reply to the reported action with no trained-arm id.
func TestRoutesReportOfAnotherModel(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.picks = []string{fixture.actions[0]}
	const conversation = "conversation-other"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	before := fixture.stored(t, conversation)

	answer := fixture.report(t, conversation, "", executedReport(first.RouteID, "off", nil))
	if answer.Status != "recorded" || !*answer.Attributed {
		t.Fatalf("answer %+v", answer)
	}
	state := fixture.stored(t, conversation)
	if *state.PreviousArm != 1 || state.Policy.Ledger[0].ActionID != fixture.actions[2] || state.Policy.Ledger[0].ArmID != "" {
		t.Fatalf("record %+v ledger %+v", state, state.Policy.Ledger)
	}
	if state.Warmth[0] != before.Warmth[0] && (state.Warmth[0] == nil) != (before.Warmth[0] == nil) {
		t.Fatalf("the decided arm's warmth changed: %+v", state.Warmth[0])
	}
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	sent := fixture.policy.fake.received()
	attribution := sent[len(sent)-1].Attribution
	if len(attribution) != 1 || attribution[0].ActionID != fixture.actions[2] || attribution[0].ArmID != nil {
		t.Fatalf("next decide attribution %+v", attribution)
	}
}

// Rule (owner's choice): a report of a model no arm of the package serves (a
// local model) is recorded as a turn with no arm -- counted, the held arm
// cleared, no ledger or usage entry -- not refused.
func TestReportRuleModelOutsideThePackageIsRecordedUnattributed(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-local"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	answer := fixture.report(t, conversation, "", executedReport(first.RouteID, "local/llama-4", nil))
	if answer.Status != "recorded" || answer.Reason != raylinearc.ReportPendingResolved || *answer.Attributed {
		t.Fatalf("answer %+v", answer)
	}
	state := fixture.stored(t, conversation)
	if state.TurnIndex != 1 || state.PreviousArm != nil || len(state.Policy.Ledger) != 0 || state.Pending != nil {
		t.Fatalf("record %+v", state)
	}
	for arm, warmth := range state.Warmth {
		if warmth != nil {
			t.Fatalf("arm %d warmth %+v, want none", arm, warmth)
		}
	}
}

// Rule (owner's choice): a report that changed nothing is answered 200 with
// status ignored and the reason, not an error status, so a caller that
// reports and moves on does not retry it.
func TestReportRuleIgnoredReportsAnswer200WithReason(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-ignored"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	fixture.report(t, conversation, "", executedReport(first.RouteID, first.Model, nil))
	code, body := fixture.postReport(t, conversation, "", executedReport(first.RouteID, first.Model, nil))
	if code != 200 {
		t.Fatalf("a duplicate report was answered %d: %s", code, body)
	}
	var answer map[string]any
	if err := json.Unmarshal(body, &answer); err != nil {
		t.Fatal(err)
	}
	if answer["status"] != "ignored" || answer["reason"] != raylinearc.ReportDuplicate ||
		answer["route_id"] != nil || answer["turn_index"] != nil || answer["attributed"] != nil {
		t.Fatalf("answer %s", body)
	}
}

// A report after the next turn already recorded its decision as inferred is
// late: ignored, the record unchanged.
func TestRoutesReportAfterTheNextTurnIsIgnored(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-late"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask, pendingTurnReply, pendingTurnNext))
	before := fixture.stored(t, conversation)
	answer := fixture.report(t, conversation, "", executedReport(first.RouteID, first.Model, nil))
	if answer.Status != "ignored" || answer.Reason != raylinearc.ReportNotPending {
		t.Fatalf("answer %+v", answer)
	}
	if after := fixture.stored(t, conversation); after.Policy.Ledger[0].Basis != raylinearc.PolicyLedgerBasisInferred ||
		after.LastReport != nil || after.Pending.RouteID != before.Pending.RouteID {
		t.Fatalf("a late report changed the record: %+v", after)
	}
}

// With no pending decision, a report naming no route id -- here the
// gateway's report of a later pinned call -- changes nothing and is answered
// not_pending; on a conversation the cell never saw it is no_episode, and no
// record, not even a lease, is left behind.
func TestRoutesReportWithoutPendingDecision(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-pinned"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	pinned := executedReport("", first.Model, nil)
	pinned["report_id"] = "rt_pinned_1"
	fixture.report(t, conversation, raylinearc.ReportSourceGateway, pinned)
	pinned = executedReport("", "off", nil)
	pinned["report_id"] = "rt_pinned_2"
	answer := fixture.report(t, conversation, raylinearc.ReportSourceGateway, pinned)
	if answer.Status != "ignored" || answer.Reason != raylinearc.ReportNotPending {
		t.Fatalf("answer %+v", answer)
	}
	answer = fixture.report(t, "conversation-unknown", raylinearc.ReportSourceGateway, executedReport("", "off", nil))
	if answer.Status != "ignored" || answer.Reason != raylinearc.ReportNoEpisode {
		t.Fatalf("answer %+v", answer)
	}
	if state := fixture.stored(t, "conversation-unknown"); state != nil {
		t.Fatalf("a report on an unknown conversation stored %+v", state)
	}
}

// The edge's report wins: after it, a client's report of the same turn is
// superseded.
func TestRoutesReportEdgeBeforeClient(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-edge"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	if answer := fixture.report(t, conversation, raylinearc.ReportSourceGateway, executedReport("", first.Model, nil)); answer.Reason != raylinearc.ReportPendingResolved {
		t.Fatalf("edge answer %+v", answer)
	}
	if answer := fixture.report(t, conversation, "", executedReport(first.RouteID, first.Model, nil)); answer.Reason != raylinearc.ReportSuperseded {
		t.Fatalf("client answer %+v", answer)
	}
}

// With episode writes off the report is validated, then answered
// not_tracked; nothing is stored.
func TestRoutesReportWithEpisodeWritesOff(t *testing.T) {
	fixture := newPendingRoutesFixture(t, false, "")
	answer := fixture.report(t, "conversation-off", "", executedReport("rte_1", "off", nil))
	if answer.Status != "ignored" || answer.Reason != raylineReportNotTracked {
		t.Fatalf("answer %+v", answer)
	}
	if state := fixture.stored(t, "conversation-off"); state != nil {
		t.Fatalf("stored %+v", state)
	}
	if code, _ := fixture.postReport(t, "conversation-off", "", map[string]any{"executed": map[string]any{}}); code != 400 {
		t.Fatalf("a malformed report with writes off was answered %d", code)
	}
}

// The model a report names maps to an action: an action id, the model with
// its thinking level, the model alone (the decided action when it is the
// model's, else its only action); a level or action the model does not have
// is refused.
func TestRoutesReportModelMapping(t *testing.T) {
	cases := map[string]struct {
		pick   int
		model  string
		extra  map[string]any
		action int
		status int
	}{
		"model alone, the decided level":  {pick: 1, model: "think", action: 1, status: 200},
		"model and level":                 {pick: 0, model: "think", extra: map[string]any{"thinking_level": "up"}, action: 1, status: 200},
		"model alone, its only action":    {pick: 0, model: "off", action: 2, status: 200},
		"the action's model name":         {pick: 0, model: "vendor/off", action: 2, status: 200},
		"unknown level for the model":     {pick: 0, model: "think", extra: map[string]any{"thinking_level": "max"}, status: 400},
		"action of another model":         {pick: 0, model: "off", extra: map[string]any{"action_id": "@0"}, status: 400},
		"action id of the model":          {pick: 0, model: "think", extra: map[string]any{"action_id": "@1"}, action: 1, status: 200},
		"several levels, none decided":    {pick: 2, model: "think", action: -1, status: 200},
		"malformed action id":             {pick: 0, model: "think", extra: map[string]any{"action_id": "x"}, status: 400},
		"thinking level of unknown model": {pick: 0, model: "local/x", extra: map[string]any{"thinking_level": "up"}, action: -2, status: 200},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			fixture := newPendingRoutesFixture(t, true, "")
			fixture.picks = []string{fixture.actions[tc.pick]}
			first := fixture.lookup(t, "conversation-map", pendingRoutesBody(t, pendingTurnTask))
			extra := map[string]any{}
			for key, value := range tc.extra {
				if text, ok := value.(string); ok && strings.HasPrefix(text, "@") {
					value = fixture.actions[int(text[1]-'0')]
				}
				extra[key] = value
			}
			code, body := fixture.postReport(t, "conversation-map", "", executedReport(first.RouteID, tc.model, extra))
			if code != tc.status {
				t.Fatalf("status %d, want %d: %s", code, tc.status, body)
			}
			if code != 200 {
				return
			}
			state := fixture.stored(t, "conversation-map")
			switch tc.action {
			case -2:
				if state.PreviousArm != nil || len(state.Policy.Ledger) != 0 {
					t.Fatalf("record %+v", state)
				}
			case -1:
				if len(state.Policy.Ledger) != 0 || state.PreviousArm == nil || *state.PreviousArm != 0 {
					t.Fatalf("record %+v ledger %+v", state, state.Policy.Ledger)
				}
			default:
				if len(state.Policy.Ledger) != 1 || state.Policy.Ledger[0].ActionID != fixture.actions[tc.action] {
					t.Fatalf("ledger %+v, want action %d", state.Policy.Ledger, tc.action)
				}
			}
		})
	}
}

// Malformed reports are refused in the routes envelope; the source header
// is only ever gateway or client.
func TestRoutesReportRefusals(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	future := time.Now().Add(5 * time.Minute).UTC().Format(time.RFC3339)
	cases := map[string]struct {
		conversation string
		source       string
		body         map[string]any
		status       int
		code         string
	}{
		"no session":        {"", "", executedReport("rte_1", "off", nil), 400, "missing_session"},
		"no executed":       {"c", "", map[string]any{"route_id": "rte_1"}, 400, "invalid_report"},
		"no model":          {"c", "", map[string]any{"executed": map[string]any{"finished_at": future}}, 400, "invalid_report"},
		"unknown member":    {"c", "", map[string]any{"executed": map[string]any{"model": "off", "finished_at": future}, "cost": 1}, 400, "invalid_report"},
		"future finish":     {"c", "", executedReport("", "off", map[string]any{"finished_at": future}), 400, "invalid_report"},
		"negative tokens":   {"c", "", executedReport("", "off", map[string]any{"usage": map[string]any{"prompt_tokens": -1}}), 400, "invalid_report"},
		"long route id":     {"c", "", executedReport(strings.Repeat("r", 129), "off", nil), 400, "invalid_report"},
		"unknown source":    {"c", "admin", executedReport("", "off", nil), 400, "invalid_report"},
		"output tokens":     {"c", "", executedReport("", "off", map[string]any{"usage": map[string]any{"output_tokens": 5}}), 400, "invalid_report"},
		"unknown tier":      {"c", "", map[string]any{"model": "arc-9", "executed": map[string]any{"model": "off", "finished_at": time.Now().UTC().Format(time.RFC3339)}}, 400, "unknown_model"},
		"ttl out of bounds": {"c", "", executedReport("", "off", map[string]any{"usage": map[string]any{"cache_ttl_seconds": 0}}), 400, "invalid_report"},
	}
	for name, tc := range cases {
		code, body := fixture.postReport(t, tc.conversation, tc.source, tc.body)
		if code != tc.status || !strings.Contains(string(body), `"code":"`+tc.code+`"`) {
			t.Fatalf("%s: status %d body %s, want %d %s", name, code, body, tc.status, tc.code)
		}
	}
	oversize := fixture.router.handleRaylineReportAPI(make([]byte, raylineReportMaxBodyBytes+1),
		reportContext(map[string]string{raylineRoutesSessionHeader: "c"}))
	if code := immediateStatusCode(t, oversize); code != 413 {
		t.Fatalf("oversize body: %d", code)
	}
}

// The report path is served by this router exactly as /v1/routes is: hidden
// where routes are off, POST only, a bodyless POST refused at the header
// phase, and never forwarded on the processing opt-out.
func TestRoutesReportPathIsServedHere(t *testing.T) {
	for name, tc := range map[string]struct {
		enabled bool
		method  string
		eos     bool
		status  int
	}{
		"off is not found": {false, "POST", false, 404},
		"wrong method":     {true, "GET", false, 405},
		"bodyless post":    {true, "POST", true, 400},
	} {
		ctx := reportContext(nil)
		response, err := routesRouter(tc.enabled).handleRequestHeaders(&ext_proc.ProcessingRequest_RequestHeaders{
			RequestHeaders: &ext_proc.HttpHeaders{Headers: routesHeaderMap(tc.method, raylineReportAPIPath), EndOfStream: tc.eos},
		}, ctx)
		if err != nil || response.GetImmediateResponse() == nil {
			t.Fatalf("%s: not answered here (%v)", name, err)
		}
		if code := immediateStatusCode(t, response); code != tc.status || !ctx.ImmediateResponseEncoded {
			t.Fatalf("%s: status %d (claimed %t), want %d", name, code, ctx.ImmediateResponseEncoded, tc.status)
		}
	}
	ctx := reportContext(nil)
	ctx.SkipProcessing = true
	response, err := routesRouter(true).HandleRequestBody(&ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(`{"executed":{}}`)},
	}, ctx)
	if err != nil || response.GetImmediateResponse() == nil {
		t.Fatalf("a skip-processing report was forwarded upstream (%v)", err)
	}
}

// A report takes the conversation's lease like any write: while another
// request holds it past the report's deadline, the report is answered busy
// with a retry-after and changes nothing; once released, it lands.
func TestRoutesReportWaitsOnTheEpisodeLease(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	fixture.router.Config.Decisions[0].Algorithm.RaylineARC.RoutesAPI.DeadlineMS = 200
	const conversation = "conversation-busy"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	lease, _, err := fixture.store.Prepare(context.Background(), raylinearc.HashEpisodeID(conversation), 2)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(executedReport(first.RouteID, first.Model, nil))
	response := fixture.router.handleRaylineReportAPI(body, reportContext(map[string]string{raylineRoutesSessionHeader: conversation}))
	if code := immediateStatusCode(t, response); code != 429 || !strings.Contains(string(response.GetImmediateResponse().Body), "episode_busy") {
		t.Fatalf("status %d: %s", code, response.GetImmediateResponse().Body)
	}
	if !hasHeader(response, "retry-after") {
		t.Fatal("a busy report carries no retry-after")
	}
	if state := fixture.stored(t, conversation); state.Pending == nil {
		t.Fatal("a refused report changed the record")
	}
	if err := fixture.store.Abort(context.Background(), lease); err != nil {
		t.Fatal(err)
	}
	if answer := fixture.report(t, conversation, "", executedReport(first.RouteID, first.Model, nil)); answer.Status != "recorded" {
		t.Fatalf("answer %+v", answer)
	}
}

// A relaxed cell commits a report if the episode is unchanged since it was
// read, as its lookups commit.
func TestRoutesReportOnARelaxedCell(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, config.RaylineARCConsistencyRelaxed)
	const conversation = "conversation-relaxed"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	if answer := fixture.report(t, conversation, "", executedReport(first.RouteID, first.Model, nil)); answer.Reason != raylinearc.ReportPendingResolved {
		t.Fatalf("answer %+v", answer)
	}
	if state := fixture.stored(t, conversation); state.Pending != nil || state.TurnIndex != 1 {
		t.Fatalf("record %+v", state)
	}
}

func hasHeader(response *ext_proc.ProcessingResponse, name string) bool {
	for _, header := range response.GetImmediateResponse().GetHeaders().GetSetHeaders() {
		if header.GetHeader().GetKey() == name {
			return true
		}
	}
	return false
}

// Anything after the report object is refused before the tier is read: a
// trailing delimiter must not let a report for an unknown tier fall back to
// the default tier's episode.
func TestRoutesReportRefusesTrailingData(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	first := fixture.lookup(t, "conversation-trailing", pendingRoutesBody(t, pendingTurnTask))
	body, _ := json.Marshal(map[string]any{"model": "arc-9", "route_id": first.RouteID, "executed": map[string]any{
		"model": first.Model, "finished_at": time.Now().UTC().Format(time.RFC3339),
	}})
	for _, trailing := range []string{"}", " {}", "x"} {
		response := fixture.router.handleRaylineReportAPI(append(append([]byte{}, body...), trailing...),
			reportContext(map[string]string{raylineRoutesSessionHeader: "conversation-trailing"}))
		if code := immediateStatusCode(t, response); code != 400 {
			t.Fatalf("trailing %q: status %d: %s", trailing, code, response.GetImmediateResponse().Body)
		}
	}
	if state := fixture.stored(t, "conversation-trailing"); state.Pending == nil {
		t.Fatal("a refused report changed the record")
	}
}

// A finish time a little ahead of the cell's clock is accepted as sent; a
// retry of the report, nothing being pending any more, changes nothing.
func TestRoutesReportKeepsAFinishTimeAheadOfTheClock(t *testing.T) {
	fixture := newPendingRoutesFixture(t, true, "")
	const conversation = "conversation-skew"
	first := fixture.lookup(t, conversation, pendingRoutesBody(t, pendingTurnTask))
	report := executedReport("", first.Model, map[string]any{"finished_at": time.Now().Add(30 * time.Second).UTC().Format(time.RFC3339Nano)})
	if answer := fixture.report(t, conversation, "", report); answer.Reason != raylinearc.ReportPendingResolved {
		t.Fatalf("first: %+v", answer)
	}
	time.Sleep(5 * time.Millisecond)
	if answer := fixture.report(t, conversation, "", report); answer.Status != "ignored" || answer.Reason != raylinearc.ReportNotPending {
		t.Fatalf("retry: %+v", answer)
	}
}

// When the report's model is known but its action is not (several levels,
// none named), the turn records the reported model, not another model the
// same worker also binds.
func TestReportedRunRecordsTheReportedModelWhenTheActionIsUnknown(t *testing.T) {
	scorer := &policyServiceScorer{
		workerIDs: []string{"shared"},
		workers:   []raylinearc.WorkerManifest{{ID: "shared", Model: "vendor/shared"}},
		bindings: map[string]policyBinding{
			"a0": {arm: 0, model: "vendor/a", level: "none"},
			"b0": {arm: 0, model: "vendor/b", level: "none"},
			"b1": {arm: 0, model: "vendor/b", level: "up"},
		},
		actionOrder: []string{"a0", "b0", "b1"},
	}
	report := raylineReport{model: "vendor/b"}
	candidates, refusal := scorer.reportCandidates(report)
	if refusal != nil || len(candidates) != 2 {
		t.Fatalf("candidates %v (%v)", candidates, refusal)
	}
	run := scorer.reportedRun(report, candidates, "a0")
	if run.Arm != 0 || run.ActionID != "" || run.Model != "vendor/b" {
		t.Fatalf("run %+v, want arm 0, no action, model vendor/b", run)
	}
}
