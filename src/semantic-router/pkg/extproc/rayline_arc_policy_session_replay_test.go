//go:build !windows && cgo

package extproc

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// A session replay drives a script of client calls through the router's real
// ext_proc stream, one Process call per HTTP exchange: the request phases,
// then the scripted upstream response (status and body chunks, or a stream
// cut mid-response). Nothing leaves the process for a provider; the upstream
// is the script.
//
// The policy service is a local server that records every decide call's
// bytes and answers it either from the script (oracle mode: each call names
// the action to select) or from a recording of a real service's answers to
// the same request bytes (playback mode). A playback replay whose every
// decide request is the recording's next one, byte for byte, is
// the router and the service in closed loop: each answer the router consumed
// is the one the service gave to that exact request.
//
// After each call the episode's committed state is read back, and calls that
// carry expectations are checked against them: the decide inputs (episode,
// context epoch, attribution, available and held actions) and the state
// (completed turns, epoch, ledger size, compaction boundary).
//
// It runs only when RAYLINE_ARC_SESSION_REPLAY names a spec file.

type sessionReplaySpec struct {
	// Config is a router config whose policy_service.base_url is the literal
	// REPLAY_POLICY_URL.
	Config        string            `json:"config"`
	Script        string            `json:"script"`
	Out           string            `json:"out"`
	Alias         string            `json:"alias"`
	PackageSHA256 string            `json:"package_sha256"`
	Catalog       []string          `json:"catalog"`
	Workers       int               `json:"workers"`
	Env           map[string]string `json:"env"`
	// Playback, when set, is a JSONL recording of {request, status,
	// response}; otherwise the oracle answers.
	Playback string `json:"playback"`
	// OracleRefuseRelaxed, when set, makes the oracle answer every relaxed
	// decide 422 unsupported_request with this reason.
	OracleRefuseRelaxed string `json:"oracle_refuse_relaxed"`
}

type sessionReplayCall struct {
	Seq      int               `json:"seq"`
	Kind     string            `json:"kind"`
	CaseID   string            `json:"case_id"`
	Episode  string            `json:"episode"`
	Headers  map[string]string `json:"headers"`
	Body     string            `json:"body"`
	Response struct {
		Status      int      `json:"status"`
		ContentType string   `json:"content_type"`
		Chunks      []string `json:"chunks"`
		// Cut ends the stream with a receive error after the chunks.
		Cut bool `json:"cut"`
	} `json:"response"`
	Oracle *struct {
		SelectedActionID string `json:"selected_action_id"`
		SelectedArmID    string `json:"selected_arm_id"`
	} `json:"oracle"`
	Expect *struct {
		EpisodeIDHash      string                         `json:"episode_id_hash"`
		ContextEpoch       string                         `json:"context_epoch"`
		RequestFormat      string                         `json:"request_format"`
		Attribution        []raylinearc.PolicyAttribution `json:"attribution"`
		AvailableActionIDs []string                       `json:"available_action_ids"`
		HeldActionID       *string                        `json:"held_action_id"`
		// Decides, when set, is how many decide calls this exchange makes.
		Decides *int `json:"decides"`
	} `json:"expect"`
	ExpectState *sessionReplayExpectedState `json:"expect_state"`
}

type sessionReplayState struct {
	TurnIndex       uint64 `json:"turn_index"`
	Epoch           int    `json:"epoch"`
	Ledger          int    `json:"ledger"`
	EpochStartTurn  uint64 `json:"epoch_start_turn"`
	CompactionCount int    `json:"compaction_count"`
	PreviousArm     *int   `json:"previous_arm"`
}

// sessionReplayExpectedState is the committed state a call must leave.
// CompactionCount and PreviousArm are checked only when the script sets them.
type sessionReplayExpectedState struct {
	TurnIndex       uint64 `json:"turn_index"`
	Epoch           int    `json:"epoch"`
	Ledger          int    `json:"ledger"`
	EpochStartTurn  uint64 `json:"epoch_start_turn"`
	CompactionCount *int   `json:"compaction_count"`
	PreviousArm     *int   `json:"previous_arm"`
}

type sessionReplayExchange struct {
	Request  string `json:"request"`
	Status   int    `json:"status"`
	Response string `json:"response"`
	// Unmatched marks a playback request the recording does not hold.
	Unmatched bool `json:"unmatched,omitempty"`
}

type sessionReplayResult struct {
	Seq             int                     `json:"seq"`
	CaseID          string                  `json:"case_id"`
	Kind            string                  `json:"kind"`
	Decides         []sessionReplayExchange `json:"decides"`
	ImmediateStatus int                     `json:"immediate_status"`
	ImmediateBody   string                  `json:"immediate_body,omitempty"`
	ProcessError    string                  `json:"process_error,omitempty"`
	ProviderModel   string                  `json:"provider_model"`
	State           *sessionReplayState     `json:"state"`
	Mismatches      []string                `json:"mismatches"`
}

func TestRaylineARCPolicySessionReplay(t *testing.T) {
	specPath := os.Getenv("RAYLINE_ARC_SESSION_REPLAY")
	if specPath == "" {
		t.Skip("RAYLINE_ARC_SESSION_REPLAY names no replay spec")
	}
	var spec sessionReplaySpec
	readJSONFile(t, specPath, &spec)
	for key, value := range spec.Env {
		t.Setenv(key, value)
	}
	calls := readSessionReplayScript(t, spec.Script)
	service := newSessionReplayService(t, spec)
	configText, err := os.ReadFile(spec.Config)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(configText), "REPLAY_POLICY_URL") {
		t.Fatal("the config does not name REPLAY_POLICY_URL")
	}
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	rendered := strings.ReplaceAll(string(configText), "REPLAY_POLICY_URL", service.server.URL)
	if err := os.WriteFile(configPath, []byte(rendered), 0o600); err != nil {
		t.Fatal(err)
	}
	router, err := NewOpenAIRouter(configPath)
	if err != nil {
		t.Fatalf("build router: %v", err)
	}
	awaitPolicySelectorArmed(t, router)

	out, err := os.Create(spec.Out)
	if err != nil {
		t.Fatal(err)
	}
	defer out.Close()
	writer := bufio.NewWriter(out)
	defer writer.Flush()
	// Decide calls before the first client call (a relaxed cell's readiness
	// probe) are part of the conversation with the service, so they are
	// recorded, as seq -1, and played back like any other.
	readiness, err := json.Marshal(sessionReplayResult{
		Seq: -1, Kind: "readiness", Decides: service.end(), Mismatches: []string{},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, _ = writer.Write(append(readiness, '\n'))
	failures := 0
	for _, call := range calls {
		result := runSessionReplayCall(t, router, service, spec, call)
		if len(result.Mismatches) > 0 {
			failures++
		}
		line, err := json.Marshal(result)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = writer.Write(append(line, '\n'))
	}
	if unmatched := service.unmatchedCount(); unmatched > 0 {
		t.Errorf("%d decide requests are not in the recording", unmatched)
	}
	if left := service.unusedCount(); left > 0 {
		t.Errorf("%d recorded decide exchanges were never requested", left)
	}
	if failures > 0 {
		t.Errorf("%d of %d calls did not meet their expectations (see %s)", failures, len(calls), spec.Out)
	}
}

func runSessionReplayCall(
	t *testing.T,
	router *OpenAIRouter,
	service *sessionReplayService,
	spec sessionReplaySpec,
	call sessionReplayCall,
) sessionReplayResult {
	t.Helper()
	result := sessionReplayResult{Seq: call.Seq, CaseID: call.CaseID, Kind: call.Kind, Mismatches: []string{}}
	service.begin(call)
	headers := []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: "/v1/messages"},
		{Key: "content-type", Value: "application/json"},
	}
	for key, value := range call.Headers {
		headers = append(headers, &core.HeaderValue{Key: key, Value: value})
	}
	messages := []*ext_proc.ProcessingRequest{
		{Request: &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: headers},
		}}},
		{Request: &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{
			Body: []byte(call.Body), EndOfStream: true,
		}}},
		{Request: &ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":status", Value: fmt.Sprint(call.Response.Status)},
				{Key: "content-type", Value: call.Response.ContentType},
			}},
		}}},
	}
	for index, chunk := range call.Response.Chunks {
		messages = append(messages, &ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{
				Body: []byte(chunk), EndOfStream: !call.Response.Cut && index == len(call.Response.Chunks)-1,
			},
		}})
	}
	end := io.EOF
	if call.Response.Cut {
		end = status.Error(codes.Canceled, "stream cut mid-response")
	}
	stream := &sessionReplayStream{requests: messages, end: end, ctx: context.Background()}
	if err := router.Process(stream); err != nil && !(call.Response.Cut && status.Code(err) == codes.Canceled) {
		result.ProcessError = err.Error()
		if call.Kind != "failed_attempt" {
			result.Mismatches = append(result.Mismatches, "process: "+err.Error())
		}
	}
	result.ImmediateStatus, result.ImmediateBody, result.ProviderModel = stream.outcome()
	if result.ImmediateStatus != 0 && call.Kind != "failed_attempt" {
		result.Mismatches = append(result.Mismatches, fmt.Sprintf("refused: %d %s", result.ImmediateStatus, result.ImmediateBody))
	}
	result.Decides = service.end()
	result.State = readSessionReplayState(t, router, call.Episode, spec.Workers)

	if expect := call.Expect; expect != nil {
		switch {
		case expect.Decides != nil && len(result.Decides) != *expect.Decides:
			result.Mismatches = append(result.Mismatches,
				fmt.Sprintf("decide calls: got %d, want %d", len(result.Decides), *expect.Decides))
		case expect.Decides == nil && len(result.Decides) == 0:
			// Expected inputs with no decide call to compare them against is a
			// failure, not a pass; a script that means zero calls says so.
			result.Mismatches = append(result.Mismatches, "decide calls: none, but the call expects decide inputs")
		}
		if len(result.Decides) > 0 {
			var sent raylinearc.PolicyDecisionRequest
			// The last call is the one whose answer was served (a refused
			// relaxed side call is retried strict).
			if err := json.Unmarshal([]byte(result.Decides[len(result.Decides)-1].Request), &sent); err != nil {
				result.Mismatches = append(result.Mismatches, "decide request: "+err.Error())
			} else {
				result.Mismatches = append(result.Mismatches, compareSessionReplayInputs(sent, call)...)
			}
		}
	}
	if want := call.ExpectState; want != nil {
		if result.State == nil {
			result.Mismatches = append(result.Mismatches, "episode state: none")
		} else if got := *result.State; got.TurnIndex != want.TurnIndex || got.Epoch != want.Epoch ||
			got.Ledger != want.Ledger || got.EpochStartTurn != want.EpochStartTurn ||
			(want.CompactionCount != nil && got.CompactionCount != *want.CompactionCount) ||
			(want.PreviousArm != nil && (got.PreviousArm == nil || *got.PreviousArm != *want.PreviousArm)) {
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(want)
			result.Mismatches = append(result.Mismatches, fmt.Sprintf("episode state: got %s, want %s", gotJSON, wantJSON))
		}
	}
	return result
}

func compareSessionReplayInputs(sent raylinearc.PolicyDecisionRequest, call sessionReplayCall) []string {
	expect := call.Expect
	var mismatches []string
	check := func(field string, got, want any) {
		if !reflect.DeepEqual(got, want) {
			gotJSON, _ := json.Marshal(got)
			wantJSON, _ := json.Marshal(want)
			mismatches = append(mismatches, fmt.Sprintf("%s: got %s, want %s", field, gotJSON, wantJSON))
		}
	}
	check("episode_id_hash", sent.EpisodeIDHash, expect.EpisodeIDHash)
	check("context_epoch", sent.ContextEpoch, expect.ContextEpoch)
	check("request_format", sent.RequestFormat, expect.RequestFormat)
	attribution := sent.Attribution
	if attribution == nil {
		attribution = []raylinearc.PolicyAttribution{}
	}
	wantAttribution := expect.Attribution
	if wantAttribution == nil {
		wantAttribution = []raylinearc.PolicyAttribution{}
	}
	check("attribution", attribution, wantAttribution)
	check("available_action_ids", sent.Selection.AvailableActionIDs, expect.AvailableActionIDs)
	check("held_action_id", sent.Selection.HeldActionID, expect.HeldActionID)
	return mismatches
}

func readSessionReplayState(t *testing.T, router *OpenAIRouter, episode string, workers int) *sessionReplayState {
	t.Helper()
	store := router.RaylineARCEpisodeStore
	if store == nil {
		t.Fatal("the router has no episode store")
	}
	var state *raylinearc.EpisodeState
	if snapshots, ok := store.(raylinearc.EpisodeSnapshotStore); ok {
		// A read that takes no lease, so it changes nothing a relaxed
		// commit's version check reads.
		read, _, err := snapshots.Snapshot(context.Background(), raylinearc.HashEpisodeID(episode), workers)
		if err != nil {
			t.Fatalf("read episode %q: %v", episode, err)
		}
		state = read
	} else {
		lease, read, err := store.Prepare(context.Background(), raylinearc.HashEpisodeID(episode), workers)
		if err != nil {
			t.Fatalf("read episode %q: %v", episode, err)
		}
		defer func() { _ = store.Abort(context.Background(), lease) }()
		state = read
	}
	if state == nil {
		return nil
	}
	read := &sessionReplayState{TurnIndex: state.TurnIndex, PreviousArm: state.PreviousArm}
	if policy := state.Policy; policy != nil {
		read.Epoch, read.Ledger = policy.Epoch, len(policy.Ledger)
		read.EpochStartTurn, read.CompactionCount = policy.EpochStartTurn, policy.CompactionCount
	}
	return read
}

// sessionReplayStream feeds one exchange's ext_proc messages and records the
// router's answers. Once the router answers with an immediate response, as
// Envoy would, nothing further is sent.
type sessionReplayStream struct {
	ext_proc.ExternalProcessor_ProcessServer
	requests  []*ext_proc.ProcessingRequest
	end       error
	ctx       context.Context
	mu        sync.Mutex
	next      int
	responses []*ext_proc.ProcessingResponse
}

func (stream *sessionReplayStream) Context() context.Context { return stream.ctx }

func (stream *sessionReplayStream) Send(response *ext_proc.ProcessingResponse) error {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	stream.responses = append(stream.responses, response)
	return nil
}

func (stream *sessionReplayStream) Recv() (*ext_proc.ProcessingRequest, error) {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	for _, response := range stream.responses {
		if response.GetImmediateResponse() != nil {
			return nil, io.EOF
		}
	}
	if stream.next >= len(stream.requests) {
		return nil, stream.end
	}
	request := stream.requests[stream.next]
	stream.next++
	return request, nil
}

// outcome is the immediate response's status and body, if the router
// refused the request, and the model of the provider-bound body otherwise.
func (stream *sessionReplayStream) outcome() (int, string, string) {
	stream.mu.Lock()
	defer stream.mu.Unlock()
	for _, response := range stream.responses {
		if immediate := response.GetImmediateResponse(); immediate != nil {
			body := string(immediate.GetBody())
			if len(body) > 600 {
				body = body[:600]
			}
			return int(immediate.GetStatus().GetCode()), body, ""
		}
		if mutation := response.GetRequestBody().GetResponse().GetBodyMutation(); mutation != nil {
			var body struct {
				Model string `json:"model"`
			}
			_ = json.Unmarshal(mutation.GetBody(), &body)
			return 0, "", body.Model
		}
	}
	return 0, "", ""
}

// sessionReplayService is the policy service a replay talks to.
type sessionReplayService struct {
	t        *testing.T
	server   *httptest.Server
	listing  *fakePolicyService
	spec     sessionReplaySpec
	mu       sync.Mutex
	current  sessionReplayCall
	captured []sessionReplayExchange
	// recorded is the recording in its global order, and next the index of
	// the exchange the next decide request must match.
	recorded  []sessionReplayExchange
	next      int
	unmatched int
}

func newSessionReplayService(t *testing.T, spec sessionReplaySpec) *sessionReplayService {
	t.Helper()
	service := &sessionReplayService{
		t: t, spec: spec,
		listing: &fakePolicyService{t: t, alias: spec.Alias, sha256: spec.PackageSHA256, catalog: spec.Catalog},
	}
	if spec.Playback != "" {
		service.recorded = []sessionReplayExchange{}
		file, err := os.Open(spec.Playback)
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		scanner := bufio.NewScanner(file)
		scanner.Buffer(make([]byte, 1<<20), 1<<28)
		for scanner.Scan() {
			var exchange sessionReplayExchange
			if err := json.Unmarshal(scanner.Bytes(), &exchange); err != nil {
				t.Fatal(err)
			}
			service.recorded = append(service.recorded, exchange)
		}
		if err := scanner.Err(); err != nil {
			t.Fatal(err)
		}
	}
	service.server = httptest.NewServer(http.HandlerFunc(service.serve))
	t.Cleanup(service.server.Close)
	return service
}

func (service *sessionReplayService) begin(call sessionReplayCall) {
	service.mu.Lock()
	defer service.mu.Unlock()
	service.current, service.captured = call, nil
}

func (service *sessionReplayService) end() []sessionReplayExchange {
	service.mu.Lock()
	defer service.mu.Unlock()
	captured := service.captured
	service.captured = nil
	if captured == nil {
		return []sessionReplayExchange{}
	}
	return captured
}

func (service *sessionReplayService) unmatchedCount() int {
	service.mu.Lock()
	defer service.mu.Unlock()
	return service.unmatched
}

func (service *sessionReplayService) unusedCount() int {
	service.mu.Lock()
	defer service.mu.Unlock()
	return len(service.recorded) - service.next
}

func (service *sessionReplayService) serve(writer http.ResponseWriter, request *http.Request) {
	if request.URL.Path != "/v1/rayline/arc/policy/decide" {
		service.listing.serve(writer, request)
		return
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	exchange := sessionReplayExchange{Request: string(body)}
	service.mu.Lock()
	if service.recorded != nil {
		// Playback is ordered: a request must be the recording's next one, so
		// a router that sends the same requests in another order diverges.
		if service.next < len(service.recorded) && service.recorded[service.next].Request == exchange.Request {
			exchange.Status, exchange.Response = service.recorded[service.next].Status, service.recorded[service.next].Response
			service.next++
		} else {
			exchange.Unmatched = true
			exchange.Status = http.StatusServiceUnavailable
			exchange.Response = `{"error":"backend_unavailable","detail":{}}`
			service.unmatched++
		}
	} else {
		exchange.Status, exchange.Response = service.oracle(body)
	}
	service.captured = append(service.captured, exchange)
	service.mu.Unlock()
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(exchange.Status)
	_, _ = writer.Write([]byte(exchange.Response))
}

// oracle answers with the action the script names for the current call, in
// the shape the strict decoder accepts. Hold service.mu.
func (service *sessionReplayService) oracle(body []byte) (int, string) {
	var decide raylinearc.PolicyDecisionRequest
	if err := json.Unmarshal(body, &decide); err != nil {
		return http.StatusBadRequest, `{"error":"invalid_request","detail":{}}`
	}
	if service.spec.OracleRefuseRelaxed != "" && decide.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed {
		refusal, _ := json.Marshal(map[string]any{
			"error": "unsupported_request", "detail": map[string]any{"reason": service.spec.OracleRefuseRelaxed},
		})
		return http.StatusUnprocessableEntity, string(refusal)
	}
	var response raylinearc.PolicyDecisionResponse
	switch {
	case service.current.Oracle != nil:
		response = service.listing.decision(decide, service.current.Oracle.SelectedActionID)
		response.Decision.SelectedArmID = service.current.Oracle.SelectedArmID
	case len(decide.Selection.AvailableActionIDs) > 0:
		// No client call is running: a readiness probe, answered with any
		// offered action.
		response = service.listing.decision(decide, decide.Selection.AvailableActionIDs[0])
	default:
		return http.StatusServiceUnavailable, `{"error":"backend_unavailable","detail":{}}`
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		service.t.Errorf("oracle: %v", err)
	}
	return http.StatusOK, string(encoded)
}

func readSessionReplayScript(t *testing.T, path string) []sessionReplayCall {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	var calls []sessionReplayCall
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1<<20), 1<<28)
	for scanner.Scan() {
		var call sessionReplayCall
		if err := json.Unmarshal(scanner.Bytes(), &call); err != nil {
			t.Fatalf("script line %d: %v", len(calls)+1, err)
		}
		calls = append(calls, call)
	}
	if err := scanner.Err(); err != nil {
		t.Fatal(err)
	}
	return calls
}

func readJSONFile(t *testing.T, path string, target any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
}
