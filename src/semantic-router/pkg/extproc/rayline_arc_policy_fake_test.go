package extproc

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// fakePolicyService is a contract-faithful policy service: it serves the
// packages listing and answers each decide call with the action choose names,
// scoring every catalog action. Responses are built from the shared v1
// fixture, so they pass the router's strict decoder.
type fakePolicyService struct {
	t       *testing.T
	server  *httptest.Server
	alias   string
	sha256  string
	catalog []string

	mu sync.Mutex
	// answerAs, when set, is the package the decide responses name.
	answerAs *raylinearc.PolicyPackageRef
	// encodeUnreported answers timing_ms.encode null, as the reference
	// service does.
	encodeUnreported bool
	// cold lists no loaded package, as a service still loading its package.
	cold   atomic.Bool
	choose func(raylinearc.PolicyDecisionRequest) string
	// bodies are the decide request bodies as received, in order.
	bodies   [][]byte
	failWith string
	// failDetail is the detail failWith answers with ({} when nil).
	failDetail map[string]any
	// failFirst answers the next failFirstLeft decide calls with this code.
	failFirst     string
	failFirstLeft int
	// refuseCold, when set, answers 422 selection_refused /
	// stage_one_held_unknown for every decide it holds true of, as a
	// two-stage package does for a turn it would decide cold.
	refuseCold func(raylinearc.PolicyDecisionRequest) bool
	requests   []raylinearc.PolicyDecisionRequest
	// relaxedUnsupported, when set, answers every relaxed decide with 422
	// unsupported_request and this detail.reason, as a service whose
	// package cannot serve relaxed calls.
	relaxedUnsupported string
	// nullRevision answers every decide with session_revision null, as only
	// a relaxed call may.
	nullRevision bool
	// ignoreEpisodeMode answers a relaxed decide as a strict one, with a
	// session revision, as a service that predates episode_mode.
	ignoreEpisodeMode bool
	// verification, when set, is reported on the packages listing and on
	// every decide response, as a service from pathfinder#3120 on.
	verification raylinearc.PolicyVerification
	// barrier, when set, holds each decide call until that many are in
	// flight at once (or it times out), and inflight/maxInflight count them.
	barrier     int
	inflight    int
	maxInflight int
	arrived     chan struct{}
}

func newFakePolicyService(t *testing.T, alias, sha256 string, catalog []string) *fakePolicyService {
	t.Helper()
	fake := &fakePolicyService{t: t, alias: alias, sha256: sha256, catalog: catalog}
	fake.server = httptest.NewServer(http.HandlerFunc(fake.serve))
	t.Cleanup(fake.server.Close)
	return fake
}

func (fake *fakePolicyService) URL() string { return fake.server.URL }

func (fake *fakePolicyService) chooseWith(choose func(raylinearc.PolicyDecisionRequest) string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.choose = choose
}

// failNext makes every later decide call answer with this contract error.
func (fake *fakePolicyService) failNext(code string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.failWith = code
}

// failFirstCalls makes the next n decide calls answer with this contract
// error and later ones succeed.
func (fake *fakePolicyService) failFirstCalls(code string, n int) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.failFirst, fake.failFirstLeft = code, n
}

func (fake *fakePolicyService) received() []raylinearc.PolicyDecisionRequest {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]raylinearc.PolicyDecisionRequest(nil), fake.requests...)
}

// receivedBodies are the decide request bodies byte for byte, so a test can
// tell an omitted member from a null one.
func (fake *fakePolicyService) receivedBodies() [][]byte {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([][]byte(nil), fake.bodies...)
}

func (fake *fakePolicyService) serve(writer http.ResponseWriter, request *http.Request) {
	switch request.URL.Path {
	case "/v1/rayline/arc/policy/packages":
		// Like pathfinder's store-mode service, the package is listed only
		// for the pin that names it; a bare listing loads nothing.
		query := request.URL.Query()
		listing := fake.packages()
		if query.Get("alias") != fake.alias || query.Get("package_sha256") != fake.sha256 {
			listing.Packages = listing.Packages[:0]
		}
		fake.writeJSON(writer, http.StatusOK, listing)
	case "/v1/rayline/arc/policy/decide":
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		var decide raylinearc.PolicyDecisionRequest
		if err := json.Unmarshal(body, &decide); err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		fake.mu.Lock()
		fake.requests = append(fake.requests, decide)
		fake.bodies = append(fake.bodies, body)
		choose, failWith, failDetail, relaxedUnsupported := fake.choose, fake.failWith, fake.failDetail, fake.relaxedUnsupported
		refuseCold := fake.refuseCold
		if fake.failFirstLeft > 0 {
			fake.failFirstLeft--
			failWith = fake.failFirst
		}
		fake.mu.Unlock()
		if relaxedUnsupported != "" && decide.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed {
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
				"error": "unsupported_request", "detail": map[string]any{"reason": relaxedUnsupported},
			})
			return
		}
		if refuseCold != nil && refuseCold(decide) {
			fake.writeJSON(writer, http.StatusUnprocessableEntity, map[string]any{
				"error": "selection_refused", "detail": map[string]any{"reason": "stage_one_held_unknown"},
			})
			return
		}
		fake.awaitBarrier()
		if failWith != "" {
			if failDetail == nil {
				failDetail = map[string]any{}
			}
			fake.writeJSON(writer, http.StatusServiceUnavailable, map[string]any{"error": failWith, "detail": failDetail})
			return
		}
		fake.writeJSON(writer, http.StatusOK, fake.decision(decide, choose(decide)))
	default:
		http.NotFound(writer, request)
	}
}

// refuseRelaxed answers relaxed decide calls with this relaxed_* reason, or
// serves them again when reason is empty.
func (fake *fakePolicyService) refuseRelaxed(reason string) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.relaxedUnsupported = reason
}

// holdUntilConcurrent makes each later decide call wait until n are in flight.
func (fake *fakePolicyService) holdUntilConcurrent(n int) {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	fake.barrier, fake.arrived = n, make(chan struct{})
}

func (fake *fakePolicyService) peakConcurrency() int {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return fake.maxInflight
}

func (fake *fakePolicyService) awaitBarrier() {
	fake.mu.Lock()
	if fake.barrier == 0 {
		fake.mu.Unlock()
		return
	}
	fake.inflight++
	fake.maxInflight = max(fake.maxInflight, fake.inflight)
	arrived := fake.arrived
	if fake.inflight == fake.barrier {
		close(arrived)
	}
	fake.mu.Unlock()
	select {
	case <-arrived:
	case <-time.After(3 * time.Second):
	}
	fake.mu.Lock()
	fake.inflight--
	fake.mu.Unlock()
}

func (fake *fakePolicyService) packages() raylinearc.PolicyPackagesResponse {
	var listing raylinearc.PolicyPackagesResponse
	fake.readFixture("packages_response.v1.json", &listing)
	listing.Packages = listing.Packages[:1]
	if fake.cold.Load() {
		listing.Packages = listing.Packages[:0]
		return listing
	}
	listing.Packages[0].Alias, listing.Packages[0].PackageSHA256 = fake.alias, fake.sha256
	fake.mu.Lock()
	listing.Packages[0].Verification = fake.verification
	fake.mu.Unlock()
	return listing
}

func (fake *fakePolicyService) decision(request raylinearc.PolicyDecisionRequest, selected string) raylinearc.PolicyDecisionResponse {
	var response raylinearc.PolicyDecisionResponse
	fake.readFixture("decision_response.v1.json", &response)
	response.Package = raylinearc.PolicyPackageRef{Alias: fake.alias, PackageSHA256: fake.sha256}
	fake.mu.Lock()
	if fake.answerAs != nil {
		response.Package = *fake.answerAs
	}
	if fake.encodeUnreported {
		response.TimingMillis.Encode = nil
	}
	nullRevision, ignoreEpisodeMode := fake.nullRevision, fake.ignoreEpisodeMode
	response.PackageVerification = fake.verification
	fake.mu.Unlock()
	response.Shadow = []raylinearc.PolicyShadowResult{}
	available := make(map[string]bool, len(request.Selection.AvailableActionIDs))
	for _, actionID := range request.Selection.AvailableActionIDs {
		available[actionID] = true
	}
	response.Actions = response.Actions[:0]
	for index, actionID := range fake.catalog {
		response.Actions = append(response.Actions, raylinearc.PolicyActionScore{
			ActionID: actionID, Available: available[actionID], Supported: true,
			Selected: actionID == selected, Score: float64(index) / 10,
		})
	}
	response.Decision.SelectedActionID = selected
	response.Decision.SelectedArmID = "arm-" + selected[:8]
	// A relaxed call advances no session (pathfinder#3068).
	if (request.EpisodeMode == raylinearc.PolicyEpisodeModeRelaxed && !ignoreEpisodeMode) || nullRevision {
		response.Encoding.SessionRevision = nil
		response.Encoding.SessionAction = "rebuilt"
	}
	return response
}

func (fake *fakePolicyService) readFixture(name string, target any) {
	fake.t.Helper()
	raw, err := os.ReadFile("../selection/raylinearc/testdata/policy_service/" + name)
	if err != nil {
		fake.t.Fatal(err)
	}
	if err := json.Unmarshal(raw, target); err != nil {
		fake.t.Fatal(err)
	}
}

func (fake *fakePolicyService) writeJSON(writer http.ResponseWriter, status int, value any) {
	encoded, err := json.Marshal(value)
	if err != nil {
		fake.t.Errorf("fake policy service: %v", err)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_, _ = writer.Write(encoded)
}
