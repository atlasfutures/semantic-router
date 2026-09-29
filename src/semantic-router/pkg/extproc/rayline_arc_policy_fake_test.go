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
	cold     atomic.Bool
	choose   func(raylinearc.PolicyDecisionRequest) string
	failWith string
	requests []raylinearc.PolicyDecisionRequest
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

func (fake *fakePolicyService) received() []raylinearc.PolicyDecisionRequest {
	fake.mu.Lock()
	defer fake.mu.Unlock()
	return append([]raylinearc.PolicyDecisionRequest(nil), fake.requests...)
}

func (fake *fakePolicyService) serve(writer http.ResponseWriter, request *http.Request) {
	switch request.URL.Path {
	case "/v1/rayline/arc/policy/packages":
		fake.writeJSON(writer, http.StatusOK, fake.packages())
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
		choose, failWith := fake.choose, fake.failWith
		fake.mu.Unlock()
		if failWith != "" {
			fake.writeJSON(writer, http.StatusServiceUnavailable, map[string]any{"error": failWith, "detail": map[string]any{}})
			return
		}
		fake.writeJSON(writer, http.StatusOK, fake.decision(decide, choose(decide)))
	default:
		http.NotFound(writer, request)
	}
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
