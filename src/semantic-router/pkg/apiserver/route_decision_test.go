//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

const routeDecisionTestBody = `{"model":"claude-sonnet","messages":[{"role":"user","content":"hi"}]}`

type fakeRouteDecisionRuntime struct {
	requests []routerruntime.RouteDecisionRequest
	decision routerruntime.RouteDecision
	err      error
}

func (f *fakeRouteDecisionRuntime) RouteDecision(
	_ context.Context,
	request routerruntime.RouteDecisionRequest,
) (routerruntime.RouteDecision, error) {
	f.requests = append(f.requests, request)
	if f.err != nil {
		return routerruntime.RouteDecision{}, f.err
	}
	return f.decision, nil
}

func routeDecisionTestServer(runtime routerruntime.RouteDecisionRuntime) *ClassificationAPIServer {
	registry := routerruntime.NewRegistry(&config.RouterConfig{})
	if runtime != nil {
		registry.SetRouteDecisionRuntime(runtime)
	}
	return &ClassificationAPIServer{
		config:          &config.RouterConfig{},
		runtimeRegistry: registry,
	}
}

func routeDecisionRequest(body string, headers map[string][]string) *http.Request {
	request := httptest.NewRequest(http.MethodPost, routeDecisionPath, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	for name, values := range headers {
		for _, value := range values {
			request.Header.Add(name, value)
		}
	}
	return request
}

func postRouteDecision(
	t *testing.T,
	server *ClassificationAPIServer,
	body string,
	headers map[string][]string,
) (*httptest.ResponseRecorder, map[string]interface{}) {
	t.Helper()
	recorder := httptest.NewRecorder()
	server.handleRouteDecision(recorder, routeDecisionRequest(body, headers))
	decoded := map[string]interface{}{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("response body is not a JSON object: %v (body=%q)", err, recorder.Body.String())
	}
	return recorder, decoded
}

func okRouteDecisionRuntime() *fakeRouteDecisionRuntime {
	return &fakeRouteDecisionRuntime{
		decision: routerruntime.RouteDecision{
			SelectedWorker: "worker-b",
			WorkerModel:    "vendor/model-b",
			Provider:       "vendor-slug",
		},
	}
}

// The caller discards the whole 200 when either required field is missing or
// is not a string, so both presence and type are part of the contract.
func TestRouteDecisionEmitsRequiredSelectionFields(t *testing.T) {
	server := routeDecisionTestServer(okRouteDecisionRuntime())

	recorder, decoded := postRouteDecision(t, server, routeDecisionTestBody, map[string][]string{
		routeIDHeader: {"rt_0a1b2c3d"},
	})

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", recorder.Code, recorder.Body.String())
	}
	for field, want := range map[string]string{
		"selected_worker": "worker-b",
		"worker_model":    "vendor/model-b",
		"provider":        "vendor-slug",
	} {
		value, ok := decoded[field].(string)
		if !ok {
			t.Fatalf("%s = %#v, want a string", field, decoded[field])
		}
		if value != want {
			t.Fatalf("%s = %q, want %q", field, value, want)
		}
	}
	if _, ok := decoded["decision_latency_ms"].(float64); !ok {
		t.Fatalf("decision_latency_ms = %#v, want a number", decoded["decision_latency_ms"])
	}
}

func TestRouteDecisionEchoesCallerRouteID(t *testing.T) {
	server := routeDecisionTestServer(okRouteDecisionRuntime())

	_, decoded := postRouteDecision(t, server, routeDecisionTestBody, map[string][]string{
		routeIDHeader: {"rt_ml_0a1b-2c3d"},
	})

	if decoded["decision_id"] != "rt_ml_0a1b-2c3d" {
		t.Fatalf("decision_id = %#v, want the echoed caller route id", decoded["decision_id"])
	}
}

func TestRouteDecisionMintsDecisionIDWhenHeaderAbsent(t *testing.T) {
	runtime := okRouteDecisionRuntime()
	server := routeDecisionTestServer(runtime)

	recorder, decoded := postRouteDecision(t, server, routeDecisionTestBody, nil)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", recorder.Code, recorder.Body.String())
	}
	minted, ok := decoded["decision_id"].(string)
	if !ok || minted == "" {
		t.Fatalf("decision_id = %#v, want a nonempty minted string", decoded["decision_id"])
	}
	if runtime.requests[0].DecisionID != minted {
		t.Fatalf("runtime saw decision id %q, response echoed %q", runtime.requests[0].DecisionID, minted)
	}
}

// Un-sourced optional fields must be absent, never zero-valued: a caller
// joining these rows offline cannot tell an invented value from a real one.
func TestRouteDecisionOmitsUnsourcedOptionalFields(t *testing.T) {
	server := routeDecisionTestServer(&fakeRouteDecisionRuntime{
		decision: routerruntime.RouteDecision{
			SelectedWorker: "worker-a",
			WorkerModel:    "vendor/model-a",
		},
	})

	_, decoded := postRouteDecision(t, server, routeDecisionTestBody, nil)

	if _, present := decoded["provider"]; present {
		t.Fatalf("provider present as %#v, want the key omitted when unsourced", decoded["provider"])
	}
	for _, invented := range []string{
		"bundle_version",
		"pricing_snapshot_version",
		"catalog_provider",
		"policy",
		"route_call_index",
	} {
		if _, present := decoded[invented]; present {
			t.Fatalf("response invented %q = %#v; this router has no source for it", invented, decoded[invented])
		}
	}
}

func TestRouteDecisionRejectsMalformedRequests(t *testing.T) {
	cases := []struct {
		name    string
		body    string
		headers map[string][]string
	}{
		{name: "body is not JSON", body: "not json"},
		{name: "body is not an object", body: `[{"role":"user"}]`},
		{name: "messages missing", body: `{"model":"m"}`},
		{name: "messages empty", body: `{"messages":[]}`},
		{name: "messages not a list", body: `{"messages":"hi"}`},
		{name: "message member not an object", body: `{"messages":["hi"]}`},
		{name: "model not a string", body: `{"messages":[{"role":"user"}],"model":7}`},
		{
			name:    "duplicate route id",
			body:    routeDecisionTestBody,
			headers: map[string][]string{routeIDHeader: {"rt_0a1b", "rt_0a1b"}},
		},
		{
			name:    "malformed route id",
			body:    routeDecisionTestBody,
			headers: map[string][]string{routeIDHeader: {"route-42"}},
		},
		{
			name:    "route id too long",
			body:    routeDecisionTestBody,
			headers: map[string][]string{routeIDHeader: {"rt_" + strings.Repeat("a", 41)}},
		},
		{
			name:    "duplicate executed model",
			body:    routeDecisionTestBody,
			headers: map[string][]string{executedModelHeader: {"vendor/a", "vendor/b"}},
		},
		{
			name:    "malformed executed model",
			body:    routeDecisionTestBody,
			headers: map[string][]string{executedModelHeader: {"/leading-slash"}},
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			runtime := okRouteDecisionRuntime()
			server := routeDecisionTestServer(runtime)

			recorder, decoded := postRouteDecision(t, server, testCase.body, testCase.headers)

			if recorder.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400 (body=%s)", recorder.Code, recorder.Body.String())
			}
			if detail, ok := decoded["detail"].(string); !ok || detail == "" {
				t.Fatalf("detail = %#v, want a nonempty explanation", decoded["detail"])
			}
			if len(runtime.requests) != 0 {
				t.Fatalf("rejected request still reserved episode state: %+v", runtime.requests)
			}
		})
	}
}

// Every failure this endpoint reports uses one shape, including the ones the
// generic body reader raises. The caller reads them against one contract.
func TestRouteDecisionReportsOversizedBodiesInItsOwnErrorShape(t *testing.T) {
	runtime := okRouteDecisionRuntime()
	server := routeDecisionTestServer(runtime)
	mux := server.setupRoutes()
	oversized := `{"messages":[{"role":"user","content":"` +
		strings.Repeat("x", int(routeDecisionBodyLimit)) + `"}]}`

	request := routeDecisionRequest(oversized, nil)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)
	decoded := map[string]interface{}{}
	if err := json.Unmarshal(recorder.Body.Bytes(), &decoded); err != nil {
		t.Fatalf("response body is not a JSON object: %v (body=%q)", err, recorder.Body.String())
	}

	if recorder.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", recorder.Code)
	}
	if detail, ok := decoded["detail"].(string); !ok || detail == "" {
		t.Fatalf("body = %#v, want this endpoint's detail shape", decoded)
	}
	if _, present := decoded["error"]; present {
		t.Fatalf("body used the management envelope: %#v", decoded)
	}
	if len(runtime.requests) != 0 {
		t.Fatalf("oversized request still reached the runtime: %+v", runtime.requests)
	}
}

func TestRouteDecisionAcceptsConversationAboveFormerFourMiBLimit(t *testing.T) {
	runtime := okRouteDecisionRuntime()
	server := routeDecisionTestServer(runtime)
	mux := server.setupRoutes()
	body := `{"messages":[{"role":"user","content":"` +
		strings.Repeat("x", 5_500_000) + `"}]}`

	request := routeDecisionRequest(body, nil)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 for a 5.5 MB conversation (body=%s)",
			recorder.Code, recorder.Body.String())
	}
	if len(runtime.requests) != 1 {
		t.Fatalf("runtime saw %d requests, want 1", len(runtime.requests))
	}
}

func TestRouteDecisionAcceptsWellFormedRouteIDs(t *testing.T) {
	for _, routeID := range []string{"rt_0a1b2c3d", "rt_ml_0a1b2c3d", "rt_0A1B-2C3D", "rt_a"} {
		t.Run(routeID, func(t *testing.T) {
			server := routeDecisionTestServer(okRouteDecisionRuntime())

			recorder, _ := postRouteDecision(t, server, routeDecisionTestBody, map[string][]string{
				routeIDHeader: {routeID},
			})

			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, want 200 for route id %q (body=%s)",
					recorder.Code, routeID, recorder.Body.String())
			}
		})
	}
}

// Negative control for the record-only rule. The executed model must reach the
// runtime as its own labelled field and must change nothing else about the
// consult: if the adapter ever folded it into the routing inputs, the two
// requests below would stop being identical and this assertion would fail.
func TestRouteDecisionKeepsExecutedModelRecordOnly(t *testing.T) {
	runtime := okRouteDecisionRuntime()
	server := routeDecisionTestServer(runtime)
	headers := map[string][]string{
		routeIDHeader:      {"rt_0a1b2c3d"},
		routeSessionHeader: {"session-7"},
	}

	withoutExecuted, decodedWithout := postRouteDecision(t, server, routeDecisionTestBody, headers)
	headers[executedModelHeader] = []string{"vendor/model-a"}
	withExecuted, decodedWith := postRouteDecision(t, server, routeDecisionTestBody, headers)

	if withoutExecuted.Code != http.StatusOK || withExecuted.Code != http.StatusOK {
		t.Fatalf("statuses = %d and %d, want 200 and 200", withoutExecuted.Code, withExecuted.Code)
	}
	if len(runtime.requests) != 2 {
		t.Fatalf("runtime saw %d consults, want 2", len(runtime.requests))
	}
	first, second := runtime.requests[0], runtime.requests[1]
	if first.ExecutedModel != "" {
		t.Fatalf("ExecutedModel = %q with no header, want empty", first.ExecutedModel)
	}
	if second.ExecutedModel != "vendor/model-a" {
		t.Fatalf("ExecutedModel = %q, want the reported model recorded", second.ExecutedModel)
	}
	if string(first.Body) != string(second.Body) || first.SessionID != second.SessionID {
		t.Fatalf("executed model changed the routing inputs: %+v vs %+v", first, second)
	}
	if decodedWithout["selected_worker"] != decodedWith["selected_worker"] {
		t.Fatalf("executed model changed the selection: %#v vs %#v",
			decodedWithout["selected_worker"], decodedWith["selected_worker"])
	}
}

func TestRouteDecisionForwardsBodyAndSessionUnmutated(t *testing.T) {
	runtime := okRouteDecisionRuntime()
	server := routeDecisionTestServer(runtime)

	postRouteDecision(t, server, routeDecisionTestBody, map[string][]string{
		routeSessionHeader: {"session-7"},
	})

	if string(runtime.requests[0].Body) != routeDecisionTestBody {
		t.Fatalf("Body = %q, want the client payload unmutated", runtime.requests[0].Body)
	}
	if runtime.requests[0].SessionID != "session-7" {
		t.Fatalf("SessionID = %q, want session-7", runtime.requests[0].SessionID)
	}
}

// Fail-closed: the algorithm has no fallback worker, so a failed consult must
// never be answered with a 200 carrying a default.
func TestRouteDecisionFailsClosedOnSelectionFailure(t *testing.T) {
	server := routeDecisionTestServer(&fakeRouteDecisionRuntime{err: errors.New("selector unavailable")})

	recorder, decoded := postRouteDecision(t, server, routeDecisionTestBody, nil)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 (body=%s)", recorder.Code, recorder.Body.String())
	}
	if _, present := decoded["selected_worker"]; present {
		t.Fatalf("failed consult still published a worker: %#v", decoded["selected_worker"])
	}
}

// Contention is not unavailability. A 503 sends the caller into fallback and
// reads as an outage; a contended consult is a healthy router that is already
// busy with this session, so it must answer 429 and say when to come back.
func TestRouteDecisionReportsContentionAsBackPressure(t *testing.T) {
	for name, failure := range map[string]string{
		"lease wait timed out": "episode_timeout",
		"store at capacity":    "episode_capacity",
	} {
		t.Run(name, func(t *testing.T) {
			server := routeDecisionTestServer(&fakeRouteDecisionRuntime{
				err: fmt.Errorf("%w: could not prepare the episode: %s",
					routerruntime.ErrRouteDecisionContended, failure),
			})

			recorder, decoded := postRouteDecision(t, server, routeDecisionTestBody, nil)

			if recorder.Code != http.StatusTooManyRequests {
				t.Fatalf("status = %d, want 429 (body=%s)", recorder.Code, recorder.Body.String())
			}
			if retryAfter := recorder.Header().Get("Retry-After"); retryAfter == "" {
				t.Fatal("contended consult carried no Retry-After")
			}
			if _, present := decoded["selected_worker"]; present {
				t.Fatalf("contended consult still published a worker: %#v", decoded["selected_worker"])
			}
		})
	}
}

// Every other failure keeps its 503: the caller cannot fix it by waiting, so
// inviting a retry would turn one broken consult into a retry storm.
func TestRouteDecisionKeepsUnavailableForNonContendedFailures(t *testing.T) {
	server := routeDecisionTestServer(&fakeRouteDecisionRuntime{
		err: errors.New("decision-only routing could not prepare the episode: episode_store"),
	})

	recorder, _ := postRouteDecision(t, server, routeDecisionTestBody, nil)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 (body=%s)", recorder.Code, recorder.Body.String())
	}
	if retryAfter := recorder.Header().Get("Retry-After"); retryAfter != "" {
		t.Fatalf("non-contended failure invited a retry: Retry-After=%q", retryAfter)
	}
}

func TestRouteDecisionFailsClosedOnIncompleteDecision(t *testing.T) {
	for name, decision := range map[string]routerruntime.RouteDecision{
		"no worker": {WorkerModel: "vendor/model-a"},
		"no model":  {SelectedWorker: "worker-a"},
	} {
		t.Run(name, func(t *testing.T) {
			server := routeDecisionTestServer(&fakeRouteDecisionRuntime{decision: decision})

			recorder, _ := postRouteDecision(t, server, routeDecisionTestBody, nil)

			if recorder.Code != http.StatusServiceUnavailable {
				t.Fatalf("status = %d, want 503 for a partial decision (body=%s)",
					recorder.Code, recorder.Body.String())
			}
		})
	}
}

// The management listener answers before the router finishes starting, so an
// unpublished runtime is a normal startup state and must fail closed, not panic.
func TestRouteDecisionFailsClosedBeforeRouterPublishes(t *testing.T) {
	server := routeDecisionTestServer(nil)

	recorder, _ := postRouteDecision(t, server, routeDecisionTestBody, nil)

	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 (body=%s)", recorder.Code, recorder.Body.String())
	}
}

func TestRouteDecisionIsServedAtTheLiteralClientPath(t *testing.T) {
	server := routeDecisionTestServer(okRouteDecisionRuntime())
	server.config = &config.RouterConfig{}
	mux := server.setupRoutes()

	request := routeDecisionRequest(routeDecisionTestBody, nil)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("POST %s through the mux = %d, want 200 (body=%s)",
			routeDecisionPath, recorder.Code, recorder.Body.String())
	}
}

// Default-deny. The caller is a trusted proxy, not a console user, so no
// built-in role reaches this route: a bearer deployment must grant
// route.decision on purpose.
func TestRouteDecisionDeniesBuiltInRolesUnderBearerAuth(t *testing.T) {
	for _, role := range []string{"viewer", "operator"} {
		t.Run(role, func(t *testing.T) {
			const token = "route-decision-token"
			t.Setenv("VSR_ROUTE_DECISION_TOKEN", token)
			server := routeDecisionTestServer(okRouteDecisionRuntime())
			server.config = &config.RouterConfig{
				ManagementAPI: config.ManagementAPIConfig{
					Auth: config.ManagementAPIAuthConfig{
						Mode:   config.ManagementAuthModeBearer,
						Tokens: []config.ManagementAPITokenRef{{Env: "VSR_ROUTE_DECISION_TOKEN", Role: role}},
						Roles:  config.DefaultManagementAPIRoles(),
					},
				},
			}
			mux := server.setupRoutes()

			request := routeDecisionRequest(routeDecisionTestBody, nil)
			request.Header.Set("Authorization", "Bearer "+token)
			recorder := httptest.NewRecorder()
			mux.ServeHTTP(recorder, request)

			if recorder.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want 403 for role %q", recorder.Code, role)
			}
		})
	}
}

func TestRouteDecisionAllowsAGrantedRoleUnderBearerAuth(t *testing.T) {
	const token = "route-decision-token"
	t.Setenv("VSR_ROUTE_DECISION_TOKEN", token)
	roles := config.DefaultManagementAPIRoles()
	roles["router-proxy"] = []string{string(PermRouteDecision)}
	server := routeDecisionTestServer(okRouteDecisionRuntime())
	server.config = &config.RouterConfig{
		ManagementAPI: config.ManagementAPIConfig{
			Auth: config.ManagementAPIAuthConfig{
				Mode:   config.ManagementAuthModeBearer,
				Tokens: []config.ManagementAPITokenRef{{Env: "VSR_ROUTE_DECISION_TOKEN", Role: "router-proxy"}},
				Roles:  roles,
			},
		},
	}
	mux := server.setupRoutes()

	request := routeDecisionRequest(routeDecisionTestBody, nil)
	request.Header.Set("Authorization", "Bearer "+token)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%s)", recorder.Code, recorder.Body.String())
	}
}

func TestRouteDecisionRouteCarriesItsOwnPermission(t *testing.T) {
	for _, route := range apiRoutes() {
		if route.Path != routeDecisionPath {
			continue
		}
		if route.Permission != PermRouteDecision {
			t.Fatalf("permission = %q, want %q", route.Permission, PermRouteDecision)
		}
		if route.Method != http.MethodPost {
			t.Fatalf("method = %q, want POST", route.Method)
		}
		return
	}
	t.Fatalf("no route registered at %s", routeDecisionPath)
}
