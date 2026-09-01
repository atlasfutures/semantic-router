package routerruntime

import (
	"context"
	"testing"
)

type stubRouteDecisionRuntime struct {
	request RouteDecisionRequest
	result  RouteDecision
}

func (s *stubRouteDecisionRuntime) RouteDecision(
	_ context.Context,
	request RouteDecisionRequest,
) (RouteDecision, error) {
	s.request = request
	return s.result, nil
}

func TestRegistryRouteDecisionRuntimeStartsUnset(t *testing.T) {
	registry := NewRegistry(nil)
	if runtime := registry.RouteDecisionRuntime(); runtime != nil {
		t.Fatalf("RouteDecisionRuntime() = %v, want nil before the router publishes", runtime)
	}
	var nilRegistry *Registry
	if runtime := nilRegistry.RouteDecisionRuntime(); runtime != nil {
		t.Fatalf("nil Registry RouteDecisionRuntime() = %v, want nil", runtime)
	}
}

func TestRegistryPublishesRouteDecisionRuntime(t *testing.T) {
	registry := NewRegistry(nil)
	stub := &stubRouteDecisionRuntime{
		result: RouteDecision{SelectedWorker: "worker-b", WorkerModel: "model-b"},
	}
	registry.SetRouteDecisionRuntime(stub)

	runtime := registry.RouteDecisionRuntime()
	if runtime == nil {
		t.Fatal("RouteDecisionRuntime() = nil, want the published runtime")
	}
	decision, err := runtime.RouteDecision(
		context.Background(),
		RouteDecisionRequest{
			Body:          []byte(`{"messages":[]}`),
			DecisionID:    "rt_abc",
			SessionID:     "session-1",
			ExecutedModel: "vendor/model",
		},
	)
	if err != nil {
		t.Fatalf("RouteDecision() error = %v, want nil", err)
	}
	if decision.SelectedWorker != "worker-b" || decision.WorkerModel != "model-b" {
		t.Fatalf("RouteDecision() = %+v, want worker-b/model-b", decision)
	}
	if stub.request.DecisionID != "rt_abc" ||
		stub.request.SessionID != "session-1" ||
		stub.request.ExecutedModel != "vendor/model" {
		t.Fatalf("seam dropped request fields: %+v", stub.request)
	}
}
