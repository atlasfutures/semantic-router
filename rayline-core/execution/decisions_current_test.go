package execution

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

func currentDecisionFixture(t *testing.T, config Config, response map[string]any) *HTTPDecisions {
	t.Helper()
	t.Setenv("TEST_DECISIONS_KEY", "current-api-test-key")
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		var routing map[string]json.RawMessage
		if err := json.Unmarshal(body["routing"], &routing); err != nil {
			t.Error(err)
		}
		if _, exists := routing["execution"]; exists {
			t.Error("current API received unsupported local cache state")
		}
		var expected, actual map[string]any
		if err := json.Unmarshal(config.Decisions.Routing, &expected); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(body["routing"], &actual); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(expected, actual) {
			t.Error("current API routing preferences changed in transit")
		}
		writeJSON(w, 200, response)
	}))
	t.Cleanup(endpoint.Close)
	config.Decisions.URL = endpoint.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func TestCurrentSelectionRefusalLogsIdentityWithoutRequestMaterial(t *testing.T) {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, nil)))
	defer slog.SetDefault(previous)
	config := currentTestConfig(t)
	response := map[string]any{"route_id": "route-refused", "model": "unbound/model", "worker": "unbound-worker", "policy_action_id": strings.Repeat("e", 64)}
	input := currentTestInput()
	input.Body = []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"private task sentinel"}]}`)
	_, err := currentDecisionFixture(t, config, response).Decide(t.Context(), input)
	if err == nil {
		t.Fatal("unbound selection accepted")
	}
	var event map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(logs.Bytes()), &event); err != nil {
		t.Fatal(err)
	}
	if event["msg"] != "decision refused" || event["reason"] != "unbound_action" || event["route_id"] != "route-refused" || event["selected_model"] != "unbound/model" || event["selected_worker"] != "unbound-worker" || event["selected_action_id"] != strings.Repeat("e", 64) || event["selected_control_id"] != nil {
		t.Fatalf("refusal identity missing: %s", logs.Bytes())
	}
	if strings.Contains(logs.String(), "private task sentinel") || strings.Contains(logs.String(), "current-api-test-key") {
		t.Fatal("request material entered refusal log")
	}
}

func currentTestConfig(t *testing.T) Config {
	config := testConfig(t, "http://127.0.0.1:1/v1/chat/completions", false)
	config.Decisions.APIMode = DecisionsCurrent
	config.Decisions.OfferSHA256 = ""
	return config
}

func currentTestInput() DecisionInput {
	return DecisionInput{Body: []byte(`{"model":"arc-1.0","messages":[{"role":"user","content":"test"}]}`), Format: llmprotocol.OpenAIChatV1, State: ExecutionContext{HeldActionID: "local-held", Workers: []WorkerEvidence{{DeploymentID: "local"}}}}
}

func TestCurrentTopBindsManifestWithoutClaimingReturnedPins(t *testing.T) {
	config := currentTestConfig(t)
	action := config.Actions[0]
	response := map[string]any{"route_id": "route-current", "model": config.Deployments[0].RegistryModel, "worker": action.Worker, "policy_action_id": action.ID, "routing": map[string]any{"unapplied_keys": []string{}}}
	provider := currentDecisionFixture(t, config, response)
	decision, err := provider.Decide(t.Context(), currentTestInput())
	if err != nil {
		t.Fatal(err)
	}
	if decision.IdentitySource != "configured_manifest" || decision.Model != action.Model || decision.ReportedModel != config.Deployments[0].RegistryModel || decision.ControlID != action.ControlID || decision.AppliedValueUSD != nil {
		t.Fatalf("incorrect identity attribution: %+v", decision)
	}
}

func TestCurrentTopRejectsUnboundActionAndUnappliedRequests(t *testing.T) {
	for _, kind := range []string{"action", "model", "worker", "unapplied", "configured offer pin"} {
		t.Run(kind, func(t *testing.T) {
			config := currentTestConfig(t)
			action := config.Actions[0]
			response := map[string]any{"route_id": "route-current", "model": action.Model, "worker": action.Worker, "policy_action_id": action.ID}
			switch kind {
			case "action":
				response["policy_action_id"] = strings.Repeat("d", 64)
			case "model":
				response["model"] = "another-model"
			case "worker":
				response["worker"] = "another-worker"
			case "unapplied":
				response["routing"] = map[string]any{"unapplied_keys": []string{"value_usd"}}
			case "configured offer pin":
				config.Decisions.OfferSHA256 = strings.Repeat("b", 64)
			}
			if _, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput()); err == nil {
				t.Fatal("invalid selection accepted")
			}
		})
	}
}

func TestCurrentCustomRequiresAppliedValueAndExactIdentity(t *testing.T) {
	for _, kind := range []string{"valid", "no custom", "unapplied value", "changed value", "changed control", "missing price identity", "missing exclusive"} {
		t.Run(kind, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.Result = "custom"
			config.Decisions.OfferSHA256 = strings.Repeat("b", 64)
			config.Decisions.Routing = json.RawMessage(`{"value_usd":0.5}`)
			action := config.Actions[0]
			prices := map[string]any{"basis": "catalog", "price_sha256": nil, "exclusive": false}
			identity := map[string]any{"package_sha256": config.PackageSHA256, "thinking_controls_sha256": config.RegistrySHA256, "offer_sha256": config.Decisions.OfferSHA256, "action_id": action.ID, "policy_action_id": action.ID, "control_id": action.ControlID, "prices": prices}
			value := map[string]any{"status": "applied", "requested": 0.5}
			response := map[string]any{"route_id": "route-custom", "model": "wrong-top-result", "custom": map[string]any{"model": action.Model, "worker": action.Worker, "policy_action_id": action.ID, "identity": identity}, "routing": map[string]any{"value_usd": value, "unapplied_keys": []string{}}}
			switch kind {
			case "no custom":
				delete(response, "custom")
			case "unapplied value":
				value["status"] = "unsupported"
			case "changed value":
				value["requested"] = 1.0
			case "changed control":
				identity["control_id"] = "different"
			case "missing price identity":
				delete(identity, "prices")
			case "missing exclusive":
				delete(prices, "exclusive")
			}
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			if kind != "valid" {
				if err == nil {
					t.Fatal("unsupported custom selection accepted")
				}
				return
			}
			if err != nil || decision.Model != action.Model || decision.Result != "custom" || decision.IdentitySource != "response" || decision.AppliedValueUSD == nil || *decision.AppliedValueUSD != 0.5 {
				t.Fatalf("custom selection: %+v %v", decision, err)
			}
		})
	}
}

func TestCurrentAPIIsExplicitAndNeverDowngradesValue(t *testing.T) {
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	for _, kind := range []string{"unknown mode", "top with value", "custom without value", "unknown routing"} {
		t.Run(kind, func(t *testing.T) {
			config := currentTestConfig(t)
			config.Decisions.URL = "http://127.0.0.1:1/decisions"
			switch kind {
			case "unknown mode":
				config.Decisions.APIMode = "auto"
			case "top with value":
				config.Decisions.Routing = json.RawMessage(`{"value_usd":1}`)
			case "custom without value":
				config.Decisions.Result = "custom"
			case "unknown routing":
				config.Decisions.Routing = json.RawMessage(`{"response_fields":["predictions"]}`)
			}
			if _, err := NewHTTPDecisions(config); err == nil {
				t.Fatal("ambiguous or unsupported mode accepted")
			}
		})
	}
}
