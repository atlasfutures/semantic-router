package execution

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

// Verbatim public contract fixture from semantic-router PR #304, commit
// 0f6c78b8b30b61280eba2ca13f3fd84c1c355cd1, pkg/extproc/testdata/
// arc_public_contract/response_presets_custom.json. It has no invented identity.
func currentContractResponse(t *testing.T) map[string]any {
	t.Helper()
	raw, err := os.ReadFile("testdata/current_presets_custom.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Body map[string]any `json:"body"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	return fixture.Body
}

func currentContractConfig(t *testing.T, response map[string]any, result string) (Config, map[string]any) {
	t.Helper()
	config := currentTestConfig(t)
	config.Decisions.Result = result
	selected := response["custom"].(map[string]any)
	config.Decisions.Routing = json.RawMessage(`{"value_usd":2}`)
	if result != "custom" {
		selected = response["presets"].(map[string]any)[result].(map[string]any)
		config.Decisions.Routing = json.RawMessage(`{}`)
		delete(response, "routing")
		delete(response, "custom")
	}
	config.Actions[0].ID = selected["policy_action_id"].(string)
	config.Actions[0].Model = selected["model"].(string)
	config.Actions[0].Worker = selected["worker"].(string)
	return config, selected
}

func TestCurrentContractFixtureSelectsExactResult(t *testing.T) {
	for _, result := range []string{"custom", "swift", "sage"} {
		t.Run(result, func(t *testing.T) {
			response := currentContractResponse(t)
			config, selected := currentContractConfig(t, response, result)
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			if err != nil {
				t.Fatal(err)
			}
			if decision.ActionID != selected["policy_action_id"] || decision.Model != selected["model"] || decision.Result != result || decision.IdentitySource != "configured_manifest" || decision.PackageSHA256 != config.PackageSHA256 || decision.ControlID != config.Actions[0].ControlID || decision.AppliedValueUSD == nil || *decision.AppliedValueUSD != selected["value_usd"] {
				t.Fatalf("selected result did not retain its manifest and value provenance: %+v", decision)
			}
		})
	}
}

func TestCurrentContractDoesNotDowngradeAdvancedMode(t *testing.T) {
	response := currentContractResponse(t)
	config, _ := currentContractConfig(t, response, "custom")
	config.Decisions.APIMode = DecisionsExecutionContextV1
	config.Decisions.OfferSHA256 = strings.Repeat("b", 64)
	t.Setenv("TEST_DECISIONS_KEY", "test-key")
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, response)
	}))
	defer endpoint.Close()
	config.Decisions.URL = endpoint.URL
	provider, err := NewHTTPDecisions(config)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := provider.Decide(t.Context(), currentTestInput()); err == nil {
		t.Fatal("advanced mode accepted current envelope without its identity/evidence contract")
	}
}

func TestCurrentContractCustomRefusesWrongSelectedReceipt(t *testing.T) {
	for _, scenario := range []string{"action", "worker", "model", "route", "offer pin", "missing value", "value", "missing weight", "weight", "missing prices", "catalog digest", "caller basis", "missing routing receipt", "unapplied value", "echo value"} {
		t.Run(scenario, func(t *testing.T) {
			response := currentContractResponse(t)
			config, selected := currentContractConfig(t, response, "custom")
			routing := response["routing"].(map[string]any)
			switch scenario {
			case "action":
				selected["policy_action_id"] = strings.Repeat("e", 64)
			case "worker":
				selected["worker"] = "unbound"
			case "model":
				selected["model"] = "unbound"
			case "route":
				delete(response, "route_id")
			case "offer pin":
				config.Decisions.OfferSHA256 = strings.Repeat("b", 64)
			case "missing value":
				delete(selected, "value_usd")
			case "value":
				selected["value_usd"] = 0.1
			case "missing weight":
				delete(selected, "cost_weight")
			case "weight":
				selected["cost_weight"] = 10
			case "missing prices":
				delete(selected, "prices")
			case "catalog digest":
				selected["prices"].(map[string]any)["price_sha256"] = strings.Repeat("d", 64)
			case "caller basis":
				selected["prices"].(map[string]any)["basis"] = "caller"
			case "missing routing receipt":
				delete(routing, "unapplied_keys")
			case "unapplied value":
				routing["value_usd"].(map[string]any)["status"] = "unsupported"
			case "echo value":
				routing["value_usd"].(map[string]any)["requested"] = 0.1
			}
			if _, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput()); err == nil {
				t.Fatal("incorrect selected receipt accepted")
			}
		})
	}
}

func TestCurrentContractCustomProfilePricesBelongToSelectedResult(t *testing.T) {
	for _, scenario := range []string{"valid profile", "latest profile", "inline prices", "wrong result digest", "wrong echo digest", "wrong configured digest", "catalog result", "missing prices", "unsupported prices", "missing exclusive", "wrong exclusive", "missing profile", "wrong profile", "wrong version", "bad content hash"} {
		t.Run(scenario, func(t *testing.T) {
			response := currentContractResponse(t)
			config, selected := currentContractConfig(t, response, "custom")
			config.Decisions.Routing = json.RawMessage(`{"value_usd":2,"price_profile_id":"profile-fixture","price_profile_version":2,"exclusive":true}`)
			digest := strings.Repeat("d", 64)
			config.Decisions.PriceSHA256 = digest
			prices := map[string]any{"basis": "caller", "price_sha256": digest}
			selected["prices"] = prices // no per-result exclusive in the real contract
			routing := response["routing"].(map[string]any)
			echo := map[string]any{"status": "applied", "price_sha256": digest, "exclusive": true}
			profile := map[string]any{"status": "applied", "id": "profile-fixture", "version": 2, "content_hash": "sha256:" + strings.Repeat("e", 64)}
			routing["prices"], routing["price_profile"] = echo, profile
			switch scenario {
			case "latest profile":
				config.Decisions.Routing = json.RawMessage(`{"value_usd":2,"price_profile_id":"profile-fixture","exclusive":true}`)
				config.Decisions.PriceSHA256 = ""
			case "inline prices":
				config.Decisions.Routing = json.RawMessage(`{"value_usd":2,"prices":{"fixture":{"input":1,"cache_read":1,"cache_write":1,"output":1}},"exclusive":true}`)
				delete(routing, "price_profile")
			case "wrong result digest":
				prices["price_sha256"] = strings.Repeat("a", 64)
			case "wrong echo digest":
				echo["price_sha256"] = strings.Repeat("a", 64)
			case "wrong configured digest":
				config.Decisions.PriceSHA256 = strings.Repeat("a", 64)
			case "catalog result":
				prices["basis"], prices["price_sha256"] = "catalog", nil
			case "missing prices":
				delete(routing, "prices")
			case "unsupported prices":
				echo["status"] = "unsupported"
			case "missing exclusive":
				delete(echo, "exclusive")
			case "wrong exclusive":
				echo["exclusive"] = false
			case "missing profile":
				delete(routing, "price_profile")
			case "wrong profile":
				profile["id"] = "other"
			case "wrong version":
				profile["version"] = 1
			case "bad content hash":
				profile["content_hash"] = digest
			}
			decision, err := currentDecisionFixture(t, config, response).Decide(t.Context(), currentTestInput())
			valid := scenario == "valid profile" || scenario == "latest profile" || scenario == "inline prices"
			if !valid {
				if err == nil {
					t.Fatal("mismatched selected price world accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if decision.IdentitySource != "configured_manifest" || (scenario != "inline prices" && (decision.PriceProfile == nil || decision.PriceProfile.ContentHash != profile["content_hash"])) {
				t.Fatalf("profile and manifest provenance lost: %+v", decision)
			}
		})
	}
}

func TestCurrentCustomHTTPExecutionRequiresValidReceipt(t *testing.T) {
	for _, valid := range []bool{true, false} {
		t.Run(map[bool]string{true: "valid", false: "wrong price world"}[valid], func(t *testing.T) {
			var calls atomic.Int32
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				_, _ = io.WriteString(w, `{"id":"chat1","object":"chat.completion","created":1,"model":"qwen-fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ready"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":1,"total_tokens":21}}`)
			}))
			defer upstream.Close()
			config := currentTestConfig(t)
			config.Deployments[0].URL = upstream.URL
			config.Decisions.Result, config.Decisions.Routing = "custom", json.RawMessage(`{"value_usd":2}`)
			response := currentContractResponse(t)
			selected := response["custom"].(map[string]any)
			action := config.Actions[0]
			selected["policy_action_id"], selected["worker"], selected["model"] = action.ID, action.Worker, action.Model
			if !valid {
				selected["prices"].(map[string]any)["basis"] = "caller"
			}
			provider := currentDecisionFixture(t, config, response)
			registry, err := thinkingcontrol.Embedded()
			if err != nil {
				t.Fatal(err)
			}
			server, err := NewServer(config, registry, provider, testCoreToken)
			if err != nil {
				t.Fatal(err)
			}
			endpoint := httptest.NewServer(server)
			defer endpoint.Close()
			status, _ := executeHTTP(t, endpoint.URL, "/v1/chat/completions", `{"model":"arc-1.0","messages":[{"role":"user","content":"hello"}]}`, "current-custom")
			state, err := server.states.acquire(t.Context(), "current-custom", "main")
			if err != nil {
				t.Fatal(err)
			}
			defer state.release()
			if valid {
				if status != 200 || calls.Load() != 1 || state.completed != 1 || state.held != action.ID {
					t.Fatal("valid custom did not execute and commit exactly once")
				}
			} else if status == 200 || calls.Load() != 0 || state.completed != 0 || len(state.workers) != 0 {
				t.Fatal("refused custom reached inference or committed cache state")
			}
		})
	}
}
