package execution

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestDecisionsCellSelectionRequiresMatchingGatewayReceipt(t *testing.T) {
	for _, echoed := range []string{"pf-hybrid-proof", "default", ""} {
		t.Run(echoed, func(t *testing.T) {
			t.Setenv("TEST_DECISIONS_KEY", "test-key")
			config := currentTestConfig(t)
			config.Decisions.CellTag = "pf-hybrid-proof"
			action := config.Actions[0]
			endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("X-Rayline-Cell") != config.Decisions.CellTag {
					t.Error("configured cell selection was not sent")
				}
				w.Header().Set("X-Rayline-Cell", echoed)
				writeJSON(w, http.StatusOK, map[string]any{
					"route_id": "route-cell", "model": config.Deployments[0].RegistryModel,
					"worker": action.Worker, "policy_action_id": action.ID,
					"routing": map[string]any{"unapplied_keys": []string{}},
				})
			}))
			defer endpoint.Close()
			config.Decisions.URL = endpoint.URL
			provider, err := NewHTTPDecisions(config)
			if err != nil {
				t.Fatal(err)
			}
			_, err = provider.Decide(t.Context(), currentTestInput())
			if echoed == config.Decisions.CellTag {
				if err != nil {
					t.Fatal(err)
				}
			} else if err == nil || !strings.Contains(err.Error(), "cell_tag") {
				t.Fatalf("missing or ignored cell selection was not refused: %v", err)
			}
		})
	}
}

func TestDecisionsCellSelectionRejectsInvalidHeaderValues(t *testing.T) {
	for _, tag := range []string{"UPPER", "two,tags", "tag\r\nInjected: value", " padded", strings.Repeat("a", 47)} {
		config := currentTestConfig(t)
		config.Decisions.CellTag = tag
		if _, err := NewHTTPDecisions(config); err == nil || !strings.Contains(err.Error(), "cell_tag") {
			t.Fatalf("invalid cell tag accepted: %q (%v)", tag, err)
		}
	}
}
