package raylinearc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"
)

// The caller price world refusals keep their bounded reason, and an
// unknown_price_model refusal the models it names (strings only), so the
// cell can answer each with its own code and error.models.
func TestPolicyServiceKeepsCallerPriceRefusalDetail(t *testing.T) {
	for _, tc := range []struct {
		code   string
		detail map[string]any
		class  string
		want   map[string]any
	}{
		{"unsupported_request", map[string]any{"reason": "caller_prices_unsupported", "note": "dropped"},
			"unsupported_request", map[string]any{"reason": "caller_prices_unsupported"}},
		{"unsupported_request", map[string]any{"reason": "caller_prices_inconsistent"},
			"unsupported_request", map[string]any{"reason": "caller_prices_inconsistent"}},
		{"unsupported_request", map[string]any{"reason": "something_else"}, "unsupported_request", nil},
		{"selection_refused", map[string]any{"reason": "unknown_price_model", "models": []any{"m/a", 7, "m/b"}},
			PolicySelectionRefusedClass, map[string]any{"reason": "unknown_price_model", "models": []any{"m/a", "m/b"}}},
		{"selection_refused", map[string]any{"reason": "unknown_price_model"},
			PolicySelectionRefusedClass, map[string]any{"reason": "unknown_price_model"}},
	} {
		server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			writer.WriteHeader(http.StatusUnprocessableEntity)
			_ = json.NewEncoder(writer).Encode(map[string]any{"error": tc.code, "detail": tc.detail})
		}))
		client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: time.Second})
		_, err := client.Decide(context.Background(), PolicyDecisionRequest{})
		server.Close()
		var failure *PolicyServiceError
		if !errors.As(err, &failure) || failure.Class != tc.class {
			t.Fatalf("%v: err = %v, want class %s", tc.detail, err, tc.class)
		}
		if !reflect.DeepEqual(failure.Detail, tc.want) {
			t.Fatalf("%v: detail = %#v, want %#v", tc.detail, failure.Detail, tc.want)
		}
	}
}
