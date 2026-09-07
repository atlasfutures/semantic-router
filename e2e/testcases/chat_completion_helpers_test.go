package testcases

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
)

// Under full duplex the router can end a downstream response itself, so an
// error envelope arrives with a 200 status. A case that asserts the status and
// never reads the body counts that turn as healthy. These are the bodies the
// check has to tell apart.
func TestAssertChatCompletionSucceeded(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name: "a completion",
			body: `{"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`,
		},
		{
			name: "a completion whose error field is explicitly null",
			body: `{"object":"chat.completion","error":null,"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}`,
		},
		{
			name:    "an error envelope served with 200",
			body:    `{"error":{"message":"upstream refused","type":"invalid_request_error"}}`,
			wantErr: "error envelope",
		},
		{
			// An error envelope that also carries choices is still an error.
			name:    "an error envelope beside choices",
			body:    `{"error":{"message":"cut"},"choices":[{"index":0,"message":{"role":"assistant"}}]}`,
			wantErr: "error envelope",
		},
		{
			name:    "a 200 with no choices at all",
			body:    `{"object":"chat.completion","choices":[]}`,
			wantErr: "no choices",
		},
		{
			name:    "a 200 that is not JSON",
			body:    `upstream connection reset`,
			wantErr: "not JSON",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := assertChatCompletionSucceeded([]byte(test.body), "probe")
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("a healthy completion was rejected: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("a failed turn passed as a healthy one: %s", test.body)
			}
			if !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %q, want it to name %q", err, test.wantErr)
			}
			if !strings.Contains(err.Error(), "probe") {
				t.Fatalf("error = %q, want it to name the subject", err)
			}
		})
	}
}

// A Response API error envelope decodes cleanly into the success struct and
// leaves every field zero, so a decode that returns no error proves nothing.
// That is what this check exists to catch.
func TestAssertResponseAPISucceeded(t *testing.T) {
	tests := []struct {
		name     string
		response *fixtures.ResponseAPIResponse
		wantErr  string
	}{
		{
			name: "a completed response",
			response: &fixtures.ResponseAPIResponse{
				Object: "response", Status: "completed",
				Output: []map[string]interface{}{{"type": "message"}},
			},
		},
		{
			// What an error envelope decodes to.
			name:     "an error envelope decoded into the success struct",
			response: &fixtures.ResponseAPIResponse{},
			wantErr:  "not a response object",
		},
		{
			name:     "a decode that produced nothing at all",
			response: nil,
			wantErr:  "not a response object",
		},
		{
			name:     "a response object carrying no output",
			response: &fixtures.ResponseAPIResponse{Object: "response", Status: "completed"},
			wantErr:  "no output",
		},
		{
			// A run that is still going is not a failure, and the pricing
			// telemetry it is asserted against is already written.
			name: "a response still in progress",
			response: &fixtures.ResponseAPIResponse{
				Object: "response", Status: "in_progress",
				Output: []map[string]interface{}{{"type": "message"}},
			},
		},
		{
			// The shape a failed run takes: the right object, a partial output
			// item, and a status that says the turn did not succeed. Checking
			// only object and output would pass this.
			name: "a failed run with a partial output item",
			response: &fixtures.ResponseAPIResponse{
				Object: "response", Status: "failed",
				Output: []map[string]interface{}{{"type": "message"}},
			},
			wantErr: "failed",
		},
		{
			name: "a run that stopped short",
			response: &fixtures.ResponseAPIResponse{
				Object: "response", Status: "incomplete",
				Output: []map[string]interface{}{{"type": "message"}},
			},
			wantErr: "incomplete",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := assertResponseAPISucceeded(test.response, []byte(`{"error":{"message":"x"}}`), "probe")
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("a completed response was rejected: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatal("a failed turn passed as a healthy one")
			}
			if !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("error = %q, want it to name %q", err, test.wantErr)
			}
			if !strings.Contains(err.Error(), "probe") {
				t.Fatalf("error = %q, want it to name the subject", err)
			}
		})
	}
}
