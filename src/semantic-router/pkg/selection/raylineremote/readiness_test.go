package raylineremote

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

type readinessDriftCase struct {
	name               string
	mutateCapabilities func(map[string]any)
	mutateWorkers      func([]map[string]any)
}

var readinessDriftCases = []readinessDriftCase{
	{
		name: "transaction version",
		mutateCapabilities: func(value map[string]any) {
			value["transaction_schema_version"] = "v2"
		},
	},
	{
		name: "bundle",
		mutateCapabilities: func(value map[string]any) {
			value["bundle_version"] = "latest"
		},
	},
	{
		name: "operation",
		mutateCapabilities: func(value map[string]any) {
			value["operations"] = []string{"prepare", "commit"}
		},
	},
	{
		name: "missing configured worker",
		mutateCapabilities: func(value map[string]any) {
			value["workers"] = []string{"worker-a", "worker-c"}
		},
	},
	{
		name: "lease",
		mutateCapabilities: func(value map[string]any) {
			value["lease_seconds"] = 31
		},
	},
	{
		name: "journal",
		mutateCapabilities: func(value map[string]any) {
			value["pending_journal"] = "durable-unknown"
		},
	},
	{
		name: "provider model",
		mutateWorkers: func(workers []map[string]any) {
			workers[1]["model"] = "wrong-model"
		},
	},
	{
		name: "provider backend",
		mutateWorkers: func(workers []map[string]any) {
			workers[1]["backend"] = "openai"
		},
	},
	{
		name: "thinking mode",
		mutateWorkers: func(workers []map[string]any) {
			workers[1]["thinking_mode"] = "off"
		},
	},
	{
		name: "provider fallback",
		mutateWorkers: func(workers []map[string]any) {
			workers[1]["openrouter_allow_fallbacks"] = true
		},
	},
	{
		name: "price drift",
		mutateWorkers: func(workers []map[string]any) {
			prices := workers[1]["per_token_prices"].(map[string]any)
			prices["output"] = 0.000003
		},
	},
	{
		name: "negative price",
		mutateWorkers: func(workers []map[string]any) {
			prices := workers[1]["per_token_prices"].(map[string]any)
			prices["output"] = -1
		},
	},
}

func TestReadinessRejectsCapabilityAndWorkerDrift(t *testing.T) {
	for _, test := range readinessDriftCases {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(readinessDriftHandler(t, test))
			defer server.Close()
			client, err := NewClient(validClientConfig(server.URL))
			if err != nil {
				t.Fatal(err)
			}
			err = client.CheckReadiness(context.Background())
			if !IsFailureClass(err, FailureContract) {
				t.Fatalf("error = %v, want contract failure", err)
			}
		})
	}
}

func readinessDriftHandler(
	t *testing.T,
	test readinessDriftCase,
) http.Handler {
	t.Helper()
	return http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		switch request.URL.Path {
		case "/v1/route/capabilities":
			capabilities := validCapabilities()
			if test.mutateCapabilities != nil {
				test.mutateCapabilities(capabilities)
			}
			writeJSON(t, writer, capabilities)
		case "/v1/workers":
			workers := []map[string]any{
				validWorker("worker-a", "provider-model-a"),
				validWorker("worker-b", "provider-model-b"),
			}
			if test.mutateWorkers != nil {
				test.mutateWorkers(workers)
			}
			writeJSON(t, writer, workers)
		default:
			writer.WriteHeader(http.StatusNotFound)
		}
	})
}

func TestReadinessAllowsCatalogWorkersOutsideDecisionMapping(
	t *testing.T,
) {
	server := httptest.NewServer(http.HandlerFunc(func(
		writer http.ResponseWriter,
		request *http.Request,
	) {
		switch request.URL.Path {
		case "/v1/route/capabilities":
			capabilities := validCapabilities()
			capabilities["workers"] = []string{
				"worker-c",
				"worker-b",
				"worker-a",
			}
			writeJSON(t, writer, capabilities)
		case "/v1/workers":
			writeJSON(t, writer, []map[string]any{
				validWorker("worker-c", "provider-model-c"),
				validWorker("worker-b", "provider-model-b"),
				validWorker("worker-a", "provider-model-a"),
			})
		default:
			writer.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	client, err := NewClient(validClientConfig(server.URL))
	if err != nil {
		t.Fatal(err)
	}
	if err := client.CheckReadiness(context.Background()); err != nil {
		t.Fatalf("extra non-dispatchable catalog worker rejected: %v", err)
	}
}

func validCapabilities() map[string]any {
	return map[string]any{
		"schema_version":             CapabilitiesSchemaVersion,
		"transaction_schema_version": TransactionSchemaVersion,
		"bundle_version":             "bundle-test-v1",
		"protocols":                  []string{OpenAIChatProtocol},
		"operations": []string{
			"prepare",
			"renew",
			"commit",
			"abort",
			"settle",
		},
		"workers":         []string{"worker-a", "worker-b"},
		"lease_seconds":   30,
		"pending_journal": pendingJournalMVP,
	}
}
