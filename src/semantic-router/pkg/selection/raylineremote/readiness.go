package raylineremote

import (
	"context"
	"math"
	"slices"
)

const pendingJournalMVP = "bounded_in_process_single_replica"

type capabilitiesWireResponse struct {
	SchemaVersion            string   `json:"schema_version"`
	TransactionSchemaVersion string   `json:"transaction_schema_version"`
	BundleVersion            string   `json:"bundle_version"`
	Protocols                []string `json:"protocols"`
	Operations               []string `json:"operations"`
	Workers                  []string `json:"workers"`
	LeaseSeconds             float64  `json:"lease_seconds"`
	PendingJournal           string   `json:"pending_journal"`
}

type workerPriceWire struct {
	Input      float64 `json:"input"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"`
	Output     float64 `json:"output"`
}

type workerWireResponse struct {
	ID                     string          `json:"id"`
	Backend                string          `json:"backend"`
	Model                  string          `json:"model"`
	ThinkingMode           string          `json:"thinking_mode"`
	OpenRouterFallbacks    bool            `json:"openrouter_allow_fallbacks"`
	CapabilityTags         []string        `json:"capability_tags"`
	PricingSnapshotVersion string          `json:"pricing_snapshot_version"`
	PerTokenPrices         workerPriceWire `json:"per_token_prices"`
}

// CheckReadiness validates the same authenticated policy and worker surfaces
// used by live selection. Liveness alone is never sufficient.
func (client *Client) CheckReadiness(ctx context.Context) error {
	if client == nil {
		return newFailure(
			FailureRequest,
			"readiness",
			"client_required",
			0,
		)
	}
	var capabilities capabilitiesWireResponse
	if err := client.get(
		ctx,
		"capabilities",
		"/v1/route/capabilities",
		&capabilities,
	); err != nil {
		return err
	}
	if err := client.validateCapabilities(capabilities); err != nil {
		return err
	}
	var workers []workerWireResponse
	if err := client.get(
		ctx,
		"workers",
		"/v1/workers",
		&workers,
	); err != nil {
		return err
	}
	return client.validateWorkers(workers)
}

func (client *Client) validateCapabilities(
	capabilities capabilitiesWireResponse,
) error {
	expectedWorkers := make([]string, 0, len(client.config.Workers))
	for _, worker := range client.config.Workers {
		expectedWorkers = append(expectedWorkers, worker.ID)
	}
	requiredOperations := []string{
		"prepare",
		"renew",
		"commit",
		"abort",
		"settle",
	}
	if capabilities.SchemaVersion != CapabilitiesSchemaVersion ||
		capabilities.TransactionSchemaVersion !=
			TransactionSchemaVersion ||
		capabilities.BundleVersion != client.config.BundleVersion ||
		!slices.Equal(capabilities.Protocols, []string{
			OpenAIChatProtocol,
		}) ||
		!sameStringSet(capabilities.Operations, requiredOperations) ||
		!containsUniqueStrings(
			capabilities.Workers,
			expectedWorkers,
		) ||
		capabilities.LeaseSeconds !=
			client.config.LeaseTTL.Seconds() ||
		capabilities.PendingJournal != pendingJournalMVP {
		return newFailure(
			FailureContract,
			"capabilities",
			"readiness_drift",
			0,
		)
	}
	return nil
}

func (client *Client) validateWorkers(
	workers []workerWireResponse,
) error {
	actualByID, ok := validatedWorkerCatalog(workers)
	if !ok {
		return workerCatalogDrift()
	}
	for _, expected := range client.config.Workers {
		actual, ok := actualByID[expected.ID]
		if !ok || !workerContractMatches(actual, expected) {
			return workerCatalogDrift()
		}
	}
	return nil
}

func validatedWorkerCatalog(
	workers []workerWireResponse,
) (map[string]workerWireResponse, bool) {
	actualByID := make(map[string]workerWireResponse, len(workers))
	for _, actual := range workers {
		if actual.ID == "" ||
			actual.Model == "" ||
			actual.Backend == "" ||
			!validThinkingMode(actual.ThinkingMode) ||
			!validWorkerPrices(actual.PerTokenPrices) ||
			actualByID[actual.ID].ID != "" {
			return nil, false
		}
		actualByID[actual.ID] = actual
	}
	return actualByID, true
}

func workerContractMatches(
	actual workerWireResponse,
	expected WorkerContract,
) bool {
	return actual.Model == expected.Model &&
		(expected.Backend == "" || actual.Backend == expected.Backend) &&
		(expected.ThinkingMode == "" ||
			actual.ThinkingMode == expected.ThinkingMode) &&
		actual.OpenRouterFallbacks == expected.AllowFallbacks &&
		(expected.Prices == nil ||
			workerPricesEqual(actual.PerTokenPrices, *expected.Prices))
}

func workerCatalogDrift() error {
	return newFailure(
		FailureContract,
		"workers",
		"catalog_drift",
		0,
	)
}

func validThinkingMode(mode string) bool {
	switch mode {
	case "disabled",
		"enabled",
		"minimal",
		"low",
		"medium",
		"high",
		"xhigh",
		"max",
		"provider_default",
		"legacy_unspecified",
		"implicit",
		"unknown",
		"auto",
		"none",
		"off",
		"no_thinking",
		"openrouter_none",
		"false",
		"0",
		"on":
		return true
	default:
		return false
	}
}

func containsUniqueStrings(
	actual []string,
	required []string,
) bool {
	if len(actual) < len(required) {
		return false
	}
	seen := make(map[string]bool, len(actual))
	for _, value := range actual {
		if value == "" || seen[value] {
			return false
		}
		seen[value] = true
	}
	for _, value := range required {
		if !seen[value] {
			return false
		}
	}
	return true
}

func workerPricesEqual(
	actual workerPriceWire,
	expected WorkerPrices,
) bool {
	return workerPriceEqual(actual.Input, expected.Input) &&
		workerPriceEqual(actual.CacheRead, expected.CacheRead) &&
		workerPriceEqual(actual.CacheWrite, expected.CacheWrite) &&
		workerPriceEqual(actual.Output, expected.Output)
}

func workerPriceEqual(actual float64, expected float64) bool {
	delta := math.Abs(actual - expected)
	tolerance := math.Max(
		1e-15,
		math.Abs(expected)*1e-9,
	)
	return delta <= tolerance
}

func sameStringSet(left []string, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	seen := make(map[string]bool, len(left))
	for _, value := range left {
		if value == "" || seen[value] {
			return false
		}
		seen[value] = true
	}
	for _, value := range right {
		if !seen[value] {
			return false
		}
	}
	return true
}

func validWorkerPrices(prices workerPriceWire) bool {
	for _, value := range []float64{
		prices.Input,
		prices.CacheRead,
		prices.CacheWrite,
		prices.Output,
	} {
		if math.IsNaN(value) ||
			math.IsInf(value, 0) ||
			value < 0 ||
			value > 1_000_000 {
			return false
		}
	}
	return true
}
