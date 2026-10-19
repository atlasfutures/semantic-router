package raylinearc

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"
)

// The capabilities listing (pathfinder#5454) decodes strictly: the top-level
// capabilities, a package's bounds, and a package that declares none (null).
func TestDecodePolicyPackagesResponseWithCapabilities(t *testing.T) {
	response, err := DecodePolicyPackagesResponse(readPolicyFixture(t, "packages_response_capabilities.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(response.Capabilities, []string{"decisions_v2", "tier_points", "response_fields"}) {
		t.Fatalf("capabilities = %v", response.Capabilities)
	}
	if len(response.Packages) != 2 {
		t.Fatalf("%d packages, want 2", len(response.Packages))
	}
	adjustable, fixed := response.Packages[0].AdjustableCostWeight, response.Packages[1].AdjustableCostWeight
	if !adjustable.Stated || adjustable.Bounds == nil || *adjustable.Bounds != (PolicyCostWeightBounds{Min: 0.2, Max: 1.0}) {
		t.Fatalf("adjustable package = %+v", adjustable)
	}
	if !fixed.Stated || fixed.Bounds != nil {
		t.Fatalf("a null adjustable_cost_weight decoded %+v", fixed)
	}
}

// An older service's listing states neither field: no capabilities, and no
// package states bounds.
func TestDecodePolicyPackagesResponseFromAnOlderService(t *testing.T) {
	response, err := DecodePolicyPackagesResponse(readPolicyFixture(t, "packages_response.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if response.Capabilities != nil {
		t.Fatalf("capabilities = %v, want none", response.Capabilities)
	}
	for _, loaded := range response.Packages {
		if loaded.AdjustableCostWeight.Stated {
			t.Fatalf("%s states adjustable_cost_weight", loaded.Alias)
		}
	}
}

func TestDecodePolicyPackagesResponseRefusesMalformedCapabilities(t *testing.T) {
	for name, edit := range map[string]func(body map[string]any){
		"unknown bound member": func(body map[string]any) {
			entry(body)["adjustable_cost_weight"] = map[string]any{"min": 0.2, "max": 1.0, "step": 0.1}
		},
		"missing max":   func(body map[string]any) { entry(body)["adjustable_cost_weight"] = map[string]any{"min": 0.2} },
		"min above max": func(body map[string]any) { entry(body)["adjustable_cost_weight"] = map[string]any{"min": 2, "max": 1} },
		"string bound": func(body map[string]any) {
			entry(body)["adjustable_cost_weight"] = map[string]any{"min": "0", "max": 1}
		},
		"bounds as a list":    func(body map[string]any) { entry(body)["adjustable_cost_weight"] = []any{0.2, 1.0} },
		"capability a number": func(body map[string]any) { body["capabilities"] = []any{7} },
		"unknown neighbour":   func(body map[string]any) { body["capability_detail"] = map[string]any{} },
	} {
		t.Run(name, func(t *testing.T) {
			body := fixtureWith(t, "packages_response_capabilities.v1.json", edit)
			if _, err := DecodePolicyPackagesResponse(body); err == nil {
				t.Fatal("decoded")
			}
		})
	}
}

func entry(body map[string]any) map[string]any {
	return body["packages"].([]any)[0].(map[string]any)
}

func TestCheckPolicyServiceRequirements(t *testing.T) {
	listing, err := DecodePolicyPackagesResponse(readPolicyFixture(t, "packages_response_capabilities.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	older, err := DecodePolicyPackagesResponse(readPolicyFixture(t, "packages_response.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	adjustable, fixed := &listing.Packages[0], &listing.Packages[1]
	for name, check := range map[string]struct {
		listing  *PolicyPackagesResponse
		loaded   *PolicyLoadedPackage
		required PolicyServiceRequirements
		class    string
	}{
		"nothing required, older service": {older, &older.Packages[0], PolicyServiceRequirements{}, ""},
		"nothing required, new service":   {listing, fixed, PolicyServiceRequirements{}, ""},
		"advertised capabilities": {listing, adjustable, PolicyServiceRequirements{
			Capabilities: []string{"decisions_v2", "tier_points", "response_fields"},
		}, ""},
		"exploration not advertised": {listing, adjustable, PolicyServiceRequirements{
			Capabilities: []string{"decisions_v2", "exploration"},
		}, PolicyCapabilityUnsupportedClass},
		"older service, decisions_v2": {older, &older.Packages[0], PolicyServiceRequirements{
			Capabilities: []string{"decisions_v2"},
		}, PolicyCapabilityUnsupportedClass},
		"weights inside the bounds, edges included": {listing, adjustable, PolicyServiceRequirements{
			CostWeights: []float64{0.2, 0.5, 1.0},
		}, ""},
		"weight below the bounds": {listing, adjustable, PolicyServiceRequirements{
			CostWeights: []float64{0.5, 0.1},
		}, PolicyCostWeightOutOfBoundsClass},
		"weight above the bounds": {listing, adjustable, PolicyServiceRequirements{
			CostWeights: []float64{1.5},
		}, PolicyCostWeightOutOfBoundsClass},
		"weight on a package without bounds": {listing, fixed, PolicyServiceRequirements{
			CostWeights: []float64{0.5},
		}, PolicyCostWeightNotAdjustableClass},
		"weight on an older service": {older, &older.Packages[0], PolicyServiceRequirements{
			CostWeights: []float64{0.5},
		}, PolicyCostWeightNotAdjustableClass},
	} {
		checkErr := CheckPolicyServiceRequirements(check.listing, check.loaded, check.required)
		if check.class == "" {
			if checkErr != nil {
				t.Errorf("%s: %v", name, checkErr)
			}
			continue
		}
		var failure *PolicyServiceError
		if !errors.As(checkErr, &failure) || failure.Class != check.class {
			t.Errorf("%s: error %v, want class %s", name, checkErr, check.class)
		}
	}
	err = CheckPolicyServiceRequirements(listing, adjustable, PolicyServiceRequirements{Capabilities: []string{"exploration"}})
	var failure *PolicyServiceError
	if !errors.As(err, &failure) || failure.Detail["capability"] != "exploration" {
		t.Fatalf("the refusal does not name the capability: %v", err)
	}
}

// The listing is always asked for detail=capabilities, with or without a pin.
func TestPolicyServiceClientAsksForCapabilities(t *testing.T) {
	body := readPolicyFixture(t, "packages_response_capabilities.v1.json")
	var queries []map[string][]string
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		queries = append(queries, request.URL.Query())
		_, _ = writer.Write(body)
	}))
	defer server.Close()
	client := NewPolicyServiceClient(PolicyServiceConfig{BaseURL: server.URL, TotalTimeout: 5 * time.Second})
	if _, err := client.Packages(context.Background(), "", ""); err != nil {
		t.Fatal(err)
	}
	listing, loaded, err := client.ServedPackage(context.Background(),
		"rayline/arc-example-fold0", "81e4a7af2c28c3be9ecbe850888c6773d196d58cad100474cbfb48622af7abe4")
	if err != nil {
		t.Fatal(err)
	}
	if len(listing.Capabilities) != 3 || loaded.AdjustableCostWeight.Bounds == nil {
		t.Fatalf("served package = %+v in %+v", loaded, listing)
	}
	if len(queries) != 2 {
		t.Fatalf("%d listings, want 2", len(queries))
	}
	for index, query := range queries {
		if !slices.Equal(query["detail"], []string{"capabilities"}) {
			t.Errorf("listing %d query = %v", index, query)
		}
	}
	if queries[0]["alias"] != nil || !slices.Equal(queries[1]["alias"], []string{"rayline/arc-example-fold0"}) {
		t.Fatalf("pins = %v, %v", queries[0], queries[1])
	}
}
