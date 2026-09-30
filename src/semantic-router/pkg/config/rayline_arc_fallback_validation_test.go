package config

import (
	"strings"
	"testing"

	yamlv3 "gopkg.in/yaml.v3"
)

func parseReferenceConfigWith(t *testing.T, mutate func(root map[string]interface{})) error {
	t.Helper()
	root := loadReferenceConfigRaw(t)
	mutate(root)
	data, err := yamlv3.Marshal(root)
	if err != nil {
		t.Fatalf("yaml.Marshal() error = %v", err)
	}
	_, err = ParseYAMLBytes(data)
	return err
}

func referenceRecipe(t *testing.T, root map[string]interface{}, name string) map[string]interface{} {
	t.Helper()
	recipes, _ := root["recipes"].([]interface{})
	for _, item := range recipes {
		recipe, _ := item.(map[string]interface{})
		if recipe["name"] == name {
			return recipe
		}
	}
	t.Fatalf("config/config.yaml has no recipe %q", name)
	return nil
}

// A global fallback enabled: true reaches every recipe that does not override
// it, including the ARC ones, so the config must be refused.
func TestRaylineARCRefusesInheritedGlobalFallback(t *testing.T) {
	err := parseReferenceConfigWith(t, func(root map[string]interface{}) {
		global := root["global"].(map[string]interface{})
		router := global["router"].(map[string]interface{})
		router["fallback"].(map[string]interface{})["enabled"] = true
	})
	if err == nil || !strings.Contains(err.Error(), "does not support upstream fallback") {
		t.Fatalf("error = %v, want the ARC fallback refusal", err)
	}
}

func TestRaylineARCRefusesRecipeFallback(t *testing.T) {
	err := parseReferenceConfigWith(t, func(root map[string]interface{}) {
		recipe := referenceRecipe(t, root, "rayline-arc-policy")
		routing := recipe["routing"].(map[string]interface{})
		routing["fallback"] = map[string]interface{}{"version": 1, "enabled": true, "max_attempts": 2}
	})
	if err == nil || !strings.Contains(err.Error(), "does not support upstream fallback") ||
		!strings.Contains(err.Error(), "rayline-arc-policy") {
		t.Fatalf("error = %v, want the ARC fallback refusal naming rayline-arc-policy", err)
	}
}

// The refusal is scoped to recipes that hold a rayline_arc decision: a
// non-ARC recipe may enable fallback while the ARC recipes inherit the
// disabled global policy.
func TestRaylineARCAllowsFallbackInNonARCRecipe(t *testing.T) {
	enabledInOther := false
	err := parseReferenceConfigWith(t, func(root map[string]interface{}) {
		recipes, _ := root["recipes"].([]interface{})
		for _, item := range recipes {
			recipe := item.(map[string]interface{})
			routing := recipe["routing"].(map[string]interface{})
			if recipeHasRaylineARCDecision(routing) {
				continue
			}
			routing["fallback"] = map[string]interface{}{"version": 1, "enabled": true, "max_attempts": 2}
			enabledInOther = true
			return
		}
	})
	if !enabledInOther {
		t.Fatal("config/config.yaml has no recipe without a rayline_arc decision")
	}
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v, want a non-ARC recipe with fallback enabled to load", err)
	}
}

func recipeHasRaylineARCDecision(routing map[string]interface{}) bool {
	decisions, _ := routing["decisions"].([]interface{})
	for _, item := range decisions {
		decision, _ := item.(map[string]interface{})
		algorithm, _ := decision["algorithm"].(map[string]interface{})
		if algorithm["type"] == RaylineARCAlgorithmType {
			return true
		}
	}
	return false
}
