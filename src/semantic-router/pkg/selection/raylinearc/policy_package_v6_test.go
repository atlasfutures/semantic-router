package raylinearc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"
)

const (
	policyV6Fixture = "package_manifest.v6.canonical_v2.json"
	// policyV6FixtureSHA256 is pathfinder's package_sha256 for the fixture
	// (da1fbb6608, tests/fixtures/arc_policy_service).
	policyV6FixtureSHA256 = "ee0e9b47a5ff2e08b9c14e41a8436be5b08d8f1ed2b3bccf12a7993930325e13"
)

// manifestMap decodes a fixture into a generic value for editing.
func manifestMap(t *testing.T, name string) map[string]any {
	t.Helper()
	var manifest map[string]any
	if err := json.Unmarshal(readPolicyFixture(t, name), &manifest); err != nil {
		t.Fatal(err)
	}
	return manifest
}

func profileOf(manifest map[string]any) map[string]any {
	return manifest["encoding_profile"].(map[string]any)
}

func encodeManifest(t *testing.T, manifest map[string]any) []byte {
	t.Helper()
	body, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	return body
}

// pathfinder's v6 fixture decodes as v6 and through the v5-or-later entry
// point, and the bytes VSR verifies are the package_sha256 pathfinder names.
func TestDecodePolicyPackageManifestV6Fixture(t *testing.T) {
	body := readPolicyFixture(t, policyV6Fixture)
	sum := sha256.Sum256(body)
	if got := hex.EncodeToString(sum[:]); got != policyV6FixtureSHA256 {
		t.Fatalf("fixture sha256 %s, pathfinder's package_sha256 is %s", got, policyV6FixtureSHA256)
	}
	for name, decode := range map[string]func([]byte) (*PolicyPackageManifestV5, error){
		"v6": DecodePolicyPackageManifestV6, "v5 or later": DecodePolicyPackageManifestV5OrLater,
	} {
		manifest, err := decode(body)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		profile := manifest.EncodingProfile
		if manifest.SchemaVersion != PolicyPackageSchemaV6 || profile.Serializer != PolicySerializerV2 ||
			string(profile.Conversation) != `"canonical_v2"` || len(manifest.Actions) != 3 {
			t.Fatalf("%s decoded %s %s %s with %d actions", name, manifest.SchemaVersion, profile.Serializer,
				profile.Conversation, len(manifest.Actions))
		}
	}
	// v5 manifests keep decoding through the same entry point.
	for _, v5 := range []string{"package_manifest.v5.json", "package_manifest.v5.canonical_v1.json", "package_manifest.v5.yarn.json"} {
		if _, err := DecodePolicyPackageManifestV5OrLater(readPolicyFixture(t, v5)); err != nil {
			t.Errorf("%s: %v", v5, err)
		}
	}
}

// Each image member's absence or a value outside the contract's table is
// refused, and so is an unknown key at every level of the profile.
func TestDecodePolicyPackageManifestV6Refusals(t *testing.T) {
	for name, edit := range map[string]func(profile map[string]any){
		"text serializer":             func(p map[string]any) { p["serializer"] = "arc-role-blocks-v1" },
		"conversation absent":         func(p map[string]any) { delete(p, "conversation") },
		"conversation canonical_v1":   func(p map[string]any) { p["conversation"] = "canonical_v1" },
		"conversation null":           func(p map[string]any) { p["conversation"] = nil },
		"modalities absent":           func(p map[string]any) { delete(p, "modalities") },
		"modalities text only":        func(p map[string]any) { p["modalities"] = []any{"text"} },
		"modalities reordered":        func(p map[string]any) { p["modalities"] = []any{"image", "text"} },
		"image_processor absent":      func(p map[string]any) { delete(p, "image_processor") },
		"positions absent":            func(p map[string]any) { delete(p, "positions") },
		"positions other":             func(p map[string]any) { p["positions"] = "rope_1d" },
		"vision absent":               func(p map[string]any) { delete(p, "vision") },
		"unknown profile key":         func(p map[string]any) { p["audio"] = true },
		"unknown image_processor key": func(p map[string]any) { sub(p, "image_processor")["crop"] = "center" },
		"unknown vision key":          func(p map[string]any) { sub(p, "vision")["batch"] = 1 },
		"rope null":                   func(p map[string]any) { p["rope"] = nil },
		"rope other":                  func(p map[string]any) { p["rope"] = "yarn-f8" },
		"harness_injections keep":     func(p map[string]any) { p["harness_injections"] = "keep" },
		"harness_injections null":     func(p map[string]any) { p["harness_injections"] = nil },
		"harness_shell exclude":       func(p map[string]any) { p["harness_shell"] = "exclude" },
		"harness_shell include_v3":    func(p map[string]any) { p["harness_shell"] = "include_v3" },
		"environment keep":            func(p map[string]any) { p["environment"] = "keep" },
		"layer other":                 func(p map[string]any) { p["layer"] = "layer_3" },
		"projection other":            func(p map[string]any) { p["projection"] = "pathfinder.input-projection.v2" },
		"tool_definitions other":      func(p map[string]any) { p["tool_definitions"] = "summarize" },
		"encoder_model absent":        func(p map[string]any) { delete(p, "encoder_model") },
		"encoder_revision short":      func(p map[string]any) { p["encoder_revision"] = "2fc0636" },
		"dtype other":                 func(p map[string]any) { p["dtype"] = "float16" },
		"max_tokens zero":             func(p map[string]any) { p["max_tokens"] = 0 },
		"readout other":               func(p map[string]any) { p["readout"] = "mean_pool" },
		"dimension absent":            func(p map[string]any) { delete(p, "dimension") },
		"image rule other":            func(p map[string]any) { sub(p, "image_processor")["rule"] = "arc-image-v2" },
		"image resample other":        func(p map[string]any) { sub(p, "image_processor")["resample"] = "pillow_lanczos" },
		"image decode other":          func(p map[string]any) { sub(p, "image_processor")["decode"] = "all_frames" },
		"image factor mismatch":       func(p map[string]any) { sub(p, "image_processor")["factor"] = 30 },
		"image min above max":         func(p map[string]any) { sub(p, "image_processor")["min_pixels"] = 2_000_000 },
		"image integer zero":          func(p map[string]any) { sub(p, "image_processor")["max_aspect_ratio"] = 0 },
		"image integer absent":        func(p map[string]any) { delete(sub(p, "image_processor"), "max_long_edge") },
		"image integer fractional":    func(p map[string]any) { sub(p, "image_processor")["patch_size"] = 16.5 },
		"vision dtype bfloat16":       func(p map[string]any) { sub(p, "vision")["dtype"] = "bfloat16" },
		"vision embed_cast other":     func(p map[string]any) { sub(p, "vision")["embed_cast"] = "float32" },
		"per_image_calls false":       func(p map[string]any) { sub(p, "vision")["per_image_calls"] = false },
		"per_image_calls string":      func(p map[string]any) { sub(p, "vision")["per_image_calls"] = "true" },
		"per_image_calls one":         func(p map[string]any) { sub(p, "vision")["per_image_calls"] = 1 },
		"per_image_calls absent":      func(p map[string]any) { delete(sub(p, "vision"), "per_image_calls") },
	} {
		t.Run(name, func(t *testing.T) {
			manifest := manifestMap(t, policyV6Fixture)
			edit(profileOf(manifest))
			if _, err := DecodePolicyPackageManifestV6(encodeManifest(t, manifest)); err == nil {
				t.Fatal("decoded")
			}
		})
	}
}

func sub(profile map[string]any, member string) map[string]any {
	return profile[member].(map[string]any)
}

// The members a v6 profile may state beyond the fixture's, at their contract
// values, decode; layer final is the contract's default, which its writer
// omits.
func TestDecodePolicyPackageManifestV6OptionalMembers(t *testing.T) {
	for name, edit := range map[string]func(profile map[string]any){
		"harness_shell include":     func(p map[string]any) { p["harness_shell"] = "include" },
		"harness_shell include_v2":  func(p map[string]any) { p["harness_shell"] = "include_v2" },
		"environment scrub_v1":      func(p map[string]any) { p["environment"] = "scrub_v1" },
		"environment scrub_v2":      func(p map[string]any) { p["environment"] = "scrub_v2" },
		"layer block_11":            func(p map[string]any) { p["layer"] = "block_11" },
		"layer final":               func(p map[string]any) { p["layer"] = "final" },
		"rope yarn-f4":              func(p map[string]any) { p["rope"] = "yarn-f4" },
		"harness_injections absent": func(p map[string]any) { delete(p, "harness_injections") },
		"tool_definitions exclude":  func(p map[string]any) { p["tool_definitions"] = "exclude" },
		"dtype float32":             func(p map[string]any) { p["dtype"] = "float32" },
	} {
		t.Run(name, func(t *testing.T) {
			manifest := manifestMap(t, policyV6Fixture)
			edit(profileOf(manifest))
			if _, err := DecodePolicyPackageManifestV6(encodeManifest(t, manifest)); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// The schemas are not interchangeable: a v4 or v5 manifest holding any v6
// member, or the v6 profile spliced in whole, is refused, and so is the v6
// fixture with v5's canonical_v1 profile spliced in.
func TestPolicyPackageProfilesDoNotCrossSchemas(t *testing.T) {
	v6Profile := profileOf(manifestMap(t, policyV6Fixture))
	v5Profile := profileOf(manifestMap(t, "package_manifest.v5.canonical_v1.json"))
	decoders := map[string]func([]byte) error{
		"package_manifest.v4.json": func(b []byte) error { _, err := DecodePolicyPackageManifest(b); return err },
		"package_manifest.v5.json": func(b []byte) error { _, err := DecodePolicyPackageManifestV5(b); return err },
		"package_manifest.v5.canonical_v1.json": func(b []byte) error {
			_, err := DecodePolicyPackageManifestV5(b)
			return err
		},
	}
	for fixture, decode := range decoders {
		edits := map[string]func(profile map[string]any){
			"the v6 profile spliced in": func(p map[string]any) {
				clear(p)
				for key, value := range v6Profile {
					p[key] = value
				}
			},
			"serializer v2":             func(p map[string]any) { p["serializer"] = PolicySerializerV2 },
			"conversation canonical_v2": func(p map[string]any) { p["conversation"] = "canonical_v2" },
		}
		for _, member := range []string{"modalities", "image_processor", "positions", "vision"} {
			edits[member] = func(p map[string]any) { p[member] = v6Profile[member] }
		}
		for name, edit := range edits {
			t.Run(fixture+"/"+name, func(t *testing.T) {
				manifest := manifestMap(t, fixture)
				edit(profileOf(manifest))
				if err := decode(encodeManifest(t, manifest)); err == nil {
					t.Fatal("decoded")
				}
			})
		}
		if err := decode(readPolicyFixture(t, fixture)); err != nil {
			t.Errorf("%s itself: %v", fixture, err)
		}
	}
	t.Run("v6 with the v5 canonical_v1 profile spliced in", func(t *testing.T) {
		manifest := manifestMap(t, policyV6Fixture)
		manifest["encoding_profile"] = v5Profile
		if _, err := DecodePolicyPackageManifestV6(encodeManifest(t, manifest)); err == nil {
			t.Fatal("decoded")
		}
	})
}

// v6 keeps v5's top-level checks, and the v5-or-later entry point refuses
// every other schema.
func TestDecodePolicyPackageManifestV6SharesV5Checks(t *testing.T) {
	for name, edit := range map[string]func(manifest map[string]any){
		"live prices": func(m map[string]any) {
			m["pricing"].(map[string]any)["live_prices_affect_decisions"] = true
		},
		"no actions": func(m map[string]any) { m["actions"] = []any{} },
		"repeated action": func(m map[string]any) {
			actions := m["actions"].([]any)
			m["actions"] = append(actions, actions[0])
		},
		"control_id that does not recompute": func(m map[string]any) {
			m["actions"].([]any)[0].(map[string]any)["control_id"] = strings.Repeat("0", 64)
		},
		"unknown top-level key": func(m map[string]any) { m["stage_two"] = nil },
		"v5 schema":             func(m map[string]any) { m["schema_version"] = PolicyPackageSchemaV5 },
	} {
		t.Run(name, func(t *testing.T) {
			manifest := manifestMap(t, policyV6Fixture)
			edit(manifest)
			if _, err := DecodePolicyPackageManifestV6(encodeManifest(t, manifest)); err == nil {
				t.Fatal("decoded")
			}
		})
	}
	for _, schema := range []string{PolicyPackageSchema, "rayline.arc-policy-package.v7", ""} {
		manifest := manifestMap(t, policyV6Fixture)
		manifest["schema_version"] = schema
		if _, err := DecodePolicyPackageManifestV5OrLater(encodeManifest(t, manifest)); err == nil {
			t.Errorf("schema %q decoded", schema)
		}
	}
}
