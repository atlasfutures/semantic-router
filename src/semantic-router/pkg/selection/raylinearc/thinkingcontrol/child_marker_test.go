/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package thinkingcontrol

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// The child_marker corpus is pathfinder's tests/fixtures/thinking_control_child_marker,
// mirrored byte for byte with the golden corpora. It records what pathfinder's
// campaign proxy places on a post-compaction child of a registry parent: a
// reset marker whose control is derived from the parent's ladder, not taken
// from the registry. A fold parent's child carries inband_system on that
// control and is placed under FOLD_RULES.
//
// VSR does not serve such children. Its controls come only from the registry
// a package binds, and the registry lists no child-marker control, so a child
// marker can never be admitted here. This test pins that.
//
// The corpus is mirrored, not rendered: pathfinder's child placer writes the
// reset marker on a child's first call even at level none (an explicit base
// assertion after compaction), which VSR's registry placement never does, for
// a fold parent and a text parent alike. Serving children would need that
// placer ported first.
type childMarkerCase struct {
	Format  string          `json:"format"`
	Control json.RawMessage `json:"control"`
	Calls   []struct {
		ClientBody   string          `json:"client_body"`
		ExpectedBody string          `json:"expected_body"`
		Receipt      json.RawMessage `json:"receipt"`
	} `json:"calls"`
	RegistryPin string `json:"registry_pin"`
}

func childMarkerCases(t *testing.T) map[string]string {
	t.Helper()
	cases := map[string]string{}
	for _, format := range childMarkerFormats {
		dir := filepath.Join("testdata", "child_marker", format)
		for _, name := range caseNames(t, dir) {
			cases[format+"/"+name] = filepath.Join(dir, name)
		}
	}
	if len(cases) == 0 {
		t.Fatal("the child_marker corpus is empty")
	}
	return cases
}

func readChildMarkerCase(t *testing.T, dir string) childMarkerCase {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	var c childMarkerCase
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

// A child marker's control parses under pathfinder's id, and no cell admits
// it: the registry names no child control, so no package can bind one.
func TestAChildMarkerControlIsNeverAdmitted(t *testing.T) {
	reg, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for name, dir := range childMarkerCases(t) {
		t.Run(name, func(t *testing.T) {
			c := readChildMarkerCase(t, dir)
			if c.RegistryPin != reg.SHA256 {
				t.Fatalf("case pins registry %s, loaded %s", c.RegistryPin, reg.SHA256)
			}
			control, err := ParseControl(c.Control)
			if err != nil {
				t.Fatalf("parse the marker's control: %v", err)
			}
			var receipt struct {
				ControlID string `json:"control_id"`
			}
			if err := json.Unmarshal(c.Calls[0].Receipt, &receipt); err != nil {
				t.Fatal(err)
			}
			if id := control.ID(); id != receipt.ControlID {
				t.Fatalf("control id %s, pathfinder's %s", id, receipt.ControlID)
			}
			if _, listed := reg.Control(receipt.ControlID); listed {
				t.Fatalf("the registry lists child control %s", prefix12(receipt.ControlID))
			}
			for key := range reg.cells {
				if _, err := reg.Admit(key.model, key.provider, key.format, control, true); err == nil {
					t.Fatalf("%v admits a child-marker control", key)
				}
			}
		})
	}
}
