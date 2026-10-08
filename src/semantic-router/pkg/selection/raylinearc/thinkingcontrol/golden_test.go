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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The corpora are pathfinder's tests/fixtures/thinking_control_golden and
// thinking_control_placement at 5dd3377dfa (pathfinder#3933), mirrored byte for byte; each
// directory's SHA256SUMS pins the copy, so drift is a diff to review.
//
// The golden corpus respects admission: a case whose control its cell
// refuses records the registry's refusal and no bytes. The placement corpus
// holds those same cases rendered without admission
// (placement_only_not_admitted), so the renderer's placement keeps its byte
// coverage; it is never a router oracle.
var goldenFormats = []string{FormatMessages, FormatChat, FormatResponses}

var placementFormats = []string{FormatChat, FormatResponses}

func corpusDirs() []string {
	var dirs []string
	for _, format := range goldenFormats {
		dirs = append(dirs, filepath.Join("testdata", "golden", format))
	}
	for _, format := range placementFormats {
		dirs = append(dirs, filepath.Join("testdata", "placement", format))
	}
	return dirs
}

type goldenCase struct {
	Description       string `json:"description"`
	Format            string `json:"format"`
	Provider          string `json:"provider"`
	Model             string `json:"model"`
	WireModel         string `json:"wire_model"`
	Admission         string `json:"admission"`
	AllowExperimental bool   `json:"allow_experimental"`
	RegistryPin       string `json:"registry_pin"`
	Calls             []struct {
		ClientBody   string          `json:"client_body"`
		ControlID    *string         `json:"control_id"`
		ExpectedBody string          `json:"expected_body"`
		Receipt      json.RawMessage `json:"receipt"`
	} `json:"calls"`
	Refusal *struct {
		Error string `json:"error"`
	} `json:"refusal"`
}

func TestGoldenCorporaArePinned(t *testing.T) {
	for _, dir := range corpusDirs() {
		format := dir
		sums, err := os.ReadFile(filepath.Join(dir, "SHA256SUMS"))
		if err != nil {
			t.Fatal(err)
		}
		pinned := map[string]bool{}
		for _, line := range strings.Split(strings.TrimSpace(string(sums)), "\n") {
			want, name, ok := strings.Cut(line, "  ")
			if !ok {
				t.Fatalf("%s: malformed SHA256SUMS line %q", format, line)
			}
			raw, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(name)))
			if err != nil {
				t.Fatal(err)
			}
			if got := sha256.Sum256(raw); hex.EncodeToString(got[:]) != want {
				t.Fatalf("%s/%s does not match its pinned sha256", format, name)
			}
			pinned[name] = true
		}
		err = filepath.WalkDir(dir, func(path string, entry os.DirEntry, err error) error {
			if err != nil || entry.IsDir() || entry.Name() == "SHA256SUMS" {
				return err
			}
			rel, _ := filepath.Rel(dir, path)
			if !pinned[filepath.ToSlash(rel)] {
				t.Errorf("%s/%s is not pinned by SHA256SUMS", format, rel)
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}
}

// Every case of every golden corpus renders to pathfinder's provider bytes
// and receipts, or refuses where pathfinder refuses.
func TestGoldenParity(t *testing.T) {
	runCorpus(t, "golden", goldenFormats)
}

// Every placement case renders to pathfinder's bytes and receipts without
// admission, and every golden case its cell refuses has one, so a refusal
// never costs the renderer's placement its byte coverage.
func TestPlacementParity(t *testing.T) {
	runCorpus(t, "placement", placementFormats)
	for _, format := range goldenFormats {
		dir := filepath.Join("testdata", "golden", format)
		for _, name := range caseNames(t, dir) {
			if readCase(t, filepath.Join(dir, name)).Refusal == nil {
				continue
			}
			if _, err := os.Stat(filepath.Join("testdata", "placement", format, name, "case.json")); err != nil {
				t.Errorf("golden %s/%s is refused and has no placement case", format, name)
			}
		}
	}
}

func runCorpus(t *testing.T, corpus string, formats []string) {
	t.Helper()
	reg, err := Embedded()
	if err != nil {
		t.Fatalf("load the embedded registry: %v", err)
	}
	if reg.SHA256 != EmbeddedSHA256 {
		t.Fatalf("embedded registry sha256 %s, want %s", reg.SHA256, EmbeddedSHA256)
	}
	for _, format := range formats {
		dir := filepath.Join("testdata", corpus, format)
		for _, name := range caseNames(t, dir) {
			t.Run(format+"/"+name, func(t *testing.T) {
				runGoldenCase(t, reg, filepath.Join(dir, name))
			})
		}
	}
}

func caseNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		if entry.IsDir() {
			names = append(names, entry.Name())
		}
	}
	sort.Strings(names)
	return names
}

func readCase(t *testing.T, dir string) goldenCase {
	t.Helper()
	var c goldenCase
	raw, err := os.ReadFile(filepath.Join(dir, "case.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		t.Fatal(err)
	}
	return c
}

func runGoldenCase(t *testing.T, reg *Registry, dir string) {
	t.Helper()
	c := readCase(t, dir)
	var err error
	if c.RegistryPin != reg.SHA256 {
		t.Fatalf("case pins registry %s, loaded %s", c.RegistryPin, reg.SHA256)
	}
	bodies := make([][]byte, len(c.Calls))
	controls := make([]*Control, len(c.Calls))
	for i, call := range c.Calls {
		if bodies[i], err = os.ReadFile(filepath.Join(dir, call.ClientBody)); err != nil {
			t.Fatal(err)
		}
		if call.ControlID != nil {
			control, ok := reg.Control(*call.ControlID)
			if !ok {
				t.Fatalf("call %d: control %s is not in the registry", i, *call.ControlID)
			}
			controls[i] = &control
		}
	}
	var got [][]byte
	var receipts []Receipt
	switch c.Admission {
	case "no_lever":
		for _, body := range bodies {
			out, err := Passthrough(body, c.WireModel)
			if err != nil {
				t.Fatal(err)
			}
			got = append(got, out)
		}
	case "enforced":
		got, receipts, err = RenderAdmitted(reg, bodies, controls, c.Model, c.Provider, c.Format, c.WireModel,
			c.AllowExperimental)
	case "placement_only", "placement_only_not_admitted":
		baseWire := map[string]*value{controls[0].Native: objectValue()}
		got, receipts, err = renderEpisode(bodies, controls, baseWire, c.Format, c.WireModel)
	default:
		t.Fatalf("unknown admission %q", c.Admission)
	}
	if c.Refusal != nil {
		if err == nil || err.Error() != c.Refusal.Error {
			t.Fatalf("refusal = %v, want %q", err, c.Refusal.Error)
		}
		return
	}
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	for i, call := range c.Calls {
		want, err := os.ReadFile(filepath.Join(dir, call.ExpectedBody))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got[i], want) {
			t.Errorf("call %d body:\n got %s\nwant %s", i, got[i], want)
		}
		if string(call.Receipt) == "null" || len(call.Receipt) == 0 {
			if receipts != nil {
				t.Errorf("call %d: a receipt where pathfinder records none", i)
			}
			continue
		}
		var wantReceipt, gotReceipt any
		gotRaw, _ := json.Marshal(receipts[i])
		_ = json.Unmarshal(call.Receipt, &wantReceipt)
		_ = json.Unmarshal(gotRaw, &gotReceipt)
		w, _ := json.Marshal(wantReceipt)
		g, _ := json.Marshal(gotReceipt)
		if !bytes.Equal(w, g) {
			t.Errorf("call %d receipt:\n got %s\nwant %s", i, g, w)
		}
	}
}
