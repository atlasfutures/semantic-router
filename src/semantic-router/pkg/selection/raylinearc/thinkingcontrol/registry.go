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
	_ "embed"
	"encoding/hex"
	"sync"
)

// CompiledSchemaVersion is the compiled registry artifact's schema.
const CompiledSchemaVersion = "rayline-router.thinking-controls.v1"

// Wire formats, as the registry names them.
const (
	FormatMessages  = "messages"
	FormatChat      = "chat"
	FormatResponses = "responses"
)

// Admission levels of a cell's instruction.
const (
	AdmissionCertified    = "certified"
	AdmissionExperimental = "experimental"
	AdmissionRefused      = "refused"
)

// embeddedArtifact is pathfinder's configs/thinking_controls.compiled.json,
// mirrored byte for byte; EmbeddedSHA256 pins it.
//
//go:embed thinking_controls.compiled.json
var embeddedArtifact []byte

// EmbeddedSHA256 is the sha256 of the embedded artifact, as pathfinder's
// golden corpora record it in registry_pin.
const EmbeddedSHA256 = "3083a4b6bd857c951177f973b41900798ba22f4968e224b4a29ad3e9c297cff2"

var (
	embeddedOnce sync.Once
	embedded     *Registry
	embeddedErr  error
)

// Embedded is the registry VSR serves from: the artifact mirrored into this
// package, loaded and checked once.
func Embedded() (*Registry, error) {
	embeddedOnce.Do(func() { embedded, embeddedErr = Load(embeddedArtifact) })
	return embedded, embeddedErr
}

// Registry is a loaded compiled artifact.
type Registry struct {
	// SHA256 is the sha256 of the artifact's bytes.
	SHA256         string
	RegistrySHA256 string
	controls       map[string]Control
	cells          map[cellKey]*Cell
}

type cellKey struct{ model, provider, format string }

// Cell is one (model, provider, format)'s admission: the instruction's
// admission level, the wire each admitted base writes, and the ids of every
// control the cell admits.
type Cell struct {
	Model       string
	Provider    string
	Format      string
	Instruction string
	baseWire    map[string]*value
	controls    map[string]bool
}

// Load reads a compiled artifact. Its bytes must be canonical JCS and every
// control id must recompute, as pathfinder's load_compiled requires.
func Load(raw []byte) (*Registry, error) {
	parsed, err := parseJSON(raw)
	if err != nil {
		return nil, refuse("the compiled artifact is not JSON: %v", err)
	}
	canonical, err := jcs(parsed)
	if err != nil || !bytes.Equal(canonical, raw) {
		return nil, refuse("the compiled artifact is not in canonical JCS form")
	}
	if schema := parsed.get("schema_version"); !schema.isString() || schema.str != CompiledSchemaVersion {
		return nil, refuse("schema_version must be %q", CompiledSchemaVersion)
	}
	sum := sha256.Sum256(raw)
	reg := &Registry{
		SHA256:   hex.EncodeToString(sum[:]),
		controls: map[string]Control{},
		cells:    map[cellKey]*Cell{},
	}
	if registrySHA := parsed.get("registry_sha256"); registrySHA.isString() {
		reg.RegistrySHA256 = registrySHA.str
	}
	controls := parsed.get("controls")
	if controls == nil || controls.kind != kindObject {
		return nil, refuse("the compiled artifact has no controls object")
	}
	for _, m := range controls.members {
		control, err := requireControl(m.val, "control "+prefix12(m.key))
		if err != nil {
			return nil, err
		}
		if control.ID() != m.key {
			return nil, refuse("control %s does not hash to its id", prefix12(m.key))
		}
		reg.controls[m.key] = control
	}
	cells := parsed.get("cells")
	if cells == nil || cells.kind != kindArray {
		return nil, refuse("the compiled artifact has no cells list")
	}
	for _, raw := range cells.items {
		cell, err := loadCell(raw, reg.controls)
		if err != nil {
			return nil, err
		}
		reg.cells[cellKey{cell.Model, cell.Provider, cell.Format}] = cell
	}
	return reg, nil
}

func loadCell(raw *value, controls map[string]Control) (*Cell, error) {
	str := func(key string) string {
		if v := raw.get(key); v.isString() {
			return v.str
		}
		return ""
	}
	cell := &Cell{
		Model: str("model"), Provider: str("provider"), Format: str("format"), Instruction: str("instruction"),
		baseWire: map[string]*value{}, controls: map[string]bool{},
	}
	where := "cell " + cell.Model + " " + cell.Provider + "x" + cell.Format
	ids := raw.get("controls")
	wire := raw.get("base_wire")
	endpoints := raw.get("base_endpoints")
	if ids == nil || ids.kind != kindArray || wire == nil || wire.kind != kindObject ||
		endpoints == nil || endpoints.kind != kindObject {
		return nil, refuse("%s is malformed", where)
	}
	bases := map[string]bool{}
	unknown := false
	for _, id := range ids.items {
		if !id.isString() {
			return nil, refuse("%s is malformed", where)
		}
		control, ok := controls[id.str]
		if !ok {
			unknown = true
			continue
		}
		cell.controls[id.str] = true
		bases[control.Native] = true
	}
	if !sameKeys(wire, bases) || !sameKeys(endpoints, bases) {
		return nil, refuse("%s must state the wire and endpoint bound of every base it admits", where)
	}
	if unknown {
		return nil, refuse("%s names unknown controls", where)
	}
	for _, m := range wire.members {
		if m.val.kind != kindObject {
			return nil, refuse("%s: the wire of base %q is not an object", where, m.key)
		}
		cell.baseWire[m.key] = m.val
	}
	return cell, nil
}

func sameKeys(obj *value, keys map[string]bool) bool {
	if len(obj.members) != len(keys) {
		return false
	}
	for _, m := range obj.members {
		if !keys[m.key] {
			return false
		}
	}
	return true
}

func prefix12(id string) string {
	if len(id) > 12 {
		return id[:12]
	}
	return id
}

// Control returns the control with this id.
func (r *Registry) Control(id string) (Control, bool) {
	control, ok := r.controls[id]
	return control, ok
}

// Cell returns the admission cell for (model, provider, format).
func (r *Registry) Cell(model, provider, format string) (*Cell, error) {
	cell, ok := r.cells[cellKey{model, provider, format}]
	if !ok {
		return nil, refuse("no admission cell for %s on %sx%s", model, provider, format)
	}
	return cell, nil
}

// Admit returns the cell if it admits control, and refuses otherwise. An
// instruction on an experimental cell is refused unless allowExperimental.
func (r *Registry) Admit(model, provider, format string, control Control, allowExperimental bool) (*Cell, error) {
	cell, err := r.Cell(model, provider, format)
	if err != nil {
		return nil, err
	}
	id := control.ID()
	if !cell.controls[id] {
		return nil, refuse("control %s is not admitted for %s on %sx%s", prefix12(id), model, provider, format)
	}
	if control.Instruction != nil && cell.Instruction == AdmissionExperimental && !allowExperimental {
		return nil, refuse("%s on %sx%s: the instruction is experimental only", model, provider, format)
	}
	return cell, nil
}

// Admits reports whether the cell admits the control with this id.
func (c *Cell) Admits(id string) bool { return c.controls[id] }
