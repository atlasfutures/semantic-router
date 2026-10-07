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

// The per-session base owns the thinking fields: the client's own thinking,
// reasoning, reasoning_effort and output_config.effort are removed on every
// rendered call (other output_config keys stay), then the cell's base wire is
// written, so a default base sends none of them and a Messages effort gets no
// adaptive thinking block. In-band system messages after the governed turn
// stay where they are; the governed turn is the last unit before them.
//
// Serialization: compact JSON, UTF-8, no escaping beyond JSON's own, object
// key order as the client sent it with inserted keys last.

// Passthrough is a request with no lever: the client's body with only the
// model id replaced.
func Passthrough(clientBody []byte, wireModel string) ([]byte, error) {
	body, err := parseJSON(clientBody)
	if err != nil || body.kind != kindObject {
		return nil, refuse("the client body is not a JSON object")
	}
	body.set("model", stringValue(wireModel))
	return dump(body)
}

// Render places this call's control on the client's body, writes the base's
// wire and the model id, and returns the provider body and the receipt. The
// placer advances only when the call renders.
func (p *Placer) Render(clientBody []byte, control *Control, cell *Cell, wireModel string) ([]byte, Receipt, error) {
	return p.render(clientBody, control, cell.baseWire, wireModel)
}

func (p *Placer) render(
	clientBody []byte, control *Control, baseWire map[string]*value, wireModel string,
) ([]byte, Receipt, error) {
	// Every render path (Render, RenderAdmitted, the golden corpora) comes through here, so a control naming a
	// rendering unit this router does not implement is refused once, for all of them.
	if control != nil && control.Instruction != nil && control.Instruction.Unit != "" {
		return nil, Receipt{}, refuse("call %d: a %s unit is not rendered by this router", p.state.Calls,
			control.Instruction.Unit)
	}
	if err := p.check(control); err != nil {
		return nil, Receipt{}, err
	}
	wire, ok := baseWire[control.Native]
	if !ok {
		return nil, Receipt{}, refuse("call %d: base '%s' has no wire on this cell", p.state.Calls, control.Native)
	}
	body, err := parseJSON(clientBody)
	if err != nil || body.kind != kindObject {
		return nil, Receipt{}, refuse("call %d: the client body is not a JSON object", p.state.Calls)
	}
	before := p.state
	placed, receipt, err := p.place(body, control)
	if err != nil {
		return nil, Receipt{}, err
	}
	placed.set("model", stringValue(wireModel))
	ownThinkingFields(placed, wire)
	out, err := dump(placed)
	if err != nil {
		p.state = before
		return nil, Receipt{}, refuse("call %d: %v", receipt.Call, err)
	}
	return out, receipt, nil
}

// RenderAdmitted renders one episode on an admitted cell: every control must
// be admitted there.
func RenderAdmitted(
	reg *Registry, clientBodies [][]byte, controls []*Control,
	model, provider, format, wireModel string, allowExperimental bool,
) ([][]byte, []Receipt, error) {
	var cell *Cell
	for index, control := range controls {
		if control == nil {
			return nil, nil, refuse("call %d: no control; absence is not the 'none' level", index)
		}
		admitted, err := reg.Admit(model, provider, format, *control, allowExperimental)
		if err != nil {
			return nil, nil, err
		}
		cell = admitted
	}
	if cell == nil {
		return nil, nil, refuse("one control per call")
	}
	return renderEpisode(clientBodies, controls, cell.baseWire, format, wireModel)
}

// renderEpisode is placement only: every call rendered given the base's
// wire, with no admission check. The golden corpora use it to pin bytes for
// a cell no instruction is admitted on yet.
func renderEpisode(
	clientBodies [][]byte, controls []*Control, baseWire map[string]*value, format, wireModel string,
) ([][]byte, []Receipt, error) {
	if len(clientBodies) != len(controls) {
		return nil, nil, refuse("one control per call")
	}
	placer, err := NewPlacer(format)
	if err != nil {
		return nil, nil, err
	}
	bodies := make([][]byte, 0, len(controls))
	receipts := make([]Receipt, 0, len(controls))
	for index := range controls {
		body, receipt, err := placer.render(clientBodies[index], controls[index], baseWire, wireModel)
		if err != nil {
			return nil, nil, err
		}
		bodies = append(bodies, body)
		receipts = append(receipts, receipt)
	}
	return bodies, receipts, nil
}

// ownThinkingFields removes the client's thinking fields, then writes the
// base's wire (other output_config keys stay).
func ownThinkingFields(body *value, wire *value) {
	for _, key := range []string{"thinking", "reasoning", "reasoning_effort"} {
		body.remove(key)
	}
	var remaining *value
	if outputConfig := body.remove("output_config"); outputConfig != nil && outputConfig.kind == kindObject {
		remaining = outputConfig.clone()
		remaining.remove("effort")
		if len(remaining.members) > 0 {
			body.set("output_config", remaining)
		} else {
			remaining = nil
		}
	}
	for _, m := range wire.members {
		if m.key == "output_config" && remaining != nil && m.val.kind == kindObject {
			merged := remaining.clone()
			for _, field := range m.val.members {
				merged.set(field.key, field.val.clone())
			}
			body.set(m.key, merged)
			continue
		}
		body.set(m.key, m.val.clone())
	}
}
