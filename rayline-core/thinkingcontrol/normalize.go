package thinkingcontrol

// FoldInbandSystem applies the existing fold_before_unit_v1 request rule
// without adding a steering instruction. Callers must explicitly qualify this
// role-changing compatibility rule for their serving profile; it is not a
// general protocol conversion or a new admitted thinking control.
//
// Leading system rows stay. Later system/developer text and cache units move
// after the current governed turn's content, never to the leading preamble.
// The same placement refusals apply, including an already-answered turn or a
// system row inside a tool-result run. No placer state is read or advanced.
func FoldInbandSystem(clientBody []byte, format string) ([]byte, error) {
	if !knownFormat(format) {
		return nil, refuse("unknown wire format '%s'", format)
	}
	body, err := parseJSON(clientBody)
	if err != nil || body.kind != kindObject {
		return nil, refuse("the client body is not a JSON object")
	}
	units, err := unitsOf(body, format)
	if err != nil {
		return nil, err
	}
	rendered, err := renderedUnits(units, nil, format)
	if err != nil {
		return nil, err
	}
	folded, err := foldInbandSystem(units, rendered, format)
	if err != nil {
		return nil, err
	}
	body.set(unitsKey(format), arrayValue(folded...))
	return dump(body)
}
