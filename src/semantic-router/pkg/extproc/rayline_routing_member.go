package extproc

import (
	"bytes"
	"encoding/json"
	"errors"
)

// errRoutingMemberRepeated is a body naming the routing member twice: which
// one would be read is a decoder's choice, so the body is refused.
var errRoutingMemberRepeated = errors.New("the request body names routing more than once")

// removeTopLevelMember cuts every occurrence of the top-level member name
// out of a JSON object at the byte level, leaving every other byte as the
// client sent it. Re-marshalling a decoded map instead would collapse the
// body's own duplicate members and reorder it, which hides a duplicate from
// the codec's duplicate_json_field check. It returns the body unchanged and
// false when the member is absent, and refuses a repeated one.
func removeTopLevelMember(body []byte, name string) ([]byte, bool, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	open, err := decoder.Token()
	if err != nil || open != json.Delim('{') {
		return nil, false, errors.New("the request body must be a JSON object")
	}
	type span struct{ start, end int64 }
	var cuts []span
	previous := decoder.InputOffset() // just past '{', or past the last value
	first := true
	for decoder.More() {
		start := previous
		key, err := decoder.Token()
		if err != nil {
			return nil, false, err
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, false, err
		}
		end := decoder.InputOffset()
		if key == name {
			if first {
				// The first member owns no leading comma: take the comma
				// that follows it instead, when another member does.
				start = previous
				end = skipWhitespaceAndComma(body, end)
			}
			cuts = append(cuts, span{start, end})
		}
		previous = decoder.InputOffset()
		first = false
	}
	if len(cuts) == 0 {
		return body, false, nil
	}
	if len(cuts) > 1 {
		return nil, false, errRoutingMemberRepeated
	}
	cut := cuts[0]
	stripped := append(append([]byte{}, body[:cut.start]...), body[cut.end:]...)
	if !json.Valid(stripped) {
		return nil, false, errors.New("the request body must be a JSON object")
	}
	return stripped, true, nil
}

// skipWhitespaceAndComma returns the offset past any whitespace and one comma
// after at, or at itself when no comma follows (the member was the only one).
func skipWhitespaceAndComma(body []byte, at int64) int64 {
	index := at
	for index < int64(len(body)) {
		switch body[index] {
		case ' ', '\t', '\n', '\r':
			index++
		case ',':
			return index + 1
		default:
			return at
		}
	}
	return at
}

// stripRoutingMember removes the body's routing member byte for byte; a
// repeated one is refused as routing_invalid.
func stripRoutingMember(body []byte) ([]byte, *routesContractError) {
	stripped, _, err := removeTopLevelMember(body, "routing")
	switch {
	case errors.Is(err, errRoutingMemberRepeated):
		return nil, routesInvalid("routing_invalid", "routing is named more than once.")
	case err != nil:
		return nil, routesInvalid("invalid_body", "The request body must be a JSON object.")
	}
	return stripped, nil
}
