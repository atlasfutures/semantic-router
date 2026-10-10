package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
)

// withoutTopLevelMember returns body, a JSON object, with every top-level
// member whose key matches name removed and every other byte as it was, so a
// replay keeps the upstream's field order and spelling. Keys match as
// encoding/json decodes them, ignoring case, so no spelling the decoder read
// survives. It returns nil when body is not an object it can cut cleanly; a
// nil preserved response is encoded instead of replayed.
func withoutTopLevelMember(body []byte, name string) []byte {
	decoder := json.NewDecoder(bytes.NewReader(body))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	type span struct{ start, end int }
	var cuts []span
	previousEnd := int(decoder.InputOffset())
	first := true
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil
		}
		key, ok := token.(string)
		if !ok {
			return nil
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil
		}
		end := int(decoder.InputOffset())
		switch {
		case !strings.EqualFold(key, name):
			first = false
		case !first:
			// Cut from the end of the previous value: the comma before this
			// member goes with it.
			cuts = append(cuts, span{previousEnd, end})
		default:
			// The first member: cut it and the comma after it, if any.
			rest := skipJSONSpace(body, end)
			if rest < len(body) && body[rest] == ',' {
				rest = skipJSONSpace(body, rest+1)
			}
			cuts = append(cuts, span{previousEnd, rest})
			end = previousEnd
		}
		previousEnd = end
	}
	if len(cuts) == 0 {
		return body
	}
	out := make([]byte, 0, len(body))
	at := 0
	for _, cut := range cuts {
		out = append(out, body[at:cut.start]...)
		at = cut.end
	}
	return append(out, body[at:]...)
}
