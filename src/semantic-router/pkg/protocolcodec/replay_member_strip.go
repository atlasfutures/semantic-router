package protocolcodec

import (
	"bytes"
	"encoding/json"
)

// withoutTopLevelMember returns body, a JSON object, with its top-level member
// name removed and every other byte as it was, so a replay keeps the
// upstream's field order and spelling. It returns nil when body is not an
// object it can cut cleanly; a nil preserved response is encoded instead of
// replayed.
func withoutTopLevelMember(body []byte, name string) []byte {
	decoder := json.NewDecoder(bytes.NewReader(body))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	previousEnd := decoder.InputOffset()
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
		end := decoder.InputOffset()
		if key != name {
			previousEnd, first = end, false
			continue
		}
		if !first {
			// Cut from the end of the previous value: the comma before this
			// member goes with it.
			return append(append([]byte(nil), body[:previousEnd]...), body[end:]...)
		}
		// The first member: cut it and the comma after it, if any.
		rest := skipJSONSpace(body, int(end))
		if rest < len(body) && body[rest] == ',' {
			rest = skipJSONSpace(body, rest+1)
		}
		return append(append([]byte(nil), body[:previousEnd]...), body[rest:]...)
	}
	return body
}
