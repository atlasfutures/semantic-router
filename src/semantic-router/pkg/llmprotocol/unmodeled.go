package llmprotocol

import "encoding/json"

// UnmodeledFields carries top-level members of a request body that the neutral
// contract does not name.
//
// It exists because the Router selects a destination rather than editing a
// payload. When the client and the provider speak the same wire format and
// agree on a member the neutral contract happens not to model, refusing the
// request loses a route that would otherwise work.
//
// The Router never reads it and never routes on it. A codec re-emits it only
// when the target wire format equals Format. Any other target drops it and
// records a dropped diagnostic, because a member of one wire contract carries
// no meaning in another.
type UnmodeledFields struct {
	Format WireFormat
	Fields map[string]json.RawMessage
}

// Len reports how many members the carrier holds. A nil carrier holds none.
func (fields *UnmodeledFields) Len() int {
	if fields == nil {
		return 0
	}
	return len(fields.Fields)
}
