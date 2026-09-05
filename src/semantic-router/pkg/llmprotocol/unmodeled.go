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
	// Children holds the same carrier for a member the contract does name and
	// that is itself a modelled object. A protocol moves by adding a member
	// inside a struct as often as beside it -- cache_control.evict_on_complete
	// and output_config.task_budget are both that shape -- so the carrier has
	// to reach the same depth the wire does.
	Children map[string]*UnmodeledFields
}

// Len reports how many members the carrier holds at its own level. A nil
// carrier holds none.
func (fields *UnmodeledFields) Len() int {
	if fields == nil {
		return 0
	}
	return len(fields.Fields)
}

// Empty reports whether the carrier holds nothing at any depth.
func (fields *UnmodeledFields) Empty() bool {
	if fields == nil {
		return true
	}
	if len(fields.Fields) > 0 {
		return false
	}
	for _, child := range fields.Children {
		if !child.Empty() {
			return false
		}
	}
	return true
}

// UnmodeledBlock is one content block or input item that the neutral contract
// does not name, kept exactly as the client sent it.
//
// It obeys the same rule as UnmodeledFields: the Router never reads it, an
// encoder re-emits it only to Format, and any other target drops it. Type is
// the source discriminator and is recorded for diagnostics alone.
type UnmodeledBlock struct {
	Format WireFormat
	Type   string
	Raw    json.RawMessage
}
