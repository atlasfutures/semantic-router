package llmprotocol

import (
	"errors"
	"strings"
	"testing"
)

// decodeCause is the shape a decode failure wraps: a member name, a Go type and
// a JSON offset, never a decoded value. Unwrap() keeps it, Error() does not.
var decodeCause = errors.New("json: cannot unmarshal string into field usage.prompt_tokens of type int")

// TestProtocolErrorMessageIncludesCause is the ask. Error() formats only Code
// and Message, so a logged protocol error loses the part naming what failed.
func TestProtocolErrorMessageIncludesCause(t *testing.T) {
	err := NewError(ErrorUpstreamUnavailable, "invalid_upstream_response", "upstream response is not decodable", decodeCause)
	if got := err.Error(); !strings.Contains(got, decodeCause.Error()) {
		t.Errorf("Error() = %q, want it to name the cause it already carries", got)
	}
}

// TestProtocolErrorMessageDropsCauseToday pins the present behaviour so a
// change to it shows up in the diff rather than only in the test above.
func TestProtocolErrorMessageDropsCauseToday(t *testing.T) {
	err := NewError(ErrorUpstreamUnavailable, "invalid_upstream_response", "upstream response is not decodable", decodeCause)
	if got := err.Error(); got != "invalid_upstream_response: upstream response is not decodable" {
		t.Errorf("Error() = %q: the recorded behaviour has changed", got)
	}
	if !errors.Is(err, decodeCause) {
		t.Errorf("the dropped cause is no longer reachable through Unwrap()")
	}
}
