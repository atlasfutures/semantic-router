//go:build !windows && cgo

package extproc

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The decide request forwards the client's system, tools and messages
// exactly as received: for captured Claude Code bodies, each field of the
// decide request's request is byte-identical to the client's, not merely
// equal as JSON. The service hashes and projects what it receives, so any
// rewrite on the way (escaping, whitespace, key order) would be invisible to
// a JSON comparison and visible to the service.
func TestPolicyDecideRequestCarriesTheClientBytes(t *testing.T) {
	clients, err := filepath.Glob(filepath.Join(v5Golden, "messages", "*", "client-*.json"))
	if err != nil || len(clients) == 0 {
		t.Fatalf("no captured Messages bodies: %v", err)
	}
	router, fake := v5Router(t, "anthropic")
	for index, path := range clients {
		client, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		// The captures name their own model; the route is the auto alias. Only
		// the model value changes, so every other byte is the capture's.
		client = clientModel.ReplaceAll(client, []byte(`{"model":"auto"`))
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(client, &fields); err != nil {
			t.Fatalf("%s: %v", path, err)
		}
		before := len(fake.received())
		v5Turn(t, router, fake, v5GLMUp, "request-bytes-"+strings.ReplaceAll(path, "/", "-"), string(client))
		received := fake.received()
		if len(received) != before+1 {
			t.Fatalf("%s: %d decide calls, want 1", path, len(received)-before)
		}
		request := received[len(received)-1].Request
		for name, got := range map[string]json.RawMessage{
			"system": request.System, "tools": request.Tools, "messages": request.Messages,
		} {
			want, present := fields[name]
			if !present {
				// An absent field is sent as null, never invented.
				if string(got) != "null" && got != nil {
					t.Errorf("%s: %s absent from the client but sent as %.80s", path, name, got)
				}
				continue
			}
			if !bytes.Equal(got, want) {
				t.Errorf("%s (%d): decide request.%s differs from the client's bytes at offset %d (%d vs %d bytes)",
					path, index, name, firstDifference(got, want), len(got), len(want))
			}
		}
	}
}

// clientModel is a capture's top-level model; a nested "model" key never
// appears in these Messages bodies.
var clientModel = regexp.MustCompile(`^\{"model":"[^"]*"`)

func firstDifference(a, b []byte) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return min(len(a), len(b))
}
