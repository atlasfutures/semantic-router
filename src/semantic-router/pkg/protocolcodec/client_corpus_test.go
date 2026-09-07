package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The corpus is the input side of rule 4. Each entry is one wire shape that a
// named client version actually produced, on a named date, with the beta set
// it declared. The gate below decodes each one, names every member the wire
// structs do not name, and requires the dated inventory for that format to
// classify each. A member with no row fails CI.
//
// It is deliberately the same walk production runs. A second field walker
// written for the test would drift from the one that decides what a target
// carries, and then the gate would pass on paths the runtime never emits.

const clientCorpusSchemaVersion = "protocolcodec.client-corpus.v1"

type clientCorpus struct {
	SchemaVersion string              `json:"schema_version"`
	Note          string              `json:"note,omitempty"`
	Entries       []clientCorpusEntry `json:"entries"`
}

type clientCorpusEntry struct {
	ID            string                 `json:"id"`
	Client        string                 `json:"client"`
	ClientVersion string                 `json:"client_version"`
	Captured      string                 `json:"captured"`
	Surface       llmprotocol.WireFormat `json:"surface"`
	Leg           string                 `json:"leg"`
	// Betas is the anthropic-beta set the capture declared. It is what turns
	// "a member appeared" into "this beta introduced it", and it is how an
	// entry selects the inventory version that has to classify it.
	Betas []string `json:"betas,omitempty"`
	// Source names a body already on disk, relative to testdata. Body inlines
	// one instead. Exactly one of the two is set.
	Source string          `json:"source,omitempty"`
	Body   json.RawMessage `json:"body,omitempty"`
	Note   string          `json:"note,omitempty"`
}

func loadClientCorpus(t *testing.T) clientCorpus {
	t.Helper()
	body, err := os.ReadFile(filepath.Join("testdata", "client_corpus.v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	var corpus clientCorpus
	if err := decoder.Decode(&corpus); err != nil {
		t.Fatalf("invalid client corpus: %v", err)
	}
	if corpus.SchemaVersion != clientCorpusSchemaVersion {
		t.Fatalf("client corpus schema version = %q, want %q", corpus.SchemaVersion, clientCorpusSchemaVersion)
	}
	if len(corpus.Entries) == 0 {
		t.Fatal("client corpus holds no entries")
	}
	return corpus
}

func (entry clientCorpusEntry) bodyBytes(t *testing.T) []byte {
	t.Helper()
	if entry.Source == "" {
		return entry.Body
	}
	body, err := os.ReadFile(filepath.Join("testdata", filepath.FromSlash(entry.Source)))
	if err != nil {
		t.Fatalf("corpus entry %s: %v", entry.ID, err)
	}
	return body
}

func loadClientSchemaInventories(t *testing.T) map[llmprotocol.WireFormat]clientSchemaInventory {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join("testdata", "inventory", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	inventories := make(map[llmprotocol.WireFormat]clientSchemaInventory, len(paths))
	for _, path := range paths {
		inventory, err := loadClientSchemaInventory(path)
		if err != nil {
			t.Fatal(err)
		}
		if previous, duplicate := inventories[inventory.Format]; duplicate {
			t.Fatalf("format %q is claimed by two inventories, %s and %s", inventory.Format, previous.Version, inventory.Version)
		}
		inventories[inventory.Format] = inventory
	}
	return inventories
}

// observedCorpusFields names every member of one capture that the wire structs
// do not name, plus every member the request disposition table already covers.
// The two together are the extension surface of that body.
func observedCorpusFields(t *testing.T, engine *Engine, entry clientCorpusEntry) []string {
	t.Helper()
	body := entry.bodyBytes(t)
	switch entry.Leg {
	case "request":
		return observedRequestExtensionFields(t, engine, entry, body)
	case "response":
		return observedResponseExtensionFields(t, entry, body)
	case "stream":
		return observedStreamExtensionFields(t, entry, body)
	default:
		t.Fatalf("corpus entry %s has leg %q, want request, response or stream", entry.ID, entry.Leg)
		return nil
	}
}

func observedRequestExtensionFields(
	t *testing.T, engine *Engine, entry clientCorpusEntry, body []byte,
) []string {
	t.Helper()
	request, _, _, err := engine.DecodeRequestForMutation(entry.Surface, body)
	if err != nil {
		// Accept-by-default means a captured client body decodes. A refusal
		// here is rule 1 regressing, not a classification problem.
		t.Fatalf("corpus entry %s was refused at ingress: %v", entry.ID, err)
	}
	paths := unnamedMemberPaths(request.Unmodeled, "")
	for _, tool := range request.Tools {
		paths = append(paths, unnamedMemberPaths(tool.Extensions, "tools.")...)
	}
	for _, instruction := range request.Instructions {
		paths = append(paths, contentExtensionPaths(instruction.Content)...)
	}
	for _, message := range request.Messages {
		paths = append(paths, contentExtensionPaths(message.Content)...)
	}
	paths = append(paths, presentRequestFields(request)...)
	for index, path := range paths {
		paths[index] = normalizeArrayIndices(path)
	}
	return sortedDistinct(paths)
}

// normalizeArrayIndices replaces the index of an array element with "[]", so
// one inventory row classifies a member wherever in an array it sits. The
// carrier names an element by index on purpose -- a drop diagnostic has to say
// which message carried the member -- but a conversation of hundreds of
// messages would otherwise need hundreds of identical rows, and the row for
// message 4 would silently not cover message 5.
//
// A segment made only of digits is an index. No member of either protocol is
// named that way, and the carrier emits a numeric segment for nothing else.
// The notation matches the response leg, where pruneUnknownProviderFields
// already writes choices[].
func normalizeArrayIndices(path string) string {
	segments := strings.Split(path, ".")
	normalized := make([]string, 0, len(segments))
	for _, segment := range segments {
		if isDigits(segment) {
			// The index belongs to the member before it, so it becomes that
			// member's array marker rather than a segment of its own.
			if last := len(normalized) - 1; last >= 0 {
				normalized[last] += "[]"
				continue
			}
		}
		normalized = append(normalized, segment)
	}
	return strings.Join(normalized, ".")
}

func isDigits(segment string) bool {
	if segment == "" {
		return false
	}
	for _, character := range segment {
		if character < '0' || character > '9' {
			return false
		}
	}
	return true
}

func contentExtensionPaths(contents []llmprotocol.Content) []string {
	var paths []string
	for _, content := range contents {
		paths = append(paths, unnamedMemberPaths(content.Extensions, "content.")...)
		if content.Kind == llmprotocol.ContentUnmodeled && content.Unmodeled != nil {
			// A block the contract cannot name at all is named by its own
			// discriminator, which is the name the disposition table uses for
			// one: content.document is the row CP9u added for a text document.
			paths = append(paths, "content."+content.Unmodeled.Type)
		}
		if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
			paths = append(paths, contentExtensionPaths(content.ToolResult.Content)...)
		}
	}
	return paths
}

// observedResponseExtensionFields names the members an upstream body carries
// that the response wire struct does not. It runs the same prune the response
// leg runs in production, so the paths are the ones
// upstream_response_field_dropped reports.
func observedResponseExtensionFields(t *testing.T, entry clientCorpusEntry, body []byte) []string {
	t.Helper()
	var wire reflect.Type
	switch entry.Surface {
	case llmprotocol.OpenAIChatV1:
		wire = reflect.TypeOf(chatResponseWire{})
	case llmprotocol.OpenAIResponsesV1:
		wire = reflect.TypeOf(responsesResponseWire{})
	case llmprotocol.AnthropicMessagesV1:
		wire = reflect.TypeOf(anthropicResponseWire{})
	default:
		t.Fatalf("corpus entry %s is on unknown surface %q", entry.ID, entry.Surface)
	}
	_, dropped := pruneUnknownProviderFields(body, wire)
	return sortedDistinct(dropped)
}

// observedStreamExtensionFields names the members the frames of one streamed
// response carry that the chunk wire does not. A streamed response is decoded
// through a different struct from a non-streamed one -- stream_chat.go decodes
// each frame into chatChunkWire -- so the response leg says nothing about it,
// and a member added to a chunk was pruned in production with nothing in CI
// able to see it.
func observedStreamExtensionFields(t *testing.T, entry clientCorpusEntry, body []byte) []string {
	t.Helper()
	var wire reflect.Type
	switch entry.Surface {
	case llmprotocol.OpenAIChatV1:
		wire = reflect.TypeOf(chatChunkWire{})
	case llmprotocol.OpenAIResponsesV1:
		wire = reflect.TypeOf(responsesEventWire{})
	case llmprotocol.AnthropicMessagesV1:
		wire = reflect.TypeOf(anthropicEventWire{})
	default:
		t.Fatalf("corpus entry %s is on unknown surface %q", entry.ID, entry.Surface)
	}
	payloads := sseEventPayloads(t, entry, body)
	if len(payloads) == 0 {
		t.Fatalf("corpus entry %s holds no stream events", entry.ID)
	}
	var paths []string
	for _, payload := range payloads {
		_, dropped := pruneUnknownProviderFields(payload, wire)
		paths = append(paths, dropped...)
	}
	return sortedDistinct(paths)
}

// sseEventPayloads returns the JSON document of every event of a recorded
// stream, framed exactly the way production frames a live one: the same
// sseFramer splits the bytes into events, and the same parseSSEFrameAtPosition
// joins the data lines of each.
//
// The framing has to be shared, not approximated. SSE lets one event carry its
// payload over several data lines joined by newlines, and a gate that read
// each line as its own frame would find neither half parseable and skip both.
// A mixed file -- one multi-line event among ordinary ones -- would then lose
// that event's members while still looking covered.
//
// An unreadable payload fails the test rather than being skipped. A capture
// the gate cannot parse teaches nothing, and skipping it is how a whole
// surface goes quietly uncovered.
func sseEventPayloads(t *testing.T, entry clientCorpusEntry, body []byte) [][]byte {
	t.Helper()
	limit := llmprotocol.DefaultPolicy().Limits.SSEFrameBytes
	framer := newSSEFramer(limit)
	frames, err := framer.Push(body)
	if err != nil {
		t.Fatalf("corpus entry %s could not be framed: %v", entry.ID, err)
	}
	trailing, err := framer.Finalize()
	if err != nil {
		t.Fatalf("corpus entry %s has an unfinished frame: %v", entry.ID, err)
	}
	frames = append(frames, trailing...)

	payloads := make([][]byte, 0, len(frames))
	for index, frame := range frames {
		parsed, parseErr := parseSSEFrameAtPosition(frame, limit, index == 0)
		if parseErr != nil {
			t.Fatalf("corpus entry %s frame %d is not a readable event: %v", entry.ID, index, parseErr)
		}
		if !parsed.HasData {
			continue
		}
		payload := bytes.TrimSpace(parsed.Data)
		// The terminal sentinel is not a document any wire struct decodes.
		if len(payload) == 0 || string(payload) == "[DONE]" {
			continue
		}
		if !json.Valid(payload) {
			t.Fatalf("corpus entry %s frame %d carries a payload that is not JSON: %s",
				entry.ID, index, payload)
		}
		payloads = append(payloads, payload)
	}
	return payloads
}

func sortedDistinct(paths []string) []string {
	seen := make(map[string]struct{}, len(paths))
	distinct := make([]string, 0, len(paths))
	for _, path := range paths {
		if _, repeated := seen[path]; repeated {
			continue
		}
		seen[path] = struct{}{}
		distinct = append(distinct, path)
	}
	sort.Strings(distinct)
	return distinct
}
