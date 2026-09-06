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

// TestClientSchemaInventoriesAreReviewable keeps each inventory file readable
// on its own terms, before any corpus is compared against it. A row naming an
// undeclared beta classifies nothing an operator can act on.
func TestClientSchemaInventoriesAreReviewable(t *testing.T) {
	inventories := loadClientSchemaInventories(t)
	for _, format := range []llmprotocol.WireFormat{
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1,
	} {
		inventory, present := inventories[format]
		if !present {
			t.Errorf("wire format %q has no dated schema inventory", format)
			continue
		}
		if problems := validateClientSchemaInventory(inventory); len(problems) != 0 {
			t.Errorf("inventory %s@%s is not reviewable:\n  %s",
				format, inventory.Version, strings.Join(problems, "\n  "))
		}
	}
}

// TestCorpusFieldsAreClassified is the CI gate rule 4 asks for and acceptance
// criterion 3 names. A member reaching the cell that no inventory classifies
// is a protocol move nobody has reviewed, and it fails here rather than on a
// user's turn.
func TestCorpusFieldsAreClassified(t *testing.T) {
	corpus := loadClientCorpus(t)
	inventories := loadClientSchemaInventories(t)
	engine := NewBuiltinEngine()
	for _, entry := range corpus.Entries {
		t.Run(entry.ID, func(t *testing.T) {
			assertCorpusEntryIsClassified(t, engine, entry, inventories)
		})
	}
}

func assertCorpusEntryIsClassified(
	t *testing.T,
	engine *Engine,
	entry clientCorpusEntry,
	inventories map[llmprotocol.WireFormat]clientSchemaInventory,
) {
	t.Helper()
	inventory, present := inventories[entry.Surface]
	if !present {
		t.Fatalf("corpus entry %s is on %q, which has no dated inventory", entry.ID, entry.Surface)
	}
	observed := observedCorpusFields(t, engine, entry)
	unclassified := unclassifiedFields(inventory, entry.Leg, observed)
	if len(unclassified) != 0 {
		t.Fatalf(
			"%s %s %s (%s, captured %s) carries %d member(s) that %s@%s does not classify:\n  %s\n"+
				"Add a row to testdata/inventory naming the beta that introduced each, and its "+
				"disposition per target. Production carries the member meanwhile.",
			entry.Client, entry.ClientVersion, entry.Leg, entry.ID, entry.Captured,
			len(unclassified), entry.Surface, inventory.Version, strings.Join(unclassified, "\n  "),
		)
	}
}

// TestCorpusBetasAreDeclared links an entry to the inventory version that has
// to classify it. A capture declaring a beta the inventory has never heard of
// is the earliest signal available that the protocol moved, and it arrives
// before any member of that beta is seen.
func TestCorpusBetasAreDeclared(t *testing.T) {
	corpus := loadClientCorpus(t)
	inventories := loadClientSchemaInventories(t)
	for _, entry := range corpus.Entries {
		inventory, present := inventories[entry.Surface]
		if !present {
			continue
		}
		declared := make(map[string]struct{}, len(inventory.Betas))
		for _, beta := range inventory.Betas {
			declared[beta.Name] = struct{}{}
		}
		for _, beta := range entry.Betas {
			if _, known := declared[beta]; !known {
				t.Errorf("corpus entry %s declares beta %q, which %s@%s does not know",
					entry.ID, beta, entry.Surface, inventory.Version)
			}
		}
	}
}

// TestCorpusEntriesAreAttributed keeps the corpus a record of captures rather
// than a pile of bodies. An entry with no client version and no date cannot
// answer the only question the corpus exists to answer: what produced this,
// and when.
func TestCorpusEntriesAreAttributed(t *testing.T) {
	corpus := loadClientCorpus(t)
	seen := make(map[string]struct{}, len(corpus.Entries))
	for _, entry := range corpus.Entries {
		if _, duplicate := seen[entry.ID]; duplicate {
			t.Errorf("corpus entry id %q appears twice", entry.ID)
		}
		seen[entry.ID] = struct{}{}
		for name, value := range map[string]string{
			"client": entry.Client, "client_version": entry.ClientVersion,
			"captured": entry.Captured, "leg": entry.Leg,
		} {
			if strings.TrimSpace(value) == "" {
				t.Errorf("corpus entry %q states no %s", entry.ID, name)
			}
		}
		if !knownWireFormat(entry.Surface) {
			t.Errorf("corpus entry %q is on unknown surface %q", entry.ID, entry.Surface)
		}
		if (entry.Source == "") == (len(entry.Body) == 0) {
			t.Errorf("corpus entry %q must state exactly one of source and body", entry.ID)
		}
		if !json.Valid(entry.bodyBytes(t)) {
			t.Errorf("corpus entry %q holds invalid JSON", entry.ID)
		}
	}
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
	default:
		t.Fatalf("corpus entry %s has leg %q, want request or response", entry.ID, entry.Leg)
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
	return sortedDistinct(paths)
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
