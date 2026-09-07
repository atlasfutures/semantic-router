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

// TestRequestDispositionsMatchTheInventory binds the runtime table to the
// dated inventory. The table decides what a target actually does; the
// inventory is what a reviewer reads and what CI diffs the corpus against.
// A row in one and not the other means the document and the behaviour
// disagree, which is the state the four remediations of the week of
// 2026-09-01 each had to discover by hand.
func TestRequestDispositionsMatchTheInventory(t *testing.T) {
	inventories := loadClientSchemaInventories(t)
	inventory, present := inventories[llmprotocol.AnthropicMessagesV1]
	if !present {
		t.Fatal("the Messages format has no dated inventory")
	}
	rows := make(map[string]clientSchemaField, len(inventory.Fields))
	for _, field := range inventory.Fields {
		if field.Leg == "request" {
			rows[field.Path] = field
		}
	}
	for _, row := range anthropicRequestDispositions {
		assertDispositionRowIsInventoried(t, row, rows)
	}
	// Every inventoried member the table does not name is carried by the
	// generic carrier, and the carrier re-emits only to the format the member
	// arrived on. So a foreign target always drops it, and any other claim in
	// the inventory would be describing behaviour that does not exist.
	for _, field := range inventory.Fields {
		if field.Leg != "request" {
			continue
		}
		if _, named := anthropicRequestDispositionIndex[field.Path]; named {
			continue
		}
		for target, action := range field.Targets {
			if action != string(dispositionDrop) {
				t.Errorf(
					"inventory row %q is carried by the generic carrier, so target %s drops it; the inventory says %q",
					field.Path, target, action,
				)
			}
		}
	}
}

func assertDispositionRowIsInventoried(
	t *testing.T, row requestFieldRow, rows map[string]clientSchemaField,
) {
	t.Helper()
	field, inventoried := rows[row.Path]
	if !inventoried {
		t.Errorf("the disposition table names %q, and no inventory row classifies it", row.Path)
		return
	}
	for target, disposition := range row.Targets {
		stated, present := field.Targets[string(target)]
		if !present {
			t.Errorf("the table gives %q disposition %q on %s, and the inventory states none",
				row.Path, disposition.Action, target)
			continue
		}
		if stated != string(disposition.Action) {
			t.Errorf("the table gives %q disposition %q on %s, and the inventory says %q",
				row.Path, disposition.Action, target, stated)
		}
	}
	for target := range field.Targets {
		if _, stated := row.Targets[llmprotocol.WireFormat(target)]; !stated {
			t.Errorf("the inventory gives %q a disposition on %s, and the table states none",
				row.Path, target)
		}
	}
}

// A member on a message object -- beside role and content, not inside a
// content block -- is the shape the gate could not see at all before CP9u
// taught the carrier to descend into array elements. Now it can, and the
// question is what to call it. The carrier names the element by index, so a
// diagnostic says which message carried the member; an inventory row cannot
// use that name, because the same member on message 4 would need a second row
// and a conversation of hundreds of messages would need hundreds.
//
// So the gate normalises the index away and the inventory classifies the
// member once, wherever it sits. That is also the notation the response leg
// already uses: pruneUnknownProviderFields writes choices[].
func TestGateNamesAMemberOnAMessageObjectWithoutItsIndex(t *testing.T) {
	engine := NewBuiltinEngine()
	entry := clientCorpusEntry{
		ID: "probe", Surface: llmprotocol.AnthropicMessagesV1, Leg: "request",
		Body: json.RawMessage(`{"model":"m","max_tokens":16,"messages":[` +
			`{"role":"user","content":"one"},` +
			`{"role":"assistant","content":"two","brand_new_member":1}]}`),
	}
	observed := observedCorpusFields(t, engine, entry)
	if !slicesContain(observed, "messages[].brand_new_member") {
		t.Fatalf("a member on a message object is named %v, want messages[].brand_new_member", observed)
	}
	inventory, present := loadClientSchemaInventories(t)[llmprotocol.AnthropicMessagesV1]
	if !present {
		t.Fatal("the Messages format has no dated inventory")
	}
	unclassified := unclassifiedFields(inventory, "request", observed)
	if !slicesContain(unclassified, "messages[].brand_new_member") {
		t.Fatalf("a member on a message object passed the gate unclassified: %v", unclassified)
	}
	// The other half: production counts it. A member CI can see and the
	// runtime drops in silence would be the same defect one hop later.
	result, err := engine.TranslateRequest(
		llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, entry.Body, nil,
	)
	if err != nil {
		t.Fatalf("Chat target refused the request: %v", err)
	}
	counted := false
	for _, diagnostic := range result.Diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped &&
			strings.Contains(diagnostic.Field, "brand_new_member") {
			counted = true
		}
	}
	if !counted {
		t.Fatalf("a member on a message object was dropped without a count: %v",
			diagnosticFields(result.Diagnostics))
	}
}

// The same member on two different messages is one row, not two.
func TestGateReportsAMemberOnTwoMessagesOnce(t *testing.T) {
	engine := NewBuiltinEngine()
	entry := clientCorpusEntry{
		ID: "probe", Surface: llmprotocol.AnthropicMessagesV1, Leg: "request",
		Body: json.RawMessage(`{"model":"m","max_tokens":16,"messages":[` +
			`{"role":"user","content":"one","brand_new_member":1},` +
			`{"role":"assistant","content":"two","brand_new_member":2}]}`),
	}
	observed := observedCorpusFields(t, engine, entry)
	count := 0
	for _, path := range observed {
		if path == "messages[].brand_new_member" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("a member on two messages was named %d times, want 1: %v", count, observed)
	}
}

func TestNormalizeArrayIndicesLeavesANamedMemberAlone(t *testing.T) {
	for path, want := range map[string]string{
		"messages.0.brand_new_member":             "messages[].brand_new_member",
		"messages.11.tools.2.future":              "messages[].tools[].future",
		"content.cache_control.evict_on_complete": "content.cache_control.evict_on_complete",
		"output_config.task_budget":               "output_config.task_budget",
	} {
		if got := normalizeArrayIndices(path); got != want {
			t.Fatalf("normalizeArrayIndices(%q) = %q, want %q", path, got, want)
		}
	}
}

func slicesContain(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
