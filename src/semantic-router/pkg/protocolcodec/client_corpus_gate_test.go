package protocolcodec

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The gate itself. The file beside this one says what a corpus entry is and
// how its extension surface is read; this one says what CI does about it.

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
		// A recorded stream is SSE, not one JSON document. Its frames are
		// checked where they are read.
		if entry.Leg != "stream" && !json.Valid(entry.bodyBytes(t)) {
			t.Errorf("corpus entry %q holds invalid JSON", entry.ID)
		}
	}
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

// A member added to a streamed chunk is pruned in production against
// chatChunkWire, which the response leg never touches. Before the stream leg
// existed the gate could not see one at all: every response-leg row is about
// chatResponseWire, and the two structs share a name for almost nothing that
// matters here -- a chunk carries its reasoning trace on delta, a complete
// response on message.
func TestGateSeesAMemberOnAStreamChunk(t *testing.T) {
	stream := []byte(": OPENROUTER PROCESSING\n\n" +
		`data: {"id":"gen-fixture-1","object":"chat.completion.chunk","model":"m",` +
		`"choices":[{"index":0,"delta":{"content":"hi"},"brand_new_member":1}]}` + "\n\n" +
		"data: [DONE]\n\n")
	entry := clientCorpusEntry{ID: "probe", Surface: llmprotocol.OpenAIChatV1, Leg: "stream"}
	observed := observedStreamExtensionFields(t, entry, stream)
	if !slicesContain(observed, "choices[].brand_new_member") {
		t.Fatalf("a member on a stream chunk is named %v, want choices[].brand_new_member", observed)
	}
	inventory, present := loadClientSchemaInventories(t)[llmprotocol.OpenAIChatV1]
	if !present {
		t.Fatal("the Chat format has no dated inventory")
	}
	if unclassified := unclassifiedFields(inventory, "stream", observed); !slicesContain(
		unclassified, "choices[].brand_new_member",
	) {
		t.Fatalf("a member on a stream chunk passed the gate unclassified: %v", unclassified)
	}
}

// A stream-leg row does not classify the same member on the response leg, and
// the reverse. The two are different structs, so a member known on one is an
// open question on the other.
func TestStreamAndResponseLegsAreSeparate(t *testing.T) {
	inventory, present := loadClientSchemaInventories(t)[llmprotocol.OpenAIChatV1]
	if !present {
		t.Fatal("the Chat format has no dated inventory")
	}
	streamed := "choices[].delta.reasoning_details"
	if got := unclassifiedFields(inventory, "stream", []string{streamed}); len(got) != 0 {
		t.Fatalf("the stream leg does not classify %q: %v", streamed, got)
	}
	if got := unclassifiedFields(inventory, "response", []string{streamed}); len(got) != 1 {
		t.Fatalf("a stream-leg row classified %q on the response leg", streamed)
	}
}
