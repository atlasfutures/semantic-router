package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A versioned schema inventory is the reviewable half of accept-by-default.
//
// Production carries a member no wire struct names, at any depth, and counts
// the drop. That is rule 1 and it landed in CP9u. It leaves one question with
// no owner: when a member appears that nobody has seen before, who notices?
// Nothing in production may refuse it, because the gateway does not replay a
// body-level 400 and every refusal is a user-visible failed turn.
//
// So the strictness lives here instead. Each inventory is dated, names the
// wire format it describes, and lists every extension member observed on that
// format with the anthropic-beta value that introduced it and what each target
// does with it. A captured corpus is diffed against the inventory, and a
// member the inventory does not classify fails CI.
//
// The dated identity is the part that matters. "cache_control gained a member"
// is an investigation; "prompt-caching-evict-2026-05-12 introduced
// content.cache_control.evict_on_complete, dropped by both OpenAI targets" is
// a table lookup.
//
// This inventory does not restate the modelled wire fields. Those are already
// closed by TestOfficialRequestFieldInventoriesAreClosed and the nested JSON
// inventories, both of which reflect over the Go structs. Reflection is
// structurally blind to exactly the class that caused every incident of the
// week of 2026-09-01: a member that reaches the cell and that no struct names.
// That class is what this file covers.

// inventoryDispositionValues are the three dispositions rule 2 defines. They
// are the same words the request disposition table uses, so a row here and a
// row there cannot disagree about what a target does.
var inventoryDispositionValues = map[string]struct{}{
	string(dispositionCarry):     {},
	string(dispositionDrop):      {},
	string(dispositionTransform): {},
}

var inventoryLegValues = map[string]struct{}{
	"request":  {},
	"response": {},
}

var inventoryVersionPattern = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}$`)

// clientSchemaInventory is one dated inventory for one wire format.
type clientSchemaInventory struct {
	// Format is the wire contract this inventory describes.
	Format llmprotocol.WireFormat `json:"format"`
	// Version is the date the inventory was last established against a real
	// client or provider. It is not a protocol version the wire carries: the
	// Anthropic base version does not move, and the OpenAI formats publish no
	// version at all, so the date is the only honest identity available.
	Version string `json:"version"`
	// BaseVersion is the value the wire does carry, where it carries one.
	BaseVersion string `json:"base_version,omitempty"`
	// Betas declares every anthropic-beta value a Fields row may attribute a
	// member to. A row naming a beta absent from here is a review error: the
	// point of the column is that the beta is named once, with its state.
	Betas []clientSchemaBeta `json:"betas,omitempty"`
	// Fields classifies every extension member observed on this format.
	Fields []clientSchemaField `json:"fields"`
	// Note says what the inventory does not cover. An inventory that
	// classifies nothing has to state why, because "no member has been seen"
	// and "no capture has been made" look identical in an empty file and mean
	// opposite things.
	Note string `json:"note,omitempty"`
}

type clientSchemaBeta struct {
	Name string `json:"name"`
	// State says how the beta reaches a request. "client" means a client
	// upgrade introduces it; "server-gated" means it can switch on with no
	// client upgrade and no warning, which is why the corpus can go stale
	// without anyone changing anything.
	State string `json:"state,omitempty"`
	Note  string `json:"note,omitempty"`
}

type clientSchemaField struct {
	// Path is the member's dotted path in the same language the runtime
	// diagnostics use: no array indices, and a scope prefix of "content." or
	// "tools." where the member sits inside one of those.
	Path string `json:"path"`
	// Leg is "request" or "response".
	Leg string `json:"leg"`
	// Beta is the anthropic-beta value that introduced the member, empty for
	// the base contract.
	Beta string `json:"beta,omitempty"`
	// FirstSeen is the date the member was first observed in the corpus.
	FirstSeen string `json:"first_seen,omitempty"`
	// Targets says what each target format does with the member. A target
	// absent from the map carries it, which is the same default the request
	// disposition table uses.
	Targets map[string]string `json:"targets,omitempty"`
	Note    string            `json:"note,omitempty"`
}

// unclassifiedFields names the observed members this inventory does not
// classify. An empty answer is what CI requires: a member reaching the cell
// with no row is a protocol move nobody has reviewed.
func unclassifiedFields(inventory clientSchemaInventory, leg string, observed []string) []string {
	classified := make(map[string]struct{}, len(inventory.Fields))
	for _, field := range inventory.Fields {
		if field.Leg == leg {
			classified[field.Path] = struct{}{}
		}
	}
	seen := make(map[string]struct{}, len(observed))
	unclassified := make([]string, 0)
	for _, path := range observed {
		if _, known := classified[path]; known {
			continue
		}
		if _, repeated := seen[path]; repeated {
			continue
		}
		seen[path] = struct{}{}
		unclassified = append(unclassified, path)
	}
	sort.Strings(unclassified)
	return unclassified
}

// validateClientSchemaInventory reports every way one inventory file fails to
// be reviewable. It answers a list rather than an error so a reviewer sees
// every problem in one run instead of the first one.
func validateClientSchemaInventory(inventory clientSchemaInventory) []string {
	problems := make([]string, 0)
	problems = append(problems, validateInventoryIdentity(inventory)...)
	declared := make(map[string]struct{}, len(inventory.Betas))
	for _, beta := range inventory.Betas {
		if strings.TrimSpace(beta.Name) == "" {
			problems = append(problems, "a declared beta has no name")
			continue
		}
		if _, duplicate := declared[beta.Name]; duplicate {
			problems = append(problems, fmt.Sprintf("beta %q is declared twice", beta.Name))
		}
		declared[beta.Name] = struct{}{}
	}
	seen := make(map[string]struct{}, len(inventory.Fields))
	for _, field := range inventory.Fields {
		problems = append(problems, validateInventoryField(field, declared, seen)...)
	}
	return problems
}

func validateInventoryIdentity(inventory clientSchemaInventory) []string {
	problems := make([]string, 0)
	if !knownWireFormat(inventory.Format) {
		problems = append(problems, fmt.Sprintf("format %q is not a wire contract", inventory.Format))
	}
	if !inventoryVersionPattern.MatchString(inventory.Version) {
		problems = append(problems, fmt.Sprintf("version %q is not a YYYY-MM-DD date", inventory.Version))
	}
	if len(inventory.Fields) == 0 && strings.TrimSpace(inventory.Note) == "" {
		problems = append(problems, "inventory classifies no members and states no note saying why")
	}
	return problems
}

func validateInventoryField(
	field clientSchemaField,
	declared, seen map[string]struct{},
) []string {
	problems := make([]string, 0)
	if strings.TrimSpace(field.Path) == "" {
		return append(problems, "a field row has no path")
	}
	key := field.Leg + " " + field.Path
	if _, duplicate := seen[key]; duplicate {
		problems = append(problems, fmt.Sprintf("field %q appears twice on the %s leg", field.Path, field.Leg))
	}
	seen[key] = struct{}{}
	if _, ok := inventoryLegValues[field.Leg]; !ok {
		problems = append(problems, fmt.Sprintf("field %q has leg %q, want request or response", field.Path, field.Leg))
	}
	if field.Beta != "" {
		if _, ok := declared[field.Beta]; !ok {
			problems = append(problems, fmt.Sprintf("field %q names undeclared beta %q", field.Path, field.Beta))
		}
	}
	for target, action := range field.Targets {
		if !knownWireFormat(llmprotocol.WireFormat(target)) {
			problems = append(problems, fmt.Sprintf("field %q names unknown target %q", field.Path, target))
		}
		if _, ok := inventoryDispositionValues[action]; !ok {
			problems = append(problems, fmt.Sprintf("field %q gives target %q disposition %q", field.Path, target, action))
		}
	}
	return problems
}

func knownWireFormat(format llmprotocol.WireFormat) bool {
	switch format {
	case llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1:
		return true
	default:
		return false
	}
}

// loadClientSchemaInventory reads one inventory file.
func loadClientSchemaInventory(path string) (clientSchemaInventory, error) {
	var inventory clientSchemaInventory
	body, err := os.ReadFile(path)
	if err != nil {
		return inventory, err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&inventory); err != nil {
		return inventory, fmt.Errorf("invalid inventory %s: %w", path, err)
	}
	return inventory, nil
}

// The tests below pin the guard itself. The corpus gate is only worth having
// if an unclassified member actually fails it, so the failure is a test rather
// than a claim.

func TestUnclassifiedFieldsNamesAMemberWithNoRow(t *testing.T) {
	inventory := clientSchemaInventory{
		Format:  llmprotocol.AnthropicMessagesV1,
		Version: "2026-09-05",
		Fields: []clientSchemaField{
			{Path: "content.cache_control.evict_on_complete", Leg: "request"},
		},
	}
	observed := []string{
		"content.cache_control.evict_on_complete",
		"content.cache_control.retain_until",
	}
	got := unclassifiedFields(inventory, "request", observed)
	want := []string{"content.cache_control.retain_until"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unclassified members = %v, want %v", got, want)
	}
}

func TestUnclassifiedFieldsAcceptsAClassifiedMember(t *testing.T) {
	inventory := clientSchemaInventory{
		Format:  llmprotocol.AnthropicMessagesV1,
		Version: "2026-09-05",
		Fields: []clientSchemaField{
			{Path: "tools.defer_loading", Leg: "request"},
		},
	}
	if got := unclassifiedFields(inventory, "request", []string{"tools.defer_loading"}); len(got) != 0 {
		t.Fatalf("a classified member was reported unclassified: %v", got)
	}
}

// A row on the response leg does not classify a request member. The two legs
// move independently: CP9q's provider field and CP9t's citations arrived
// months apart on opposite sides of the same translation.
func TestUnclassifiedFieldsSeparatesTheTwoLegs(t *testing.T) {
	inventory := clientSchemaInventory{
		Format:  llmprotocol.AnthropicMessagesV1,
		Version: "2026-09-05",
		Fields: []clientSchemaField{
			{Path: "usage.compute_units", Leg: "response"},
		},
	}
	got := unclassifiedFields(inventory, "request", []string{"usage.compute_units"})
	want := []string{"usage.compute_units"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("a response row classified a request member: got %v, want %v", got, want)
	}
}

func TestUnclassifiedFieldsReportsEachMemberOnce(t *testing.T) {
	inventory := clientSchemaInventory{Format: llmprotocol.AnthropicMessagesV1, Version: "2026-09-05"}
	observed := []string{"content.future", "content.future", "content.future"}
	got := unclassifiedFields(inventory, "request", observed)
	if len(got) != 1 {
		t.Fatalf("a member present three times was reported %d times: %v", len(got), got)
	}
}

func TestClientSchemaInventoryRejectsAnUnreviewableRow(t *testing.T) {
	tests := []struct {
		name      string
		inventory clientSchemaInventory
		want      string
	}{
		{
			name: "undeclared beta",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields: []clientSchemaField{
					{Path: "output_config.task_budget", Leg: "request", Beta: "task-budgets-2026-03-13"},
				},
			},
			want: "undeclared beta",
		},
		{
			name: "duplicate path",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields: []clientSchemaField{
					{Path: "tools.defer_loading", Leg: "request"},
					{Path: "tools.defer_loading", Leg: "request"},
				},
			},
			want: "appears twice",
		},
		{
			name: "unknown disposition",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields: []clientSchemaField{
					{
						Path: "tools.defer_loading", Leg: "request",
						Targets: map[string]string{string(llmprotocol.OpenAIChatV1): "ignore"},
					},
				},
			},
			want: "disposition",
		},
		{
			name: "unknown target",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields: []clientSchemaField{
					{Path: "tools.defer_loading", Leg: "request", Targets: map[string]string{"openai.batch.v1": "drop"}},
				},
			},
			want: "unknown target",
		},
		{
			name: "undated version",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "v1",
				Fields:  []clientSchemaField{{Path: "tools.defer_loading", Leg: "request"}},
			},
			want: "YYYY-MM-DD",
		},
		{
			name: "unknown leg",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields:  []clientSchemaField{{Path: "tools.defer_loading", Leg: "stream"}},
			},
			want: "want request or response",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			problems := validateClientSchemaInventory(test.inventory)
			if !strings.Contains(strings.Join(problems, "\n"), test.want) {
				t.Fatalf("inventory problems = %v, want one naming %q", problems, test.want)
			}
		})
	}
}

func TestClientSchemaInventoryAcceptsAReviewableRow(t *testing.T) {
	inventory := clientSchemaInventory{
		Format:      llmprotocol.AnthropicMessagesV1,
		Version:     "2026-09-05",
		BaseVersion: "2023-06-01",
		Betas:       []clientSchemaBeta{{Name: "task-budgets-2026-03-13", State: "client"}},
		Fields: []clientSchemaField{{
			Path: "output_config.task_budget", Leg: "request",
			Beta:    "task-budgets-2026-03-13",
			Targets: map[string]string{string(llmprotocol.OpenAIChatV1): string(dispositionDrop)},
		}},
	}
	if problems := validateClientSchemaInventory(inventory); len(problems) != 0 {
		t.Fatalf("a reviewable inventory was refused: %v", problems)
	}
}

// A misspelled member in an inventory file must fail the read rather than be
// ignored. An inventory that silently drops "betas" spelled "beta" would
// classify nothing and still report no problems.
func TestLoadClientSchemaInventoryRefusesAnUnknownMember(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "typo.json")
	body := `{"format":"anthropic.messages.v1","version":"2026-09-05","beta":[],"fields":[]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := loadClientSchemaInventory(path); err == nil {
		t.Fatal("an inventory with a misspelled member was read without complaint")
	}
}

func TestLoadClientSchemaInventoryReadsAReviewableFile(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "inventory.json")
	body := `{"format":"anthropic.messages.v1","version":"2026-09-05",` +
		`"betas":[{"name":"task-budgets-2026-03-13"}],` +
		`"fields":[{"path":"output_config.task_budget","leg":"request",` +
		`"beta":"task-budgets-2026-03-13","targets":{"openai.chat.v1":"drop"}}]}`
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	inventory, err := loadClientSchemaInventory(path)
	if err != nil {
		t.Fatal(err)
	}
	if problems := validateClientSchemaInventory(inventory); len(problems) != 0 {
		t.Fatalf("a reviewable inventory file was refused: %v", problems)
	}
	if got := unclassifiedFields(inventory, "request", []string{"output_config.task_budget"}); len(got) != 0 {
		t.Fatalf("a file-loaded row classified nothing: %v", got)
	}
}
