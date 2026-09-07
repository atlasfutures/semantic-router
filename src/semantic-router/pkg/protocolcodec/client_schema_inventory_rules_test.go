package protocolcodec

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// These tests pin the guard itself. The corpus gate is only worth having if an
// unclassified member actually fails it, so the failure is a test rather than
// a claim, and it stays a test after the inventory files move on.

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

// unreviewableInventoryCase is one way a row can be unreviewable, and what the
// complaint has to name. The cases live beside the test rather than inside it:
// the list grows every time a new way to say nothing is found.
type unreviewableInventoryCase struct {
	name      string
	inventory clientSchemaInventory
	want      string
}

// The cases split the way the rules do. A row has an identity -- a path, a
// leg, a version, the beta it attributes the member to -- and it has an answer
// per target. Each half can be unreviewable on its own.
func unreviewableInventoryCases() []unreviewableInventoryCase {
	cases := unreviewableInventoryIdentityCases()
	return append(cases, unreviewableInventoryTargetCases()...)
}

func unreviewableInventoryIdentityCases() []unreviewableInventoryCase {
	return []unreviewableInventoryCase{
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
				Fields: []clientSchemaField{{
					Path: "tools.defer_loading", Leg: "trailer",
					Targets: map[string]string{string(llmprotocol.OpenAIChatV1): string(dispositionDrop)},
				}},
			},
			want: "want request, response or stream",
		},
	}
}

func unreviewableInventoryTargetCases() []unreviewableInventoryCase {
	return []unreviewableInventoryCase{
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
			// A row with no target says a member is known and says nothing
			// about it. The gate keys on path and leg, so such a row marks the
			// member classified while asserting nothing -- which is the state
			// the gate exists to prevent, reached through the inventory
			// instead of through the corpus.
			name: "no target disposition",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields:  []clientSchemaField{{Path: "future_member", Leg: "request"}},
			},
			want: "states no disposition",
		},
		{
			// Stating one foreign target and omitting the other leaves the
			// omitted one reading as carry, per the documented default. For a
			// member the generic carrier handles that is never true: the
			// carrier re-emits only to the format the member arrived on, so
			// every foreign target drops it. The binding loop cannot catch
			// this either -- it iterates the targets a row states.
			name: "request row omits a foreign target",
			inventory: clientSchemaInventory{
				Format:  llmprotocol.AnthropicMessagesV1,
				Version: "2026-09-05",
				Fields: []clientSchemaField{{
					Path: "future_member", Leg: "request",
					Targets: map[string]string{string(llmprotocol.OpenAIChatV1): string(dispositionDrop)},
				}},
			},
			want: "states nothing for openai.responses.v1",
		},
	}
}

func TestClientSchemaInventoryRejectsAnUnreviewableRow(t *testing.T) {
	for _, test := range unreviewableInventoryCases() {
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
			Beta: "task-budgets-2026-03-13",
			Targets: map[string]string{
				string(llmprotocol.OpenAIChatV1):      string(dispositionDrop),
				string(llmprotocol.OpenAIResponsesV1): string(dispositionDrop),
			},
		}},
	}
	if problems := validateClientSchemaInventory(inventory); len(problems) != 0 {
		t.Fatalf("a reviewable inventory was refused: %v", problems)
	}
}

// A member every target carries is still a decision, and it is stated the
// same way as any other: explicitly, on a target. What is refused is a row
// that states nothing at all.
func TestClientSchemaInventoryAcceptsAnExplicitCarry(t *testing.T) {
	inventory := clientSchemaInventory{
		Format:  llmprotocol.AnthropicMessagesV1,
		Version: "2026-09-05",
		Fields: []clientSchemaField{{
			Path: "future_member", Leg: "request",
			Targets: map[string]string{
				string(llmprotocol.OpenAIChatV1):      string(dispositionCarry),
				string(llmprotocol.OpenAIResponsesV1): string(dispositionCarry),
			},
		}},
	}
	if problems := validateClientSchemaInventory(inventory); len(problems) != 0 {
		t.Fatalf("an explicit carry was refused: %v", problems)
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
		`"beta":"task-budgets-2026-03-13","targets":{"openai.chat.v1":"drop",` +
		`"openai.responses.v1":"drop"}}]}`
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
