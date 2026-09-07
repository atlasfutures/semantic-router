package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"

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

// A leg is one decode surface, not one direction. A streamed response is
// decoded through a different wire struct from a non-streamed one, so a member
// classified on the response leg says nothing about the same member on a
// chunk: production prunes each against its own struct.
var inventoryLegValues = map[string]struct{}{
	"request":  {},
	"response": {},
	"stream":   {},
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
	// A row keys the gate on path and leg alone, so a row with no target
	// marks a member classified while asserting nothing about it. That is the
	// silence the gate exists to break, reached through the inventory rather
	// than through the corpus. A member every target carries states that
	// explicitly instead of by omission.
	if len(field.Targets) == 0 {
		problems = append(problems, fmt.Sprintf("field %q states no disposition on any target", field.Path))
	}
	if _, ok := inventoryLegValues[field.Leg]; !ok {
		problems = append(problems, fmt.Sprintf("field %q has leg %q, want request, response or stream", field.Path, field.Leg))
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
