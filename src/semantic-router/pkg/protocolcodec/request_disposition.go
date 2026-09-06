package protocolcodec

import (
	"encoding/json"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// One disposition table per target format, in place of a branch per field.
//
// Every request member the neutral contract does name, and that some target
// cannot express, has a row here. A row says what each target does with it:
// carry it byte for byte, drop it and count the path, or transform it into the
// target's own spelling. A target absent from a row's map carries the member,
// which is why the Messages target appears in no map: it carries everything.
//
// The members no contract names are not here and cannot be: they are unknown
// by definition, and the carrier in unnamed_members.go handles them. This
// table is for the members we do know and used to refuse. Four of those cost
// an investigation each in the week of 2026-09-01, and the refusals were
// hand-rolled branches with no field path, which is why locating them took a
// log join.
//
// Beta records which anthropic-beta value introduced the member. Nothing reads
// it yet. It is here because CP9v's dated schema inventories key on exactly
// that, and filling the column later against a table is a smaller job than
// reconstructing it from branches.

type dispositionAction string

const (
	dispositionCarry     dispositionAction = "carry"
	dispositionDrop      dispositionAction = "drop"
	dispositionTransform dispositionAction = "transform"
)

type targetDisposition struct {
	Action dispositionAction
	Reason string
}

type requestFieldRow struct {
	// Path is the field path exactly as a diagnostic reports it.
	Path string
	// Beta is the anthropic-beta value that introduced the member, empty for
	// the base contract and empty where it is not yet established. CP9v fills
	// it from the dated inventories.
	Beta string
	// Targets holds the disposition per target format. A format that is absent
	// carries the member.
	Targets map[llmprotocol.WireFormat]targetDisposition
}

const (
	fieldContentCitations    = "content.citations"
	fieldContentCacheControl = "content.cache_control"
	fieldContentCaller       = "content.caller"
	fieldContentDocumentText = "content.document"
	fieldToolsType           = "tools.type"
)

var anthropicRequestDispositions = []requestFieldRow{
	{
		// Claude Code echoes the citations of a web-search or document answer
		// back in history on every later turn. Refusing them failed two
		// operator turns on the dev cell on 2026-09-05.
		Path: fieldContentCitations,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "a Chat content part carries no block citations",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionDrop,
				Reason: "a Responses content part carries no block citations",
			},
		},
	},
	{
		// The breakpoint a client marks on a tool block. Refusing it answered
		// 500 to every follow-up turn of a tool-using session on 2026-09-04,
		// and the proxy served those turns from another provider.
		Path: fieldContentCacheControl,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "a Chat tool call carries no cache breakpoint",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionDrop,
				Reason: "a Responses tool call carries no cache breakpoint",
			},
		},
	},
	{
		// Programmatic tool calling names who issued the call. It identifies
		// the issuer rather than changing what the model is asked.
		Path: fieldContentCaller,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "a Chat tool call cannot name its issuer",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionDrop,
				Reason: "a Responses tool call cannot name its issuer",
			},
		},
	},
	{
		// A document whose source is text is the one row where dropping is
		// worse than every alternative: the block holds the text the turn is
		// about, and a Chat target that drops it sends a question with its
		// subject removed. The provider answers, the prompt token count does
		// not rise, and nothing in the response says what was lost.
		Path: fieldContentDocumentText,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionTransform,
				Reason: "a text document becomes a Chat text part",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionTransform,
				Reason: "a text document becomes a Responses text part",
			},
		},
	},
	{
		// A tool the source API runs itself. There is no function for the
		// caller to implement and no shape a Chat tool array can hold, so the
		// declaration is dropped and the capability gate routes the turn to an
		// arm that has the tool.
		Path: fieldToolsType,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "Chat Completions has no server-run tool",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionDrop,
				Reason: "Responses has no server-run tool of this kind",
			},
		},
	},
}

var anthropicRequestDispositionIndex = indexRequestDispositions(anthropicRequestDispositions)

func indexRequestDispositions(rows []requestFieldRow) map[string]requestFieldRow {
	index := make(map[string]requestFieldRow, len(rows))
	for _, row := range rows {
		index[row.Path] = row
	}
	return index
}

// dispositionFor answers what one target does with one member. A member with
// no row, and a member whose row names no disposition for this target, is
// carried: the table lists what a target cannot express, not what it can.
func dispositionFor(path string, target llmprotocol.WireFormat) targetDisposition {
	row, known := anthropicRequestDispositionIndex[path]
	if !known {
		return targetDisposition{Action: dispositionCarry}
	}
	disposition, stated := row.Targets[target]
	if !stated {
		return targetDisposition{Action: dispositionCarry}
	}
	return disposition
}

// appendRequestDispositions counts every member of this request that the
// target cannot express. It is the one call an encoder makes: the walk finds
// what the request holds and the table says what happens to it.
func appendRequestDispositions(
	diagnostics *llmprotocol.Diagnostics,
	request llmprotocol.Request,
	target llmprotocol.WireFormat,
	policy llmprotocol.Policy,
) {
	for _, path := range presentRequestFields(request) {
		disposition := dispositionFor(path, target)
		switch disposition.Action {
		case dispositionDrop:
			appendPresentationDrop(
				diagnostics, policy, request.Trusted.SourceFormat, target,
				path, disposition.Reason,
			)
		case dispositionTransform:
			*diagnostics = appendDiagnostics(*diagnostics, llmprotocol.Diagnostics{{
				Source: request.Trusted.SourceFormat, Target: target, Field: path,
				Action: llmprotocol.DiagnosticApproximated, Reason: disposition.Reason,
			}}, policy.Limits.Diagnostics)
		}
	}
}

// presentRequestFields names the table members this request actually holds,
// one entry per occurrence so a request with three cited blocks counts three
// drops. The order is sorted so a diagnostics list is stable in a golden.
func presentRequestFields(request llmprotocol.Request) []string {
	var paths []string
	for _, tool := range request.Tools {
		if tool.ServerTool() {
			paths = append(paths, fieldToolsType)
		}
	}
	for _, instruction := range request.Instructions {
		paths = append(paths, presentContentFields(instruction.Content)...)
	}
	for _, message := range request.Messages {
		paths = append(paths, presentContentFields(message.Content)...)
	}
	sort.Strings(paths)
	return paths
}

func presentContentFields(contents []llmprotocol.Content) []string {
	var paths []string
	for _, content := range contents {
		if len(content.CitationsRaw) > 0 {
			paths = append(paths, fieldContentCitations)
		}
		if _, isDocument := carriedDocumentText(content); isDocument {
			paths = append(paths, fieldContentDocumentText)
		}
		if content.Kind != llmprotocol.ContentToolCall && content.Kind != llmprotocol.ContentToolResult {
			continue
		}
		if content.Cache != nil {
			paths = append(paths, fieldContentCacheControl)
		}
		if content.Kind == llmprotocol.ContentToolCall && content.ToolCall != nil &&
			hasJSONValue(content.ToolCall.Caller) {
			paths = append(paths, fieldContentCaller)
		}
		if content.Kind == llmprotocol.ContentToolResult && content.ToolResult != nil {
			paths = append(paths, presentContentFields(content.ToolResult.Content)...)
		}
	}
	return paths
}

// anthropicDocumentTextWire is the shape of a document whose source is text.
// The neutral contract has no media reference for one, so the block is carried
// whole; this reads the two members a text part needs out of those bytes.
type anthropicDocumentTextWire struct {
	Title  string `json:"title"`
	Source struct {
		Type string `json:"type"`
		Data string `json:"data"`
	} `json:"source"`
}

// carriedDocumentText reads the text a carried document block holds, with its
// title when it states one. It answers false for every other carried block, so
// a target that cannot name the block still drops it.
func carriedDocumentText(content llmprotocol.Content) (string, bool) {
	carried := content.Unmodeled
	if content.Kind != llmprotocol.ContentUnmodeled || carried == nil ||
		carried.Format != llmprotocol.AnthropicMessagesV1 || carried.Type != "document" {
		return "", false
	}
	var document anthropicDocumentTextWire
	if err := json.Unmarshal(carried.Raw, &document); err != nil {
		return "", false
	}
	if document.Source.Type != "text" || document.Source.Data == "" {
		return "", false
	}
	if document.Title == "" {
		return document.Source.Data, true
	}
	return document.Title + "\n\n" + document.Source.Data, true
}

func hasJSONValue(raw json.RawMessage) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && trimmed != "null"
}

// appendUnchoosableToolChoiceDrop counts a tool choice the target has no tools
// to apply. Ingress defaults the choice to automatic whenever tools are
// present, and the table drops what a target cannot express, so a turn whose
// only tools are server tools would otherwise send a choice with nothing to
// choose from. Chat Completions rejects that, which fails the turn at the
// provider instead of falling back. The choice is dropped and counted, like
// every other member a target cannot express.
func appendUnchoosableToolChoiceDrop(
	diagnostics *llmprotocol.Diagnostics,
	policy llmprotocol.Policy,
	request llmprotocol.Request,
	target llmprotocol.WireFormat,
	encodedTools int,
) {
	if encodedTools > 0 || request.ToolChoice.Mode == "" {
		return
	}
	appendPresentationDrop(
		diagnostics, policy, request.Trusted.SourceFormat, target, "tool_choice",
		"the target encoded no tools for the choice to apply to",
	)
}
