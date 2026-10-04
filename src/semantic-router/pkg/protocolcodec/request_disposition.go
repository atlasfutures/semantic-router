package protocolcodec

import (
	"encoding/json"
	"regexp"
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
// table is for the members we do know and used to refuse. The reasoning rows
// are the exception to "Messages carries everything": see
// reasoningProvenanceRows. Four of those cost
// an investigation each in the week of 2026-09-01, and the refusals were
// hand-rolled branches with no field path, which is why locating them took a
// log join.
//
// Which anthropic-beta value introduced a member is not recorded here. CP9u
// left a column for it; CP9v put the attribution in the dated inventories
// under testdata/inventory instead, and kept it out of this table on purpose.
// The reason is rule 5: production reads the beta set to log it and never to
// decide anything, so a runtime table that carried an attribution would hold a
// value nothing may act on. Two homes for one fact also drift, and the
// inventory is the home CI checks. TestRequestDispositionsMatchTheInventory
// binds the two, so a row added to one and not the other fails.

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
	// Targets holds the disposition per target format. A format that is absent
	// carries the member.
	Targets map[llmprotocol.WireFormat]targetDisposition
}

const (
	// fieldAutoCache is the request-level automatic-cache directive.
	fieldAutoCache           = "cache_control"
	fieldContentCitations    = "content.citations"
	fieldContentCacheControl = "content.cache_control"
	fieldContentCaller       = "content.caller"
	fieldContentDocumentText = "content.document"
	fieldToolsType           = "tools.type"
	// fieldToolsTypeAnthropicDefined is the second row keyed by what a member
	// says rather than by its presence: a tools.type value naming one of
	// Anthropic's own caller-run tools, which the neutral contract can spell
	// out. See llmprotocol.AnthropicDefinedToolTypes.
	fieldToolsTypeAnthropicDefined = "tools.type.anthropic_defined"
	// fieldSystemBillingAttribution is the one row keyed by what a block says
	// rather than by a member it carries: a system text block that is Claude
	// Code's billing attribution line. See billingAttributionLine.
	fieldSystemBillingAttribution = "system.x-anthropic-billing-header"
	// The three reasoning rows are keyed by who can verify a block, which is
	// read off its shape: a signature or a redacted_thinking block is
	// Anthropic's and only Anthropic can check it; reasoning with no signature
	// is plain text any model can read. CarryReasoningTo applies them at
	// dispatch, before the capability gate, and counts every drop by kind.
	fieldReasoningSigned   = "content.thinking.signed"
	fieldReasoningUnsigned = "content.thinking.unsigned"
	fieldRedactedThinking  = "content.redacted_thinking"
)

// reasoningProvenanceRows are the rows that state a disposition for the
// Messages target too. Every other row describes a member Messages can
// express. Unsigned thinking is not one: a Messages host verifies thinking by
// its signature and rejects a block that has none.
var reasoningProvenanceRows = map[string]bool{
	fieldReasoningSigned:   true,
	fieldReasoningUnsigned: true,
	fieldRedactedThinking:  true,
}

// billingAttributionPrefix opens the line Claude Code prepends as system[0]
// of every request. The line names the client build and entrypoint, and since
// 2.1.260 a per-turn hash and a per-prompt id as well.
const billingAttributionPrefix = "x-anthropic-billing-header:"

// billingAttributionGrammar is the whole of the line: the prefix, then one or
// more "key=value;" fields, each led by a space, and nothing after the last
// semicolon. The captures on disk read
//
//	x-anthropic-billing-header: cc_version=2.1.260.ada; cc_entrypoint=sdk-cli; cch=14346; cc_prompt_id=22de3847-...;
//
// A value never holds whitespace or a semicolon, so a block that opens with
// the prefix and goes on in prose does not match, whatever its length.
var billingAttributionGrammar = regexp.MustCompile(`^x-anthropic-billing-header:(?: [A-Za-z_]+=[^;\s]*;)+$`)

// billingAttributionVersionField is the one field every version of the line
// has carried. A line without it is not Claude Code's.
const billingAttributionVersionField = " cc_version="

var anthropicRequestDispositions = []requestFieldRow{
	{
		// Automatic caching: cache the conversation without naming a block.
		// Messages carries it (a Messages client's member as written, any
		// other source as a placed breakpoint; see placeAutoCacheBreakpoint)
		// and Responses re-emits OpenRouter's top-level member. Chat has no
		// request-level member for it, and choosing a block for a Chat
		// provider would guess at a cache whose rules differ by model, so
		// the directive is dropped and counted.
		Path: fieldAutoCache,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "Chat Completions has no request-level cache directive",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionCarry,
				Reason: "Responses re-emits the top-level cache_control extension",
			},
		},
	},
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
		// Claude Code's billing attribution line. Anthropic reads it as
		// metadata and caches around it; every other host reads it as prompt
		// text. Two of its fields change on every turn, so on a foreign
		// target the prompt prefix differs from about token 30 onward and no
		// provider prefix cache can match. Measured on the dev cell on
		// 2026-09-04: 8 of 8 MiMo turns reported zero cached prompt tokens,
		// and the same bodies with the line held constant cached 86-99%.
		// The line says nothing to a model, so a target that cannot read it
		// as metadata drops it and counts the drop.
		Path: fieldSystemBillingAttribution,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "a Chat host reads the billing attribution line as prompt text, which defeats its prefix cache",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionDrop,
				Reason: "a Responses host reads the billing attribution line as prompt text, which defeats its prefix cache",
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
	{
		// A tool Anthropic defines and the caller runs: the text editor, bash,
		// memory. The declaration names a type in place of a schema, and a
		// target with no type table gets the documented schema written out
		// under the caller's name. Dropping it instead sent the arm a history
		// of calls to a tool it was never shown, and every Workshop agent
		// turn of 2026-09-14 declared one.
		Path: fieldToolsTypeAnthropicDefined,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionTransform,
				Reason: "an Anthropic-defined tool becomes a Chat function with its documented schema",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionTransform,
				Reason: "an Anthropic-defined tool becomes a Responses function with its documented schema",
			},
		},
	},
	{
		// Claude's thinking with its signature. The signature is a MAC under
		// Anthropic's key, so no other host can check it: it is stripped and
		// counted, and the text is carried as reasoning, as it was before the
		// table existed. A signature never crosses to another model's host,
		// and the text is never sent as visible text: pi did that on
		// 2026-10-02, and the next model imitated it.
		Path: fieldReasoningSigned,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionTransform,
				Reason: "a Chat host cannot verify an Anthropic thinking signature; the text is carried as reasoning",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionTransform,
				Reason: "a Responses host cannot verify an Anthropic thinking signature; the text is carried as reasoning",
			},
		},
	},
	{
		// Anthropic's encrypted thinking. Only its issuer can read it; an
		// Anthropic target gets it as sent (and the ARC issuer ledger drops
		// it for a different Anthropic-format issuer).
		Path: fieldRedactedThinking,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.OpenAIChatV1: {
				Action: dispositionDrop,
				Reason: "redacted_thinking is readable only by the model that issued it",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionDrop,
				Reason: "redacted_thinking is readable only by the model that issued it",
			},
		},
	},
	{
		// Reasoning with no signature: an open-weight model's text, from a
		// Chat arm or a Chat client. DeepSeek and MiMo need it back on the
		// assistant messages of a tool loop and answer 400 without it, so a
		// Chat target keeps it as reasoning_content. A Messages host rejects
		// thinking with no signature, so that target drops it.
		Path: fieldReasoningUnsigned,
		Targets: map[llmprotocol.WireFormat]targetDisposition{
			llmprotocol.AnthropicMessagesV1: {
				Action: dispositionDrop,
				Reason: "a Messages host rejects thinking that has no signature",
			},
			llmprotocol.OpenAIChatV1: {
				Action: dispositionCarry,
				Reason: "unsigned reasoning is sent back as reasoning_content, which DeepSeek and MiMo require in a tool history",
			},
			llmprotocol.OpenAIResponsesV1: {
				Action: dispositionCarry,
				Reason: "unsigned reasoning is sent as a reasoning item's reasoning_text",
			},
		},
	},
}

// reasoningProvenance names the reasoning row a content block belongs to, or
// "" for a block that is not reasoning. A reasoning content that carries only
// OpenRouter reasoning_details or a Responses encrypted_content has no
// signature and is unsigned here; its carrier has its own rules (see
// reasoning_details.go), which this row does not change.
func reasoningProvenance(content llmprotocol.Content) string {
	switch {
	case content.Kind == llmprotocol.ContentReasoning && content.Signature != "":
		return fieldReasoningSigned
	case content.Kind == llmprotocol.ContentReasoning:
		return fieldReasoningUnsigned
	case redactedThinkingBlock(content):
		return fieldRedactedThinking
	}
	return ""
}

// redactedThinkingBlock reports whether a content is a carried Anthropic
// redacted_thinking block, whatever its data holds.
func redactedThinkingBlock(content llmprotocol.Content) bool {
	block := content.Unmodeled
	return content.Kind == llmprotocol.ContentUnmodeled && block != nil &&
		block.Format == llmprotocol.AnthropicMessagesV1 && block.Type == "redacted_thinking"
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
	if request.AutoCache != nil {
		paths = append(paths, fieldAutoCache)
	}
	for _, tool := range request.Tools {
		if _, defined := tool.AnthropicDefined(); defined {
			paths = append(paths, fieldToolsTypeAnthropicDefined)
		} else if tool.ServerTool() {
			paths = append(paths, fieldToolsType)
		}
	}
	for _, instruction := range request.Instructions {
		paths = append(paths, presentContentFields(instruction.Content)...)
		for _, content := range instruction.Content {
			if billingAttributionLine(content) {
				paths = append(paths, fieldSystemBillingAttribution)
			}
		}
	}
	for _, message := range request.Messages {
		paths = append(paths, presentContentFields(message.Content)...)
	}
	sort.Strings(paths)
	return paths
}

// billingAttributionLine reports whether a block is Claude Code's billing
// attribution line: a text block that is that one line, in its grammar, with
// its version field, and nothing else. A block that opens with the prefix and
// goes on to other text is not the line; it is prompt text that happens to
// start that way, and dropping it would send a question with part of its
// instructions removed, which is the outcome the document row above exists to
// prevent.
func billingAttributionLine(content llmprotocol.Content) bool {
	return content.Kind == llmprotocol.ContentText &&
		strings.HasPrefix(content.Text, billingAttributionPrefix) &&
		strings.Contains(content.Text, billingAttributionVersionField) &&
		billingAttributionGrammar.MatchString(content.Text)
}

// instructionContentFor returns the blocks of one instruction that this
// target sends. The table says whether the target drops the billing
// attribution line; every other block is sent, and an instruction the drop
// empties comes back as an empty list. The count of what was dropped is
// appendRequestDispositions' job, so this returns the content and nothing
// else.
func instructionContentFor(contents []llmprotocol.Content, target llmprotocol.WireFormat) []llmprotocol.Content {
	if dispositionFor(fieldSystemBillingAttribution, target).Action != dispositionDrop {
		return contents
	}
	kept := make([]llmprotocol.Content, 0, len(contents))
	for _, content := range contents {
		if billingAttributionLine(content) {
			continue
		}
		kept = append(kept, content)
	}
	return kept
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
