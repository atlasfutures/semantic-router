/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

type InputProtocol string

const (
	ProtocolAnthropicMessages InputProtocol = "anthropic_messages"
	ProtocolOpenAIChat        InputProtocol = "openai_chat"
	ProtocolOpenAIResponses   InputProtocol = "openai_responses"
)

type Turn struct {
	Role string `json:"role"`
	Text string `json:"text"`
}

type TurnNormalizationError struct {
	Code string
	Path string
	// Detail is a bounded, non-sensitive discriminator for the failure, such
	// as the unrecognised block type. It never carries request content, so it
	// is safe to log. Keep it out of metric labels: it is unbounded in
	// cardinality even though it is bounded in sensitivity.
	Detail string
	Err    error
}

func (err *TurnNormalizationError) Error() string {
	if err.Path == "" {
		return fmt.Sprintf("normalize ARC turns: %s: %v", err.Code, err.Err)
	}
	return fmt.Sprintf(
		"normalize ARC turns at %s: %s: %v",
		err.Path,
		err.Code,
		err.Err,
	)
}

func (err *TurnNormalizationError) Unwrap() error {
	return err.Err
}

func TurnNormalizationErrorCode(err error) string {
	var normalizationError *TurnNormalizationError
	if errors.As(err, &normalizationError) {
		return normalizationError.Code
	}
	return ""
}

// TurnNormalizationErrorDetail returns the bounded discriminator for a
// normalization failure, or the empty string when the failure carries none.
func TurnNormalizationErrorDetail(err error) string {
	var normalizationError *TurnNormalizationError
	if errors.As(err, &normalizationError) {
		return normalizationError.Detail
	}
	return ""
}

// TurnNormalizationErrorPath returns the request path that failed to
// normalize, or the empty string when the failure is not path-scoped.
func TurnNormalizationErrorPath(err error) string {
	var normalizationError *TurnNormalizationError
	if errors.As(err, &normalizationError) {
		return normalizationError.Path
	}
	return ""
}

// TurnOptions controls what the normalizer shows the selector.
//
// System text is split into two scopes because they carry different things and
// cost different amounts. Neither option can add a turn and neither can add a
// role: the encoder wire schema pins ArcTurn.role to user or assistant, and the
// serializer writes the role string into the token stream, so a third role
// would emit tokens the trained checkpoint has never been shown. Folding the
// text into an existing turn keeps the turn count, the role alternation, and
// the token shapes as they are. Only the turn text gets longer.
type TurnOptions struct {
	// IncludeSystemText folds the conversation-opening system prompt into the
	// user turn it governs instead of discarding it. It covers the Anthropic
	// top-level system field, the Responses top-level instructions field, and
	// the leading run of system or developer messages in a message array.
	//
	// The default is false, which discards it. That is what every deployment
	// did before this option existed, and it is what the trained selector has
	// been consulted with to date, so it stays the default until measurement
	// says otherwise. An opening prompt is also the longest system text a
	// request carries, so folding it in moves the most tokens.
	IncludeSystemText bool

	// DropMidConversationSystemText discards system text that arrives after
	// the conversation has started, restoring the drop-everything behaviour
	// that shipped before the two scopes were separated.
	//
	// The default is false, so mid-conversation system text reaches the
	// selector. It is the opposite default from IncludeSystemText because it
	// is the opposite kind of text: a correction aimed at the very next reply
	// -- answer in JSON, you are in plan mode now, use the other tool -- which
	// is the signal the router exists to read, and short enough that folding
	// it in barely moves the token count.
	//
	// This field is a kill switch, which is why it is worded as a negative:
	// the Go zero value has to mean "include".
	DropMidConversationSystemText bool
}

func NormalizeTurns(
	protocol InputProtocol,
	requestBody []byte,
	options TurnOptions,
) ([]Turn, error) {
	request, err := decodeRequestObject(requestBody)
	if err != nil {
		return nil, err
	}
	switch protocol {
	case ProtocolAnthropicMessages:
		return normalizeAnthropicTurns(request, options)
	case ProtocolOpenAIChat:
		return normalizeOpenAIChatTurns(request, options)
	case ProtocolOpenAIResponses:
		return normalizeOpenAIResponsesTurns(request, options)
	default:
		return nil, turnError(
			"unsupported_protocol",
			"",
			"unsupported protocol %q",
			protocol,
		)
	}
}

func decodeRequestObject(data []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, turnError("invalid_json", "", "%v", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return nil, turnError(
				"invalid_json",
				"",
				"request contains multiple JSON values",
			)
		}
		return nil, turnError("invalid_json", "", "%v", err)
	}
	request, ok := value.(map[string]any)
	if !ok {
		return nil, turnError(
			"invalid_request",
			"",
			"request body must be an object",
		)
	}
	return request, nil
}

func requiredArray(
	object map[string]any,
	field string,
	path string,
) ([]any, error) {
	value, ok := object[field]
	if !ok {
		return nil, turnError(
			"missing_field",
			path+"."+field,
			"field is required",
		)
	}
	array, ok := value.([]any)
	if !ok {
		return nil, turnError(
			"invalid_field",
			path+"."+field,
			"field must be an array",
		)
	}
	return array, nil
}

func requiredObject(value any, path string) (map[string]any, error) {
	object, ok := value.(map[string]any)
	if !ok {
		return nil, turnError(
			"invalid_item",
			path,
			"item must be an object",
		)
	}
	return object, nil
}

func requiredString(
	object map[string]any,
	field string,
	path string,
) (string, error) {
	value, ok := object[field]
	if !ok {
		return "", turnError(
			"missing_field",
			path+"."+field,
			"field is required",
		)
	}
	text, ok := value.(string)
	if !ok {
		return "", turnError(
			"invalid_field",
			path+"."+field,
			"field must be a string",
		)
	}
	return text, nil
}

func optionalBool(
	object map[string]any,
	field string,
	path string,
) (bool, error) {
	value, ok := object[field]
	if !ok || value == nil {
		return false, nil
	}
	result, ok := value.(bool)
	if !ok {
		return false, turnError(
			"invalid_field",
			path+"."+field,
			"field must be a boolean",
		)
	}
	return result, nil
}

func turnError(
	code string,
	path string,
	format string,
	arguments ...any,
) error {
	return &TurnNormalizationError{
		Code: code,
		Path: path,
		Err:  fmt.Errorf(format, arguments...),
	}
}

// turnErrorWithDetail is turnError plus a bounded discriminator that survives
// into logs. Use it wherever the failure names something the caller cannot
// otherwise recover, such as an unrecognised block type.
func turnErrorWithDetail(
	code string,
	path string,
	detail string,
	format string,
	arguments ...any,
) error {
	return &TurnNormalizationError{
		Code:   code,
		Path:   path,
		Detail: detail,
		Err:    fmt.Errorf(format, arguments...),
	}
}

// systemTextScope separates the two kinds of system text a request carries.
// Each scope is configured on its own, so the normalizer has to decide which
// one a piece of text belongs to before it decides what to do with it.
type systemTextScope int

const (
	// systemTextOriginal is the conversation-opening prompt: the standing
	// brief for the whole exchange, written before the first question.
	systemTextOriginal systemTextScope = iota
	// systemTextMidConversation is system text inserted once the conversation
	// is already running. It governs the reply being routed rather than the
	// session.
	systemTextMidConversation
)

// systemTextScopeAt classifies a system or developer message by its position in
// a message array. The leading run -- every system message that appears before
// the first message which produces a turn -- is the opening prompt. Anything
// after that arrived mid-conversation.
func systemTextScopeAt(conversationStarted bool) systemTextScope {
	if conversationStarted {
		return systemTextMidConversation
	}
	return systemTextOriginal
}

// systemTextBuffer holds system-prompt text until a user turn can carry it.
//
// System text cannot become a turn of its own: the wire contract has only two
// roles. So it is folded into the next user turn, which is the turn it
// governs. A request that opens with a system prompt therefore lands that text
// at the front of the first user turn, which the serializer renders as the
// episode's [Task] block.
//
// The buffer is inert for a scope that is switched off. That keeps the call
// sites free of conditionals and makes each disabled path identical to the
// behaviour that shipped before the scopes existed.
type systemTextBuffer struct {
	includeOriginal        bool
	includeMidConversation bool
	pending                []string
}

func newSystemTextBuffer(options TurnOptions) systemTextBuffer {
	return systemTextBuffer{
		includeOriginal:        options.IncludeSystemText,
		includeMidConversation: !options.DropMidConversationSystemText,
	}
}

func (buffer *systemTextBuffer) wants(scope systemTextScope) bool {
	if scope == systemTextMidConversation {
		return buffer.includeMidConversation
	}
	return buffer.includeOriginal
}

// collect renders one piece of system text and buffers it, but only when its
// scope is one the caller wants. render is never called otherwise, which is
// what stops a malformed system message from failing a request whose
// configuration would have discarded that message anyway.
//
// The two scopes differ in what a read failure costs. The opening prompt is
// read only behind a deliberate opt-in, so a failure there is reported: the
// operator asked for that text and should hear that it could not be read.
// Mid-conversation text is read by default, so it must not be able to fail a
// request that routes today. A malformed instruction there contributes nothing
// and the episode continues, which is exactly what every deployment did before
// the scope was included.
func (buffer *systemTextBuffer) collect(
	scope systemTextScope,
	render func() (string, error),
) error {
	if !buffer.wants(scope) {
		return nil
	}
	text, err := render()
	if err != nil {
		if scope == systemTextOriginal {
			return err
		}
		return nil
	}
	buffer.add(text)
	return nil
}

func (buffer *systemTextBuffer) add(text string) {
	if text == "" {
		return
	}
	buffer.pending = append(buffer.pending, text)
}

// midConversationSystemText renders system text that already sits at the
// position it governs, so it needs no buffering. It contributes nothing when
// the scope is off, and nothing when the content cannot be read, on the same
// reasoning collect gives for the mid-conversation scope.
func midConversationSystemText(
	include bool,
	render func() (string, error),
) string {
	if !include {
		return ""
	}
	text, err := render()
	if err != nil {
		return ""
	}
	return text
}

// take returns the buffered text and empties the buffer.
func (buffer *systemTextBuffer) take() string {
	if len(buffer.pending) == 0 {
		return ""
	}
	text := strings.Join(buffer.pending, "\n\n")
	buffer.pending = nil
	return text
}

// flushTrailing attaches text still buffered after the last message to the
// final turn. A system message with no user turn after it still governs the
// reply being routed, so its text belongs on the turn nearest to it rather
// than discarded. Requests almost always end on a user turn, so this is the
// same destination the ordinary path would have chosen.
func (buffer *systemTextBuffer) flushTrailing(turns []Turn) []Turn {
	text := buffer.take()
	if text == "" || len(turns) == 0 {
		return turns
	}
	last := &turns[len(turns)-1]
	last.Text = joinSystemText(text, last.Text)
	return turns
}

// joinSystemText puts system text in front of the text it governs. The
// separator is a blank line and nothing else: a marker token would be a
// sequence the trained selector has never been shown.
func joinSystemText(system string, text string) string {
	switch {
	case system == "":
		return text
	case text == "":
		return system
	default:
		return system + "\n\n" + text
	}
}
