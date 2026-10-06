package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ChatUsageStreamFilter is the public form of a same-format Chat stream. It
// removes Router-requested accounting evidence when the public client did not
// opt into usage, and it states a stop at the output limit as Chat does: a
// choice whose native_finish_reason is a length reason ends with
// finish_reason "length", whatever label the upstream gave it (OpenRouter
// says "tool_calls" for a call cut at the limit, which a client would run).
// A provider error frame is restated in its public form. It preserves every
// other JSON field, including provider extensions, and is
// independent from neutral semantic decoding used for accounting.
type ChatUsageStreamFilter struct {
	framer          sseFramer
	frames          int
	pendingTerminal []byte
	failure         error
	finalized       bool
	keepUsage       bool
}

func NewChatUsageStreamFilter(limit int) *ChatUsageStreamFilter {
	return &ChatUsageStreamFilter{framer: newSSEFramer(limit)}
}

// NewChatPassthroughStreamFilter is the filter for a same-format Chat stream
// whose client asked for usage: it keeps usage and only restates a native
// length finish.
func NewChatPassthroughStreamFilter(limit int) *ChatUsageStreamFilter {
	return &ChatUsageStreamFilter{framer: newSSEFramer(limit), keepUsage: true}
}

func (filter *ChatUsageStreamFilter) Push(chunk []byte) ([]byte, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.failure != nil {
		return nil, filter.failure
	}
	if filter.finalized {
		return nil, llmprotocol.NewError(llmprotocol.ErrorConflict, "stream_terminal", "stream is already finalized", nil)
	}
	frames, err := filter.framer.Push(chunk)
	if err != nil {
		return nil, filter.poison(err)
	}
	return filter.filterFrames(frames)
}

func (filter *ChatUsageStreamFilter) Finalize() ([]byte, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.finalized {
		return nil, nil
	}
	filter.finalized = true
	if filter.failure != nil {
		return nil, filter.failure
	}
	frames, err := filter.framer.Finalize()
	if err != nil {
		return nil, filter.poison(err)
	}
	output, err := filter.filterFrames(frames)
	if err != nil {
		return nil, err
	}
	output = append(output, filter.pendingTerminal...)
	filter.pendingTerminal = nil
	return output, nil
}

func (filter *ChatUsageStreamFilter) filterFrames(frames [][]byte) ([]byte, error) {
	var output bytes.Buffer
	for _, frame := range frames {
		filtered, keep, hasData, terminal, err := filterChatUsageFrame(frame, filter.framer.limit, filter.frames == 0, filter.keepUsage)
		filter.frames++
		if err != nil {
			return nil, filter.poison(err)
		}
		if len(filter.pendingTerminal) != 0 && hasData {
			return nil, filter.poison(invalidProviderResponse(
				"stream_event_after_terminal",
				"Chat stream emitted data after its terminal sentinel",
			))
		}
		if terminal {
			filter.pendingTerminal = append(filter.pendingTerminal[:0], filtered...)
			continue
		}
		if keep {
			output.Write(filtered)
		}
	}
	return output.Bytes(), nil
}

func (filter *ChatUsageStreamFilter) poison(err error) error {
	if err != nil && filter.failure == nil {
		filter.failure = err
	}
	filter.pendingTerminal = nil
	return filter.failure
}

func filterChatUsageFrame(frame []byte, limit int, first, keepUsage bool) ([]byte, bool, bool, bool, error) {
	parsed, err := parseSSEFrameAtPosition(frame, limit, first)
	if err != nil {
		return nil, false, false, false, err
	}
	if !parsed.HasData {
		return frame, true, false, false, nil
	}
	if bytes.Equal(bytes.TrimSpace(parsed.Data), []byte("[DONE]")) {
		return frame, true, true, true, nil
	}
	var object map[string]json.RawMessage
	if err := decodeProviderWire(parsed.Data, &object, llmprotocol.DefaultPolicy()); err != nil {
		return nil, false, true, false, err
	}
	restated, err := restateChatNativeLengthFinish(object)
	if err != nil {
		return nil, false, true, false, err
	}
	// A provider error raised mid-stream reaches the client only in its
	// public form.
	publicError, err := publicChatStreamError(object)
	if err != nil {
		return nil, false, true, false, err
	}
	restated = restated || publicError
	usage, hasUsage := object["usage"]
	if !hasUsage || keepUsage {
		if !restated {
			return frame, true, true, false, nil
		}
		encoded, err := encodeSSE(parsed.Event, object)
		return encoded, err == nil, true, false, err
	}
	var choices []json.RawMessage
	if rawChoices, exists := object["choices"]; exists && !bytes.Equal(bytes.TrimSpace(rawChoices), []byte("null")) {
		if err := json.Unmarshal(rawChoices, &choices); err != nil {
			return nil, false, true, false, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
		}
	}
	if len(choices) == 0 && !bytes.Equal(bytes.TrimSpace(usage), []byte("null")) {
		return nil, false, true, false, nil
	}
	delete(object, "usage")
	filtered, err := encodeSSE(parsed.Event, object)
	return filtered, err == nil, true, false, err
}

// restateChatNativeLengthFinish sets finish_reason "length" on each choice
// whose native_finish_reason is a length reason, as the stream decoder reads
// it (decodeChatChoiceStop). It reports whether it changed anything.
func restateChatNativeLengthFinish(object map[string]json.RawMessage) (bool, error) {
	rawChoices, exists := object["choices"]
	if !exists || !bytes.Contains(rawChoices, []byte("native_finish_reason")) {
		return false, nil
	}
	var choices []map[string]json.RawMessage
	if err := json.Unmarshal(rawChoices, &choices); err != nil {
		return false, llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_chat_stream", "Chat stream choices are invalid", err)
	}
	changed := false
	for _, choice := range choices {
		var finish, native *string
		_ = json.Unmarshal(choice["finish_reason"], &finish)
		_ = json.Unmarshal(choice["native_finish_reason"], &native)
		if finish == nil || native == nil || *finish == "length" || !chatNativeLengthReason(*native) {
			continue
		}
		choice["finish_reason"] = json.RawMessage(`"length"`)
		changed = true
	}
	if !changed {
		return false, nil
	}
	encoded, err := json.Marshal(choices)
	if err != nil {
		return false, err
	}
	object["choices"] = encoded
	return true, nil
}
