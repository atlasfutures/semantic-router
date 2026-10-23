package extproc

import (
	"github.com/tidwall/gjson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// Claude Code under the message-threads-2026-08-12 beta continues a turn by
// naming the previous response instead of re-sending the conversation:
//
//	{"thread":{"type":"continue","previous_message_id":"msg_…"},
//	 "messages":[{"role":"user","content":[{"type":"tool_result",…}]}, …]}
//
// The earlier messages are meant to be rebuilt by the server from the thread.
// This cell keeps no thread, so such a body is a fragment of a conversation,
// never one it can serve: answered, the model sees a lone tool result and the
// user gets a reply with no history (2026-10-23, rt_a9845555-41e). Refused with
// the orphan tool-result 400, Claude Code re-sends the turn with its full
// history, which is served as any other turn.
//
// The same client opens every turn it sends in full with
// {"thread":{"type":"create"}} (and the previous response id under
// diagnostics). Such a turn carries its whole history and is served as any
// other; only a thread that names server-held history is refused.
//
// threadContinuationRefusal returns the refusal for a Messages body whose
// top-level thread continues server-held history (type continue, or any
// thread naming a previous_message_id), and nil for any other body. The
// refusal is the one the codec gives the same body, so the client sees byte
// for byte the 400 it already answers with a full re-send; a continuation the
// codec would accept is refused with the same code and message, naming the
// thread member.
func threadContinuationRefusal(engine *protocolcodec.Engine, format llmprotocol.WireFormat, body []byte) error {
	if format != llmprotocol.AnthropicMessagesV1 || !continuesServerThread(gjson.GetBytes(body, "thread")) {
		return nil
	}
	if _, _, _, err := engine.DecodeRequestForMutation(format, body); err != nil {
		return err
	}
	return llmprotocol.NewFieldError(llmprotocol.ErrorInvalidRequest, "orphan_tool_result",
		"tool result must follow its request tool call or carry a retained-history link", "", "thread")
}

// continuesServerThread says a thread member asks the server for history the
// request does not carry.
func continuesServerThread(thread gjson.Result) bool {
	if !thread.IsObject() {
		return false
	}
	return thread.Get("type").String() == "continue" || thread.Get("previous_message_id").Exists()
}
