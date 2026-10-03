package protocolcodec

import "encoding/json"

// Tool-result media on a Chat Completions target.
//
// Anthropic Messages and Responses let a tool result carry an image: a
// screenshot, a PNG the agent read, OpenClaw's view_image. A Chat tool message
// holds text only, so the Chat encoder used to refuse the turn, and the
// routing gate kept such a turn off every Chat arm (no_capable_arm, 503).
//
// The encoder now re-expresses it instead:
//
//   - The tool message keeps the result's text, in order. When the result had
//     no text at all, it gets toolResultMediaPointer so it is never empty.
//   - The media goes into one synthetic user message placed right after the
//     run of consecutive tool messages, not after each one. Chat requires the
//     tool messages answering an assistant's tool calls to follow it with no
//     other message between them, so a user message inside the run would be
//     refused by the provider.
//   - Inside that user message each result's media is preceded by a text label
//     naming its tool call id (toolResultMediaLabel), and results appear in
//     the order of their tool messages, so the model can tell which tool call
//     produced which image when several results carry media.
//   - Each media part is encoded exactly as it would be in a user message
//     (image_url data URL, input_audio, file), cache_control included.
//
// What the model loses is position only: the image now sits after the run of
// results rather than inside its own result. The content is unchanged, which
// is why tool-result media no longer needs a routing capability; the vision
// gate, which already reads images nested in tool results, keeps the turn on
// arms that take image input.

const toolResultMediaPointer = "(The tool returned media; it is attached in the next user message.)"

func toolResultMediaLabel(callID string) string {
	return "Media returned by tool call " + callID + ":"
}

// toolResultMediaRun collects the media of consecutive tool messages and
// writes it as one user message when the run ends.
type toolResultMediaRun struct {
	parts []chatContentWire
}

func (run *toolResultMediaRun) add(parts []chatContentWire) {
	run.parts = append(run.parts, parts...)
}

func (run *toolResultMediaRun) flush(wire *chatRequestWire) {
	if len(run.parts) == 0 {
		return
	}
	message := chatMessageWire{Role: "user"}
	message.Content, _ = json.Marshal(run.parts)
	wire.Messages = append(wire.Messages, message)
	run.parts = nil
}
