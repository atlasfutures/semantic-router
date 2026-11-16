package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Thinking-signature fingerprints.
//
// A signature 400 on a Claude turn whose history the Router served on one
// model and one host family has two candidate causes: the block the client
// replayed is not the block the Router served (the codec rebuilt it with a
// changed signature or text, or the client edited it), or it is the same
// block and the history around it changed. The served block and the refused
// one are each logged by a fingerprint: a short SHA-256 prefix and the
// length of the signature and of the thinking text. Two fingerprints that
// match are the same block; the content itself is never logged.

const (
	// thinkingFingerprintHex is the length of a fingerprint's digest prefix.
	thinkingFingerprintHex = 12
	// maxServedThinkingFingerprints bounds the blocks one served reply logs.
	maxServedThinkingFingerprints = 8
)

// thinkingFingerprint names a thinking block by digests and lengths only.
// A redacted_thinking block's data stands as its signature and it has no
// text.
type thinkingFingerprint struct {
	kind      string
	sigSHA256 string
	sigLen    int
	textSHA   string
	textLen   int
}

func thinkingDigest(value string) string {
	if value == "" {
		return ""
	}
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:])[:thinkingFingerprintHex]
}

func newThinkingFingerprint(kind, signature, text string) thinkingFingerprint {
	return thinkingFingerprint{
		kind:      kind,
		sigSHA256: thinkingDigest(signature),
		sigLen:    len(signature),
		textSHA:   thinkingDigest(text),
		textLen:   len(text),
	}
}

// fields is the fingerprint as log fields, each key prefixed.
func (fingerprint thinkingFingerprint) fields(prefix string, into map[string]interface{}) {
	into[prefix+"block_kind"] = fingerprint.kind
	into[prefix+"sig_sha256"] = fingerprint.sigSHA256
	into[prefix+"sig_len"] = fingerprint.sigLen
	into[prefix+"thinking_sha256"] = fingerprint.textSHA
	into[prefix+"thinking_len"] = fingerprint.textLen
}

func (fingerprint thinkingFingerprint) record() map[string]interface{} {
	record := map[string]interface{}{}
	fingerprint.fields("", record)
	return record
}

// wireThinkingFingerprint fingerprints a Messages wire block, and false when
// it is not thinking.
func wireThinkingFingerprint(block map[string]json.RawMessage) (thinkingFingerprint, bool) {
	key, thinking := wireThinkingKey(block)
	if !thinking {
		return thinkingFingerprint{}, false
	}
	kind := wireBlockType(block)
	var text string
	if kind == "thinking" {
		_ = json.Unmarshal(block["thinking"], &text)
	}
	return newThinkingFingerprint(kind, key, text), true
}

// refusedThinkingFields describes, by fingerprints and counts only, the
// block a Messages signature refusal names in the body it refused: where it
// sits, its fingerprint, whether its anchor key is one the episode recorded
// (a run's anchor, or a block a host refused before), and how many thinking
// blocks the history held. A refusal on another wire, or one that names no
// block, adds only the count it can.
func refusedThinkingFields(
	dispatched []byte,
	format llmprotocol.WireFormat,
	refusal []byte,
	provenance []reasoning.ReasoningProvenance,
	into map[string]interface{},
) {
	if format != llmprotocol.AnthropicMessagesV1 {
		return
	}
	messages, ok := wireMessages(dispatched)
	if !ok {
		return
	}
	blocks := 0
	for _, message := range messages {
		for _, block := range wireContentBlocks(message) {
			if _, thinking := wireThinkingKey(block); thinking {
				blocks++
			}
		}
	}
	into["history_thinking_blocks"] = blocks
	messageIndex, blockIndex, named := refusedBlockIndex(refusal)
	if !named {
		return
	}
	into["refused_message_index"] = messageIndex
	into["refused_block_index"] = blockIndex
	if messageIndex >= len(messages) {
		return
	}
	content := wireContentBlocks(messages[messageIndex])
	if blockIndex >= len(content) {
		return
	}
	fingerprint, thinking := wireThinkingFingerprint(content[blockIndex])
	if !thinking {
		into["refused_block_kind"] = wireBlockType(content[blockIndex])
		return
	}
	fingerprint.fields("refused_", into)
	into["refused_recorded_anchor"] = provenanceHoldsAnchor(provenance, refusedAnchorKey(content[blockIndex]))
}

// refusedAnchorKey is the anchor key a thinking wire block is recorded by.
func refusedAnchorKey(block map[string]json.RawMessage) string {
	key, thinking := wireThinkingKey(block)
	if !thinking || key == "" {
		return ""
	}
	if wireBlockType(block) == "redacted_thinking" {
		return protocolcodec.RedactedAnchorKey(key)
	}
	return protocolcodec.SignatureAnchorKey(key)
}

// provenanceHoldsAnchor reports whether any worker's record holds key as an
// anchor.
func provenanceHoldsAnchor(provenance []reasoning.ReasoningProvenance, key string) bool {
	if key == "" {
		return false
	}
	digest := reasoning.ThinkingRunAnchorDigest(key)
	for _, entry := range provenance {
		if entry.Block == digest {
			return true
		}
	}
	return false
}

// servedThinkingFingerprints fingerprints the thinking blocks of a served
// reply, in order, at most maxServedThinkingFingerprints, and how many it
// held in all.
func servedThinkingFingerprints(response *llmprotocol.Response) ([]map[string]interface{}, int) {
	if response == nil {
		return nil, 0
	}
	var records []map[string]interface{}
	total := 0
	for _, item := range response.Output {
		for _, content := range item.Content {
			var fingerprint thinkingFingerprint
			switch {
			case content.Kind == llmprotocol.ContentReasoning && content.Signature != "":
				fingerprint = newThinkingFingerprint("thinking", content.Signature, content.Text)
			default:
				data, ok := protocolcodec.OpaqueReasoningData(content)
				if !ok {
					continue
				}
				fingerprint = newThinkingFingerprint("redacted_thinking", data, "")
			}
			total++
			if len(records) < maxServedThinkingFingerprints {
				records = append(records, fingerprint.record())
			}
		}
	}
	return records, total
}

// logServedThinkingFingerprints logs the fingerprints of the signed thinking
// a Claude turn served (thinking_signature_served), so a later refusal's
// refused_sig_sha256 can be joined to the turn that served the block. A
// turn that served none logs nothing.
func (transaction *raylineARCEpisodeTransaction) logServedThinkingFingerprints(requestContext *RequestContext) {
	if transaction == nil || requestContext == nil || transaction.thinkingWorker == "" {
		return
	}
	blocks, total := servedThinkingFingerprints(requestContext.SemanticResponse)
	if total == 0 {
		return
	}
	fields := map[string]interface{}{
		"request_id":      requestContext.RequestID,
		"episode_id_hash": transaction.episodeIDHash,
		"reply_index":     transaction.thinkingReplyIndex,
		"thinking_blocks": total,
		"blocks":          blocks,
		"resent":          transaction.thinkingRestart,
		"upstream_served": transaction.thinkingServed,
	}
	logging.ComponentEvent("extproc", "thinking_signature_served", fields)
}
