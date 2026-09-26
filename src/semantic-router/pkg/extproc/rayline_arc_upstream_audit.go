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

package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// raylineARCUpstreamAudit is the verdict on one provider-bound body: its
// message count and chained digest, and whether it extends the last body the
// episode sent the same worker. It carries no content.
type raylineARCUpstreamAudit struct {
	Messages       int
	Digest         string
	PrefixMessages int
	Extends        *bool
}

// auditRaylineARCUpstream checks the invariant a prompt cache depends on: for
// one episode and one worker, each body's messages start with the last body's
// messages. It runs on the final bytes, after every mutation, so it judges
// what the provider actually receives.
//
// Messages are compared as a client-independent projection: cache directives
// removed, and a content array of one text part read as that text. Agent
// clients move their cache breakpoint every turn, which changes the bytes of
// an earlier message without changing what the model is asked; that is not a
// broken extension. Anything else that changes is.
//
// The new record is staged on the episode transaction and commits only with
// a turn that reached a 2xx response, so a retry is judged against the last
// committed body.
func (r *OpenAIRouter) auditRaylineARCUpstream(body []byte, ctx *RequestContext) {
	if ctx == nil || ctx.RaylineARCDispatch == nil || ctx.RaylineARCTransaction == nil {
		return
	}
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil ||
		!decision.Algorithm.RaylineARC.UpstreamAudit.Enabled {
		return
	}
	chain, ok := upstreamMessageChain(body)
	if !ok {
		return
	}
	worker := ctx.RaylineARCDispatch.ID
	audit := &raylineARCUpstreamAudit{Messages: len(chain) - 1, Digest: chain[len(chain)-1]}
	if previous, found := ctx.RaylineARCTransaction.state.UpstreamPrefixFor(worker); found {
		extends := previous.Messages <= audit.Messages && chain[previous.Messages] == previous.Digest
		audit.PrefixMessages, audit.Extends = previous.Messages, &extends
	}
	ctx.RaylineARCUpstreamAudit = audit
	ctx.RaylineARCTransaction.stageUpstreamPrefix(raylinearc.UpstreamPrefix{
		Worker: worker, Messages: audit.Messages, Digest: audit.Digest,
	})
}

// upstreamMessageChain returns the chained digest after each message: entry i
// covers the first i messages, entry 0 is the empty prefix. Digests are the
// first 16 bytes of sha256, in hex.
func upstreamMessageChain(body []byte) ([]string, bool) {
	var wire struct {
		Messages []json.RawMessage `json:"messages"`
	}
	if json.Unmarshal(body, &wire) != nil {
		return nil, false
	}
	chain := make([]string, 0, len(wire.Messages)+1)
	running := sha256.Sum256(nil)
	chain = append(chain, hex.EncodeToString(running[:16]))
	for _, raw := range wire.Messages {
		var message interface{}
		if json.Unmarshal(raw, &message) != nil {
			return nil, false
		}
		projected, _ := json.Marshal(projectUpstreamMessage(message))
		digest := sha256.Sum256(projected)
		running = sha256.Sum256(append(running[:], digest[:]...))
		chain = append(chain, hex.EncodeToString(running[:16]))
	}
	return chain, true
}

func projectUpstreamMessage(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		projected := make(map[string]interface{}, len(typed))
		for key, member := range typed {
			if key != "cache_control" {
				projected[key] = projectUpstreamMessage(member)
			}
		}
		if parts, ok := projected["content"].([]interface{}); ok && len(parts) == 1 {
			if part, ok := parts[0].(map[string]interface{}); ok && part["type"] == "text" && len(part) == 2 {
				if text, ok := part["text"].(string); ok {
					projected["content"] = text
				}
			}
		}
		return projected
	case []interface{}:
		projected := make([]interface{}, len(typed))
		for index, member := range typed {
			projected[index] = projectUpstreamMessage(member)
		}
		return projected
	default:
		return value
	}
}

func appendRaylineARCUpstreamFields(record map[string]interface{}, ctx *RequestContext) {
	if ctx == nil || ctx.RaylineARCUpstreamAudit == nil {
		return
	}
	audit := ctx.RaylineARCUpstreamAudit
	record["upstream_messages"] = audit.Messages
	record["upstream_digest"] = audit.Digest
	if audit.Extends != nil {
		record["upstream_extends_previous"] = *audit.Extends
		record["upstream_prefix_messages"] = audit.PrefixMessages
	}
}
