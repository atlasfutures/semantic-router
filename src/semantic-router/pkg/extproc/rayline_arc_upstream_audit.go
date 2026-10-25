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
	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
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
// a turn whose complete 2xx response reached the client, so a retry is judged against the last
// committed body.
func (r *OpenAIRouter) auditRaylineARCUpstream(body []byte, ctx *RequestContext) {
	if ctx == nil || ctx.RaylineARCDispatch == nil || ctx.RaylineARCTransaction == nil {
		return
	}
	stageRaylineARCWireCacheTTL(r, body, ctx)
	decision := ctx.VSRSelectedDecision
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.RaylineARC == nil ||
		!decision.Algorithm.RaylineARC.UpstreamAudit.Enabled {
		return
	}
	chain, ok := cachecontext.MessageChain(body)
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

// stageRaylineARCWireCacheTTL records the prompt-cache TTL the dispatched body
// states for a policy-service turn's cache_context. Only an Anthropic model's
// cache follows cache_control; any other's TTL is unstated.
func stageRaylineARCWireCacheTTL(r *OpenAIRouter, body []byte, ctx *RequestContext) {
	transaction := ctx.RaylineARCTransaction
	if transaction == nil || !transaction.commitOnCompletion {
		return
	}
	if r.armFamily(ctx.RaylineARCDispatch.ID) != anthropicFamily {
		transaction.stageWireCacheTTL(0, "")
		return
	}
	transaction.stageWireCacheTTL(cachecontext.WireCacheTTL(body))
}
