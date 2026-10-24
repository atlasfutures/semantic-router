package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

// applyRaylineARCReasoningIssuer decides whether this turn forwards the
// encrypted reasoning its client resends, and stages the issuer set the turn
// leaves behind (atlasfutures/semantic-router#109).
//
// A blob is readable only by the target that issued it, so it is forwarded
// only when the episode's record says every blob the client can hold came
// from this turn's target. Otherwise the codec drops the whole item, as it
// does on every route without an episode. The set is staged here and written
// only with the turn's 2xx commit, so a failed turn leaves it as it was.
func (r *OpenAIRouter) applyRaylineARCReasoningIssuer(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) {
	if request == nil || dispatch == nil || ctx == nil || ctx.RaylineARCTransaction == nil ||
		ctx.RaylineARCTransaction.state == nil {
		return
	}
	previous := ctx.RaylineARCTransaction.state.ReasoningIssuers
	issuer := reasoningIssuerFor(dispatch, r.dispatchCredential(dispatch, ctx))
	held := protocolcodec.HoldsEncryptedReasoning(*request)
	issues := dispatch.targetFormat == llmprotocol.OpenAIResponsesV1
	forwarded := held && issues && raylinearc.ReasoningIssuersAre(previous, issuer)
	request.ForwardsEncryptedReasoning = forwarded
	ctx.RaylineARCTransaction.stageReasoningIssuers(
		raylinearc.NextReasoningIssuers(previous, held, forwarded, issues, issuer),
	)
	if held && !forwarded {
		logging.ComponentEvent("extproc", "rayline_arc_encrypted_reasoning_dropped", map[string]interface{}{
			"request_id": ctx.RequestID,
			"worker":     dispatch.logicalModel,
			"reason":     encryptedReasoningDropReason(previous, issuer, issues),
		})
	}
}

// reasoningIssuerFor names the target that reads and issues a dispatch's
// encrypted reasoning: the worker, the backend it reaches, the model id that
// backend serves, and the credential the turn is sent with, because a blob is
// readable only by the account that issued it. A rotated key, or a per-user
// key injected by an auth backend, is another issuer. It is a truncated
// digest, so the record carries neither configuration names nor any secret.
//
// OpenRouter has no such name. It chooses the serving provider per request,
// even under a pin (and its Responses path does not send the pin), and
// providers cannot read each other's blobs. So an OpenRouter target is the
// unknown issuer: nothing is forwarded to it, and blobs it issues are never
// forwarded anywhere.
func reasoningIssuerFor(dispatch *providerDispatch, credential dispatchCredential) string {
	if providerIsOpenRouter(dispatch.profile) || !credential.known {
		return raylinearc.ReasoningIssuerUnknown
	}
	parts := []string{dispatch.logicalModel, dispatch.backendName, dispatch.upstreamModel, credential.key}
	digest := sha256.Sum256([]byte(strings.Join(parts, "\x00")))
	return hex.EncodeToString(digest[:8])
}

// dispatchCredential is the credential a dispatch will be sent with, resolved
// as appendProviderCredential resolves it. known is false when it cannot be
// resolved, and no blob is then forwarded.
type dispatchCredential struct {
	key   string
	known bool
}

func (r *OpenAIRouter) dispatchCredential(dispatch *providerDispatch, ctx *RequestContext) dispatchCredential {
	provider, providerAuth, err := resolveProviderAuth(dispatch.profile)
	if err != nil {
		return dispatchCredential{}
	}
	if providerAuth.Strategy == "none" {
		return dispatchCredential{known: true}
	}
	if r.CredentialResolver == nil {
		return dispatchCredential{}
	}
	key, err := r.CredentialResolver.KeyForProvider(provider, dispatch.logicalModel, ctx.Headers)
	if err != nil {
		return dispatchCredential{}
	}
	return dispatchCredential{key: key, known: true}
}

func encryptedReasoningDropReason(previous []string, issuer string, issues bool) string {
	switch {
	case !issues:
		return "target_not_responses"
	case issuer == raylinearc.ReasoningIssuerUnknown:
		return "target_provider_varies"
	case len(previous) > 1:
		return "issuers_mixed"
	case len(previous) == 0 || previous[0] == raylinearc.ReasoningIssuerUnknown:
		return "issuer_unknown"
	default:
		return "issuer_other_target"
	}
}

// dropRaylineARCOpaqueReasoningIssuedElsewhere filters private Messages
// reasoning by its recorded worker, not by wire format. OpenRouter's Messages
// endpoint can serve both Claude and GPT; both signatures and redacted blobs
// are encrypted provider state. Unknown provenance (including pre-upgrade
// signatures and evicted entries) cannot authorize replay. DropOpaqueReasoning
// removes the whole thinking group of an incompatible assistant message while
// preserving visible content and tool relationships. The source history is
// untouched, so a later turn on its issuer may still replay compatible groups.
func (r *OpenAIRouter) dropRaylineARCOpaqueReasoningIssuedElsewhere(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	ctx *RequestContext,
) bool {
	if request == nil || dispatch == nil || ctx == nil || ctx.RaylineARCTransaction == nil {
		return false
	}
	transaction := ctx.RaylineARCTransaction
	issuer := r.opaqueReasoningIssuerFor(dispatch)
	transaction.dispatchWorker = issuer
	var provenance []raylinearc.ReasoningProvenance
	if transaction.state != nil {
		provenance = transaction.state.ReasoningProvenance
	}
	dropped := protocolcodec.DropOpaqueReasoning(request, func(data string) bool {
		if dispatch.targetFormat == llmprotocol.AnthropicMessagesV1 {
			return !raylinearc.ReasoningIssuedBy(provenance, data, issuer)
		}
		return raylinearc.ReasoningIssuedElsewhere(provenance, data, issuer)
	})
	if dropped == 0 {
		return false
	}
	logging.ComponentEvent("extproc", "rayline_arc_opaque_reasoning_dropped", map[string]interface{}{
		"request_id": ctx.RequestID,
		"worker":     dispatch.logicalModel,
		"dropped":    dropped,
	})
	return true
}

// opaqueReasoningIssuerFor names the effective issuer of a dispatch's opaque
// reasoning: the serving model and the provider that runs it. On OpenRouter
// the provider is fixed only when the worker pins exactly one with no
// fallbacks; otherwise OpenRouter chooses per request and providers cannot
// read each other's blocks, so the issuer is unknown and its blocks are never
// resent anywhere (as reasoningIssuerFor treats encrypted Responses reasoning).
// The credential does not enter: an Anthropic-format block is the model's,
// not the account's.
func (r *OpenAIRouter) opaqueReasoningIssuerFor(dispatch *providerDispatch) string {
	if providerIsOpenRouter(dispatch.profile) {
		if r.Config == nil {
			return raylinearc.ReasoningIssuerUnknown
		}
		pin := r.Config.ProviderPreferencesForModel(dispatch.logicalModel)
		if pin == nil || pin.AllowFallbacks == nil || *pin.AllowFallbacks || len(pin.Order) != 1 {
			return raylinearc.ReasoningIssuerUnknown
		}
		return strings.Join([]string{dispatch.logicalModel, "openrouter", pin.Order[0], dispatch.upstreamModel}, "\x00")
	}
	return strings.Join([]string{dispatch.logicalModel, dispatch.backendName, dispatch.upstreamModel}, "\x00")
}

// anthropicFamily is the model family armFamily names for Claude: the card
// publisher, or the vendor segment of an anthropic/... provider model id.
const anthropicFamily = "anthropic"

// workerFamily names the family of the worker serving this turn, for the
// stream encoder's provenance decisions (llmprotocol.StreamContext.WorkerFamily):
// the selected worker's, or the requested model's when none was selected. It
// is empty when neither is known, and the encoder then relies on the wire
// format alone.
func (r *OpenAIRouter) workerFamily(ctx *RequestContext) string {
	if r == nil || ctx == nil {
		return ""
	}
	model := ctx.VSRSelectedModel
	if model == "" {
		model = ctx.RequestModel
	}
	if model == "" {
		return ""
	}
	return r.armFamily(model)
}

// dropReasoningForeignToClaude drops, for a Claude worker, the reasoning in
// the history that is not provably Claude's
// (protocolcodec.DropReasoningNotFromAnthropic), and returns how many contents
// it dropped. It is keyed on the worker's model family, not on the wire
// format, so a Claude worker reached over Messages, Chat or Responses gets the
// same history. A turn whose episode has no record arrives with nothing that
// names which model wrote its reasoning, and OpenRouter hands an assistant
// message's reasoning to the model: Anthropic answers another model's
// reasoning with a content_filter refusal.
//
// Each block's fate depends only on the block and the worker's family, never
// on the turn or on what follows it, so a run of turns on one model sends the
// same prefix every turn and its prompt cache keeps matching.
//
// Only a Claude worker is filtered this way. Unproven reasoning cannot be
// dropped for another worker: DeepSeek and MiMo need their own
// reasoning_content back in a tool history, and nothing in a request without
// an episode record proves that it is theirs. The reverse direction keeps the
// disposition table's rule: Claude's signed thinking reaches any other worker
// as reasoning text with its signature stripped, because those providers
// expect reasoning_content on every tool-calling turn of a history, and
// reasoning_details reach a Chat worker as the client sent them.
func (r *OpenAIRouter) dropReasoningForeignToClaude(
	request *llmprotocol.Request,
	dispatch *providerDispatch,
	source llmprotocol.WireFormat,
) int {
	if request == nil || dispatch == nil || r.armFamily(dispatch.logicalModel) != anthropicFamily {
		return 0
	}
	return protocolcodec.DropReasoningNotFromAnthropic(request, source)
}

// signClaudeThinkingForChat carries, for a Claude worker reached over Chat,
// each signed thinking block in the history as the anthropic-claude-v1
// reasoning_details item OpenRouter asks for back
// (protocolcodec.SignedThinkingAsReasoningDetails), so Claude reads its own
// thinking with the signature that verifies it. It runs after the foreign
// drop, so only reasoning with Anthropic provenance is left to rewrite, and
// before the disposition table, which would strip the signature. It returns
// how many blocks it rewrote.
func (r *OpenAIRouter) signClaudeThinkingForChat(request *llmprotocol.Request, dispatch *providerDispatch) int {
	if request == nil || dispatch == nil || dispatch.targetFormat != llmprotocol.OpenAIChatV1 ||
		r.armFamily(dispatch.logicalModel) != anthropicFamily {
		return 0
	}
	return protocolcodec.SignedThinkingAsReasoningDetails(request)
}

// restoreClaudeThinkingForMessages rebuilds, for a Claude worker reached over
// Messages, the thinking a Chat client replays as anthropic-claude-v1
// reasoning_details items into the signed thinking and redacted_thinking
// blocks Anthropic returned (protocolcodec.ClaudeThinkingFromReasoningDetails,
// atlasfutures/semantic-router#278). It is the mirror of
// signClaudeThinkingForChat: that one carries a Messages client's signed
// thinking to a Chat Claude arm, this one carries a Chat client's back to a
// Messages Claude arm. It runs after the foreign drop and before the
// disposition table, which would otherwise drop the reasoning as unsigned.
// It returns how many blocks it rebuilt.
func (r *OpenAIRouter) restoreClaudeThinkingForMessages(request *llmprotocol.Request, dispatch *providerDispatch) int {
	if request == nil || dispatch == nil || dispatch.targetFormat != llmprotocol.AnthropicMessagesV1 ||
		r.armFamily(dispatch.logicalModel) != anthropicFamily {
		return 0
	}
	return protocolcodec.ClaudeThinkingFromReasoningDetails(request)
}
