package execution

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

func validateInbandSystem(deployment Deployment, control thinkingcontrol.Control) error {
	if deployment.InbandSystem == "" {
		return nil
	}
	if deployment.InbandSystem != thinkingcontrol.InbandSystemFold || !foldsInbandSystem(deployment, control) {
		return fmt.Errorf("inband_system only supports fold_before_unit_v1 on a local Chat deployment with the native default control")
	}
	if !strings.HasSuffix(deployment.RuntimeIdentity, ":"+thinkingcontrol.InbandSystemFold) {
		return fmt.Errorf("runtime_identity must include the qualified inband_system rule suffix")
	}
	return nil
}

// foldsInbandSystem reports whether fold_before_unit_v1 applies to this
// deployment and control: a local Chat deployment under the native default
// control, without a steering instruction or budget. Local chat templates
// (Qwen's among them) refuse a system message anywhere but first, and Claude
// Code sends one after the current turn; the fold keeps the conversation
// servable there. Cloud deployments are never folded.
func foldsInbandSystem(deployment Deployment, control thinkingcontrol.Control) bool {
	return deployment.Kind == "local" && deployment.Protocol == llmprotocol.OpenAIChatV1 &&
		control.Native == "default" && control.Instruction == nil && control.BudgetTokens == nil
}

// foldInbandSystem applies fold_before_unit_v1 to an encoded Chat body. An
// explicitly configured rule applies to every request and its refusal fails
// the plan. Otherwise a local Chat deployment folds only a body that has an
// in-band system message (any other body is sent byte for byte as before),
// and a refused fold leaves the body as the client ordered it: a runtime that
// cannot serve that order fails before output and the turn falls back.
func foldInbandSystem(body []byte, deployment Deployment, selected admittedAction) ([]byte, error) {
	if deployment.InbandSystem == thinkingcontrol.InbandSystemFold {
		return thinkingcontrol.FoldInbandSystem(body, selected.format)
	}
	if !foldsInbandSystem(deployment, selected.control) || !hasInbandSystem(body) {
		return body, nil
	}
	folded, err := thinkingcontrol.FoldInbandSystem(body, selected.format)
	if err != nil {
		return body, nil
	}
	return folded, nil
}

// hasInbandSystem reports whether a Chat body has a system or developer
// message after its first other message.
func hasInbandSystem(body []byte) bool {
	var chat struct {
		Messages []struct {
			Role string `json:"role"`
		} `json:"messages"`
	}
	if json.Unmarshal(body, &chat) != nil {
		return false
	}
	leading := true
	for _, message := range chat.Messages {
		system := message.Role == "system" || message.Role == "developer"
		if !system {
			leading = false
		} else if !leading {
			return true
		}
	}
	return false
}
