package execution

import (
	"fmt"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

func validateInbandSystem(deployment Deployment, control thinkingcontrol.Control) error {
	if deployment.InbandSystem == "" {
		return nil
	}
	if deployment.InbandSystem != thinkingcontrol.InbandSystemFold ||
		deployment.Kind != "local" || deployment.Protocol != llmprotocol.OpenAIChatV1 ||
		control.Native != "default" || control.Instruction != nil || control.BudgetTokens != nil {
		return fmt.Errorf("inband_system only supports fold_before_unit_v1 on a local Chat deployment with the native default control")
	}
	if !strings.HasSuffix(deployment.RuntimeIdentity, ":"+thinkingcontrol.InbandSystemFold) {
		return fmt.Errorf("runtime_identity must include the qualified inband_system rule suffix")
	}
	return nil
}
