package extproc

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// openRouterWebPlugin is OpenRouter's web search plugin with its defaults: the
// engine OpenRouter picks and five results. OpenRouter bills it per request.
var openRouterWebPlugin = json.RawMessage(`{"id":"web"}`)

// applyOpenRouterWebSearch turns a client's web_search declaration into
// OpenRouter's web plugin on a Chat Completions dispatch to OpenRouter. The
// model must be admitted with web_search (the card's hosted_tools), and the
// declaration must ask for nothing the plugin cannot honour; see
// protocolcodec.UnrestrictedWebSearch. The codec has already dropped the
// declaration from the Chat body, since Chat names no hosted tool. The
// plugin's findings come back as url_citation annotations, which a
// Responses client receives as they are.
func applyOpenRouterWebSearch(body []byte, dispatch *providerDispatch, ctx *RequestContext) ([]byte, error) {
	if dispatch == nil || ctx == nil || ctx.SemanticRequest == nil ||
		dispatch.targetFormat != llmprotocol.OpenAIChatV1 || !providerIsOpenRouter(dispatch.profile) ||
		!slices.Contains(ctx.DispatchHostedTools, "web_search") || !declaresUnrestrictedWebSearch(ctx.SemanticRequest) {
		return body, nil
	}
	var request map[string]json.RawMessage
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}
	var plugins []json.RawMessage
	if existing := request["plugins"]; len(existing) > 0 {
		if err := json.Unmarshal(existing, &plugins); err != nil {
			return nil, fmt.Errorf("failed to parse request plugins: %w", err)
		}
	}
	for _, plugin := range plugins {
		var named struct {
			ID string `json:"id"`
		}
		if json.Unmarshal(plugin, &named) == nil && named.ID == "web" {
			return body, nil
		}
	}
	plugins = append(plugins, openRouterWebPlugin)
	encoded, err := json.Marshal(plugins)
	if err != nil {
		return nil, err
	}
	request["plugins"] = encoded
	return json.Marshal(request)
}

func declaresUnrestrictedWebSearch(request *llmprotocol.Request) bool {
	for _, tool := range request.CarriedTools {
		if protocolcodec.UnrestrictedWebSearch(tool) {
			return true
		}
	}
	return false
}
