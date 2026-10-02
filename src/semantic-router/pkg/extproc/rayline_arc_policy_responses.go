package extproc

import (
	"encoding/json"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

var (
	errPolicyResponsesInputEmpty = errors.New("a Responses request materialized no input items")
	// errPolicyResponsesHistoryUnavailable refuses a request that names
	// earlier turns this path cannot load: deciding on the current input
	// alone would score, and commit, a truncated conversation.
	errPolicyResponsesHistoryUnavailable = errors.New("previous_response_id history is not available on this path")
)

// raylineARCPolicyResponsesInput materializes a Responses request for the
// policy service: the stored history previous_response_id resolves to, then
// this turn's input, and the instructions.
//
// The items a turn sends must be a byte prefix of the items the next turn
// sends: the attribution ledger identifies its prefix by those bytes, and
// anything less stable would open a new context epoch on every turn. So every
// item is encoded one way, whether it comes from the store or from this
// turn's body: through the snapshot persistence uses, with the router's
// retention fields (item id, status) left out. Ids matter most: an input item
// without one is given a fresh random id on every snapshot, so a client that
// resends its history without ids would otherwise never repeat a byte.
func (r *OpenAIRouter) raylineARCPolicyResponsesInput(reqCtx *RequestContext) ([]json.RawMessage, *string, error) {
	var (
		history      []*responseapi.StoredResponse
		input        []responseapi.InputItem
		instructions string
	)
	switch {
	case reqCtx.ResponseObjectState != nil:
		state := reqCtx.ResponseObjectState
		history, input, instructions = state.ConversationHistory, state.Input, state.Instructions
	case reqCtx.SemanticRequest != nil:
		if reqCtx.SemanticRequest.PreviousResponseID != "" {
			return nil, nil, errPolicyResponsesHistoryUnavailable
		}
		input, instructions = snapshotResponseObjectRequest(reqCtx.RaylineARCRawBody, *reqCtx.SemanticRequest)
	}
	items := make([]json.RawMessage, 0)
	for _, stored := range history {
		if stored == nil {
			continue
		}
		storedInput, err := policyResponsesInputItems(stored.Input)
		if err != nil {
			return nil, nil, err
		}
		items = append(items, storedInput...)
		for _, output := range stored.Output {
			output.ID, output.Status = "", ""
			encoded, err := raylinearc.MarshalUnescaped(output)
			if err != nil {
				return nil, nil, err
			}
			items = append(items, raylinearc.UnescapeHTMLEscapes(encoded))
		}
		if len(stored.Output) == 0 && stored.OutputText != "" {
			encoded, err := raylinearc.MarshalUnescaped(storedOutputTextItem(stored.OutputText))
			if err != nil {
				return nil, nil, err
			}
			items = append(items, raylinearc.UnescapeHTMLEscapes(encoded))
		}
	}
	current, err := policyResponsesInputItems(input)
	if err != nil {
		return nil, nil, err
	}
	items = append(items, current...)
	if len(items) == 0 {
		return nil, nil, errPolicyResponsesInputEmpty
	}
	if instructions == "" {
		return items, nil, nil
	}
	return items, &instructions, nil
}

// storedOutputTextItem is a stored response's bare output text as the
// assistant message item it stands for (marshalStoredOutputText's item).
func storedOutputTextItem(text string) responseapi.OutputItem {
	return responseapi.OutputItem{
		Type: responseapi.ItemTypeMessage,
		Role: responseapi.RoleAssistant,
		Content: []responseapi.ContentPart{{
			Type: responseapi.ContentTypeOutputText,
			Text: text,
		}},
	}
}

func policyResponsesInputItems(items []responseapi.InputItem) ([]json.RawMessage, error) {
	encoded := make([]json.RawMessage, 0, len(items))
	for _, item := range items {
		item.ID, item.Status = "", ""
		body, err := raylinearc.MarshalUnescaped(item)
		if err != nil {
			return nil, err
		}
		// A stored item's raw fields may hold another encoder's escapes;
		// normalized, a turn's items stay a byte prefix of the next turn's.
		encoded = append(encoded, raylinearc.UnescapeHTMLEscapes(body))
	}
	return encoded, nil
}

// policyResponsesRoles reads, per item, the role attribution checks: any
// assistant-output item -- an assistant message, a function call, or
// reasoning -- counts as "assistant", since a response may hold only a call
// (pathfinder arc_policy_contract.attributable).
func policyResponsesRoles(items []json.RawMessage) ([]string, error) {
	roles := make([]string, len(items))
	for index, item := range items {
		var envelope struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		if err := json.Unmarshal(item, &envelope); err != nil {
			return nil, err
		}
		switch envelope.Type {
		case "function_call", "reasoning":
			roles[index] = "assistant"
		case "", "message":
			roles[index] = envelope.Role
		}
	}
	return roles, nil
}
