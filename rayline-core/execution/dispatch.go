package execution

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
	"github.com/atlasfutures/semantic-router/rayline-core/reasoning"
	"github.com/atlasfutures/semantic-router/rayline-core/thinkingcontrol"
)

type preparedDispatch struct {
	body         []byte
	placer       *thinkingcontrol.Placer
	key          string
	receipt      ControlReceipt
	nextIssuers  []string
	opaqueIssuer string
	response     responseMetadata
	// outputLimit is the prepared request's actual output allowance.
	outputLimit int64
}

type responseMetadata struct {
	publicModel   string
	streamOptions llmprotocol.StreamOptions
	namespaces    map[string][2]string
	workerFamily  string
}

func responseMetadataFor(request llmprotocol.Request, deployment Deployment) responseMetadata {
	metadata := responseMetadata{publicModel: request.Model, streamOptions: request.StreamOptions, namespaces: llmprotocol.ToolNamespaces(request.Tools)}
	if isAnthropicDeployment(deployment) {
		metadata.workerFamily = "anthropic"
	}
	return metadata
}

func (server *Server) prepare(request llmprotocol.Request, envelope llmprotocol.Envelope, selected admittedAction, state *branchState) (*preparedDispatch, error) {
	if request.PreviousResponseID != "" || request.ConversationID != "" {
		return nil, fmt.Errorf("hybrid execution requires complete request history; stored Responses references are not supported")
	}
	deployment := selected.deployment
	response := responseMetadataFor(request, deployment)
	nextIssuers, opaqueIssuer := prepareReasoning(&request, envelope.Format, deployment, state)
	request.Model = deployment.WireModel
	// The admitted v5 control owns native thinking. Clear the caller's neutral
	// settings before cross-format capability checks; Render writes the exact
	// native base after encoding and owns every private-append mutation.
	request.ReasoningMode = ""
	request.ReasoningEffort = ""
	request.ReasoningBudgetTokens = nil
	request.ReasoningDisplay = ""
	if request.Sampling.MaxOutputTokens == nil {
		request.Sampling.MaxOutputTokens = llmprotocol.Int64(deployment.MaxOutputTokens)
	}
	if *request.Sampling.MaxOutputTokens > deployment.MaxOutputTokens {
		return nil, fmt.Errorf("output allowance exceeds the admitted deployment limit")
	}
	if deployment.Protocol == llmprotocol.AnthropicMessagesV1 {
		thinkingcontrol.RaiseMessagesAllowance(&request, selected.control.BudgetTokens, deployment.MaxOutputTokens)
		if *request.Sampling.MaxOutputTokens > deployment.MaxOutputTokens {
			return nil, fmt.Errorf("thinking budget and output allowance exceed deployment limit")
		}
	}
	request = llmprotocol.WithUpstreamStreamUsage(request, deployment.Protocol)
	request.Generation++
	encoded, err := server.engine.EncodeRequest(deployment.Protocol, request, envelope)
	if err != nil {
		return nil, err
	}
	if deployment.InbandSystem == thinkingcontrol.InbandSystemFold {
		encoded.Body, err = thinkingcontrol.FoldInbandSystem(encoded.Body, selected.format)
		if err != nil {
			return nil, err
		}
	}
	key := thinkingcontrol.PlacementKey(deployment.ID+"|"+deployment.RuntimeIdentity, selected.format, selected.control)
	placer, err := thinkingcontrol.NewPlacer(selected.format)
	if previous, ok := state.placements[key]; ok {
		placer, err = thinkingcontrol.ResumePlacer(previous)
	}
	if err != nil {
		return nil, err
	}
	body, receipt, err := placer.Render(encoded.Body, &selected.control, selected.cell, deployment.WireModel)
	if err != nil {
		return nil, err
	}
	body, err = applyDeploymentWireBody(body, deployment)
	if err != nil {
		return nil, err
	}
	return &preparedDispatch{body: body, placer: placer, key: key, receipt: receipt,
		nextIssuers: nextIssuers, opaqueIssuer: opaqueIssuer, response: response, outputLimit: *request.Sampling.MaxOutputTokens}, nil
}

// prepareReasoning stages provenance before any codec drops or rewrites history.
// Its ordering matches cloud carryReasoningForDispatch, including filtering
// opaque blocks after Chat reasoning_details have been restored to Messages.
func prepareReasoning(request *llmprotocol.Request, source llmprotocol.WireFormat, deployment Deployment, state *branchState) ([]string, string) {
	nextIssuers := reasoning.ApplyEncryptedReasoning(request, state.issuers, encryptedDeploymentIssuer(deployment), deployment.Protocol)
	opaqueIssuer := opaqueDeploymentIssuer(deployment)
	foreignDropped := 0
	if deployment.Kind == "cloud" && cloudTransport(deployment) == cloudModelPin {
		foreignDropped = stripModelPinReasoning(request)
	}
	if isAnthropicDeployment(deployment) {
		foreignDropped += protocolcodec.DropReasoningNotFromAnthropic(request, source)
		switch deployment.Protocol {
		case llmprotocol.OpenAIChatV1:
			protocolcodec.SignedThinkingAsReasoningDetails(request)
		case llmprotocol.AnthropicMessagesV1:
			protocolcodec.ClaudeThinkingFromReasoningDetails(request)
		}
	}
	carry := protocolcodec.CarryReasoningTo(request, deployment.Protocol)
	opaqueDropped := reasoning.DropOpaqueReasoning(request, state.provenance, opaqueIssuer)
	if foreignDropped+opaqueDropped+carry.Dropped()+carry.SignaturesStripped > 0 {
		slog.Info("reasoning disposition", "deployment", deployment.ID,
			"removed", foreignDropped+opaqueDropped+carry.Dropped(), "signatures_removed", carry.SignaturesStripped)
	}
	return nextIssuers, opaqueIssuer
}

func isAnthropicDeployment(deployment Deployment) bool {
	return strings.HasPrefix(strings.TrimPrefix(deployment.RegistryModel, "anthropic/"), "claude-")
}

func encryptedDeploymentIssuer(deployment Deployment) string {
	if deployment.Kind != "cloud" || cloudTransport(deployment) == cloudModelPin {
		return reasoning.ReasoningIssuerUnknown
	}
	return reasoning.EncryptedIssuer(deployment.ID, deployment.URL, deployment.WireModel,
		deployment.IssuerSHA256, true, deployment.RegistryProvider == "openrouter")
}

func opaqueDeploymentIssuer(deployment Deployment) string {
	if deployment.Kind != "cloud" || cloudTransport(deployment) == cloudModelPin {
		return reasoning.ReasoningIssuerUnknown
	}
	// The dispatch service attests this exact approved profile's actual model,
	// provider and credential scope before generation and echoes it on response.
	// An unqualified or changing serving profile cannot satisfy that echo.
	return reasoning.OpaqueIssuer(deployment.ID, deployment.IssuerSHA256, deployment.WireModel, false, "")
}

func (server *Server) upstream(ctx context.Context, selected admittedAction, decision Decision, session, branch string, body []byte) (*http.Response, error) {
	deployment := selected.deployment
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, deployment.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	if deployment.AuthEnv != "" {
		request.Header.Set("Authorization", "Bearer "+os.Getenv(deployment.AuthEnv))
	}
	if deployment.Protocol == llmprotocol.AnthropicMessagesV1 {
		request.Header.Set("Anthropic-Version", "2023-06-01")
	}
	if deployment.Kind == "cloud" {
		if cloudTransport(deployment) == cloudModelPin {
			request.Header.Set("X-Workshop-Required-Model", deployment.WireModel)
		} else {
			request.Header.Set("X-Rayline-Dispatch-Profile", deployment.DispatchProfileSHA256)
			request.Header.Set("X-Rayline-Worker", decision.Worker)
			request.Header.Set("X-Rayline-Action-Id", decision.ActionID)
			request.Header.Set("X-Rayline-Route-Id", decision.RouteID)
		}
		request.Header.Set("X-Conversation-Id", session)
		request.Header.Set("X-Rayline-Branch-Id", branch)
	}
	return server.client.Do(request)
}

func dispatchEchoMatches(header http.Header, selected admittedAction, decision Decision) bool {
	if cloudTransport(selected.deployment) == cloudModelPin {
		return modelPinEchoMatches(header, selected.deployment)
	}
	return header.Get("X-Rayline-Dispatch-Profile") == selected.deployment.DispatchProfileSHA256 &&
		header.Get("X-Rayline-Worker") == decision.Worker && header.Get("X-Rayline-Action-Id") == decision.ActionID &&
		header.Get("X-Rayline-Dispatched-Model") == selected.deployment.WireModel &&
		header.Get("X-Rayline-Dispatch-Issuer") == selected.deployment.IssuerSHA256
}

// forwardError translates an upstream refusal received before any output. A
// provider's context-length refusal delivered on the Messages wire starts with
// Anthropic's "prompt is too long:", on which Claude Code compacts; the
// provider's own text follows.
func (server *Server) forwardError(writer http.ResponseWriter, target, source llmprotocol.WireFormat, upstream *http.Response, contextRefusal bool) {
	body, err := readBounded(upstream.Body, 1<<20)
	if err != nil {
		server.writeError(writer, target, http.StatusBadGateway, "upstream_error", "selected endpoint returned an unreadable error")
		return
	}
	translated, err := server.engine.TranslateTransportError(source, target, body, nil)
	if err != nil {
		server.writeError(writer, target, http.StatusBadGateway, "upstream_error", "selected endpoint returned an invalid error body")
		return
	}
	if contextRefusal && target == llmprotocol.AnthropicMessagesV1 {
		translated.Body = withPromptTooLong(translated.Body)
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(upstream.StatusCode)
	_, _ = writer.Write(translated.Body)
}

// isProviderContextRefusal recognizes a provider's refusal of a prompt over
// its context length (OpenAI context_length_exceeded, "maximum context
// length", Anthropic "prompt is too long", "input is too long").
func isProviderContextRefusal(status int, body []byte) bool {
	if status != http.StatusBadRequest && status != http.StatusRequestEntityTooLarge {
		return false
	}
	lower := bytes.ToLower(body)
	for _, marker := range []string{"context_length_exceeded", "maximum context length", "prompt is too long", "input is too long"} {
		if bytes.Contains(lower, []byte(marker)) {
			return true
		}
	}
	return false
}

// withPromptTooLong prefixes an Anthropic error body's message; any other
// shape is returned unchanged.
func withPromptTooLong(body []byte) []byte {
	var envelope map[string]any
	if json.Unmarshal(body, &envelope) != nil {
		return body
	}
	inner, ok := envelope["error"].(map[string]any)
	if !ok {
		return body
	}
	message, _ := inner["message"].(string)
	if strings.HasPrefix(strings.ToLower(message), "prompt is too long") {
		return body
	}
	inner["message"] = strings.TrimSuffix("prompt is too long: "+message, ": ")
	rewritten, err := json.Marshal(envelope)
	if err != nil {
		return body
	}
	return rewritten
}

func readBounded(reader io.Reader, limit int64) ([]byte, error) {
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, fmt.Errorf("response exceeds execution size limit")
	}
	return body, nil
}

func (server *Server) forwardBuffered(writer http.ResponseWriter, upstream *http.Response, target llmprotocol.WireFormat, selected admittedAction, metadata responseMetadata, pending *pendingTurn) (llmprotocol.Response, error) {
	body, err := readBounded(upstream.Body, 64<<20)
	if err != nil {
		server.writeError(writer, target, http.StatusBadGateway, "upstream_read_failed", "selected endpoint response was incomplete")
		return llmprotocol.Response{}, err
	}
	result, err := server.engine.TranslateResponse(selected.deployment.Protocol, target, body, func(response *llmprotocol.Response) error {
		response.Model = metadata.publicModel
		llmprotocol.RestoreResponseToolNamespaces(response, metadata.namespaces)
		return nil
	})
	if err != nil {
		server.writeError(writer, target, http.StatusBadGateway, "upstream_response_invalid", err.Error())
		return llmprotocol.Response{}, err
	}
	writer.Header().Set("Content-Type", "application/json")
	count, err := writer.Write(result.Body)
	pending.delivered = count > 0
	if err == nil && result.Response.Error != nil {
		err = result.Response.Error
	}
	return result.Response, err
}

func (server *Server) forwardStream(writer http.ResponseWriter, incoming *http.Request, upstream *http.Response, target llmprotocol.WireFormat, selected admittedAction, metadata responseMetadata, pending *pendingTurn) (llmprotocol.Response, error) {
	stream, err := server.engine.NewStreamWithMutation(selected.deployment.Protocol, target, llmprotocol.StreamContext{
		Context: incoming.Context(), PublicModel: metadata.publicModel, ProviderModel: selected.deployment.WireModel,
		Options: metadata.streamOptions, WorkerFamily: metadata.workerFamily,
	}, func(event *llmprotocol.Event) error {
		llmprotocol.RestoreToolNamespace(event.ToolCall, metadata.namespaces)
		return nil
	})
	if err != nil {
		server.writeError(writer, target, http.StatusBadGateway, "stream_translation_failed", err.Error())
		return llmprotocol.Response{}, err
	}
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")
	writer.Header().Set("X-Accel-Buffering", "no")
	controller := http.NewResponseController(writer)
	accumulator := protocolcodec.NewResponseAccumulator()
	write := func(frames [][]byte, events []llmprotocol.Event) error {
		if err := accumulator.Apply(events); err != nil {
			return err
		}
		for _, frame := range frames {
			count, err := writer.Write(frame)
			pending.delivered = pending.delivered || count > 0
			if err != nil {
				return err
			}
			if err := controller.Flush(); err != nil {
				return err
			}
		}
		return nil
	}
	buffer := make([]byte, 32<<10)
	var streamErr error
	for {
		count, readErr := upstream.Body.Read(buffer)
		if count > 0 {
			frames, events, _, decodeErr := stream.Push(buffer[:count])
			if err := write(frames, events); err != nil {
				streamErr = err
				break
			}
			if decodeErr != nil {
				streamErr = decodeErr
				break
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				streamErr = readErr
			}
			break
		}
	}
	frames, events, _, finalErr := stream.Finalize(streamErr)
	writeErr := write(frames, events)
	if streamErr != nil {
		return llmprotocol.Response{}, streamErr
	}
	if finalErr != nil {
		return llmprotocol.Response{}, finalErr
	}
	if writeErr != nil {
		return llmprotocol.Response{}, writeErr
	}
	response, err := accumulator.Response()
	if err == nil && response.Error != nil {
		err = response.Error
	}
	return response, err
}
