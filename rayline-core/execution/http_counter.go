package execution

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
	"github.com/atlasfutures/semantic-router/rayline-core/protocolcodec"
)

// TokenCounterConfig names a same-host HTTP counter for one local
// deployment. The counter renders the prepared request with the serving
// model's own chat template and counts it with the model's own tokenizer,
// images included, so its counts are local_measured.
//
// Wire contract (JSON over loopback HTTP):
//   - GET  url → {"units": "<token units>"}; the units must equal the
//     deployment's token_units, checked once before readiness.
//   - POST url, body: the prepared OpenAI Chat request exactly as it will be
//     dispatched → {"prompt_tokens": N, "units": "<token units>"}.
//
// A bearer credential, when auth_env names one, is sent on both.
//
// A Messages or Responses deployment's prepared request is posted in the
// OpenAI Chat form (chatFormForCounting); the request dispatched is
// unchanged. The runtime renders its own protocol through the same template
// but by its own conversion, which the Chat form only approximates, so such
// a count is raised by translatedCountAllowance.
type TokenCounterConfig struct {
	URL     string `json:"url"`
	AuthEnv string `json:"auth_env,omitempty"`
}

// The counter returns no token IDs, so it cannot identify a carried prefix;
// prefix overlap stays unknown and only capacity uses its count.
type httpCounter struct {
	identity cachecontext.PromptIdentity
	// engine translates a Messages or Responses request to the Chat form;
	// nil for a Chat deployment.
	engine     *protocolcodec.Engine
	url        string
	credential string
	client     *http.Client
}

const httpCounterTimeout = 30 * time.Second

func newHTTPCounter(ctx context.Context, deployment Deployment, config Config, engine *protocolcodec.Engine) (*httpCounter, error) {
	counterConfig := deployment.TokenCounter
	if counterConfig == nil || deployment.Kind != "local" || deployment.TokenUnits == "" {
		return nil, fmt.Errorf("token_counter requires a local deployment with token_units")
	}
	switch deployment.Protocol {
	case llmprotocol.OpenAIChatV1:
		engine = nil
	case llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIResponsesV1:
	default:
		return nil, fmt.Errorf("token_counter does not count %s requests", deployment.Protocol)
	}
	endpoint, err := url.Parse(counterConfig.URL)
	if err != nil {
		// The parse error quotes the URL, which may embed credentials.
		return nil, fmt.Errorf("token_counter url is not a valid URL")
	}
	host := endpoint.Hostname()
	ip := net.ParseIP(host)
	if endpoint.Scheme != "http" || (host != "localhost" && (ip == nil || !ip.IsLoopback())) || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		return nil, fmt.Errorf("token_counter url must be a same-host http URL without credentials or query")
	}
	counter := &httpCounter{
		identity: promptIdentity(deployment),
		engine:   engine,
		url:      endpoint.String(),
		client:   &http.Client{Timeout: httpCounterTimeout, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
	}
	if counterConfig.AuthEnv != "" {
		counter.credential = os.Getenv(counterConfig.AuthEnv)
		if counter.credential == "" {
			return nil, fmt.Errorf("token_counter auth_env %s is not set", counterConfig.AuthEnv)
		}
		// Like local inference, the counter never receives a hosted credential,
		// by name or by an alias holding the same value.
		hosted := []string{config.Decisions.AuthEnv}
		for _, other := range config.Deployments {
			if other.Kind != "local" {
				hosted = append(hosted, other.AuthEnv)
			}
		}
		for _, name := range hosted {
			if name != "" && (name == counterConfig.AuthEnv || os.Getenv(name) == counter.credential) {
				return nil, fmt.Errorf("token_counter auth_env must not carry a hosted credential")
			}
		}
	}
	var identity struct {
		Units string `json:"units"`
	}
	if err := counter.exchange(ctx, http.MethodGet, nil, &identity); err != nil {
		return nil, err
	}
	if identity.Units != deployment.TokenUnits {
		return nil, fmt.Errorf("token_counter counts in %q, the deployment in %q", identity.Units, deployment.TokenUnits)
	}
	return counter, nil
}

func (counter *httpCounter) Identity() cachecontext.PromptIdentity { return counter.identity }

func (counter *httpCounter) Count(ctx context.Context, prompt cachecontext.PreparedPrompt) (cachecontext.PromptMeasurement, error) {
	body, charged := prompt.Body, 0
	if counter.engine != nil {
		chat, dropped, err := chatFormForCounting(counter.engine, llmprotocol.WireFormat(counter.identity.Protocol), prompt.Body)
		if err != nil {
			return cachecontext.PromptMeasurement{}, err
		}
		body, charged = chat, dropped
	}
	var result struct {
		PromptTokens *int   `json:"prompt_tokens"`
		Units        string `json:"units"`
	}
	if err := counter.exchange(ctx, http.MethodPost, body, &result); err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	if result.PromptTokens == nil || *result.PromptTokens < 0 {
		return cachecontext.PromptMeasurement{}, fmt.Errorf("token_counter returned no prompt_tokens")
	}
	if result.Units != counter.identity.Units {
		return cachecontext.PromptMeasurement{}, fmt.Errorf("token_counter changed units to %q", result.Units)
	}
	tokens := *result.PromptTokens
	if counter.engine != nil {
		tokens += charged + translatedCountAllowance(tokens)
	}
	return cachecontext.PromptMeasurement{PromptTokens: tokens, Provenance: "local_measured"}, nil
}

// uncountedRequest is a prepared request whose Chat form would leave out
// something the runtime may render, so a count over it could fall short.
// Capacity then sizes the request by its bytes instead.
type uncountedRequest struct{ reason string }

func (uncounted *uncountedRequest) Error() string {
	return "token_counter cannot count this request: " + uncounted.reason
}

// countNeutralMembers are request members the Chat form drops that render no
// prompt text: routing, caching and response options, a message's own
// output_config (effort), and a tool's defer_loading flag (the Chat form
// keeps the tool, so it is counted). A message index reads as *.
var countNeutralMembers = setOf("client_metadata", "include", "prompt_cache_key", "reasoning.summary",
	"messages.*.output_config", "tools.defer_loading")

var anyIndex = regexp.MustCompile(`\.[0-9]+`)

// countChargedMembers are Messages members the Chat form drops that a runtime
// may render as text, charged by their bytes (droppedTextBytes) instead:
// Claude Code's billing line in system, and context_management, whose edits
// may carry compaction instructions.
var countChargedMembers = setOf("system.x-anthropic-billing-header", "context_management")

// chatFormForCounting translates a prepared Messages or Responses request to
// the OpenAI Chat request the counter reads, with its reasoning history
// carried as a Chat dispatch carries it (CarryReasoningTo): signatures
// removed, readable reasoning kept as reasoning_content. It also returns the
// tokens to charge for text the Chat form drops (countChargedMembers). A
// request whose translation fails, drops reasoning, carries a block of
// another format, or drops or approximates any other member that may render
// prompt text (a server tool, a document) is uncounted.
func chatFormForCounting(engine *protocolcodec.Engine, protocol llmprotocol.WireFormat, body []byte) ([]byte, int, error) {
	var carry protocolcodec.ReasoningCarry
	result, err := engine.TranslateRequest(protocol, llmprotocol.OpenAIChatV1, body, func(request *llmprotocol.Request) error {
		carry = protocolcodec.CarryReasoningTo(request, llmprotocol.OpenAIChatV1)
		return nil
	})
	if err != nil {
		return nil, 0, &uncountedRequest{reason: "no Chat form: " + err.Error()}
	}
	if carry.Dropped() > 0 {
		return nil, 0, &uncountedRequest{reason: "the Chat form drops reasoning"}
	}
	// The Chat form drops a carried block without always reporting it (one
	// in system, for instance), so any carried block is refused here.
	if kind := carriedBlock(result.Request); kind != "" {
		return nil, 0, &uncountedRequest{reason: "the request carries a " + logName(kind) + " block whole"}
	}
	for _, diagnostic := range result.Diagnostics {
		if member := anyIndex.ReplaceAllString(diagnostic.Field, ".*"); !countNeutralMembers[member] && !countChargedMembers[member] {
			return nil, 0, &uncountedRequest{reason: fmt.Sprintf("the Chat form %s %s", diagnostic.Action, diagnostic.Field)}
		}
	}
	charged, err := droppedTextBytes(body)
	if err != nil {
		return nil, 0, &uncountedRequest{reason: "its dropped text cannot be read"}
	}
	return result.Body, charged, nil
}

// carriedBlock is the type of the first content block the request carries
// whole in another format, in its instructions, messages or tool results.
func carriedBlock(request llmprotocol.Request) string {
	var find func([]llmprotocol.Content) string
	find = func(contents []llmprotocol.Content) string {
		for _, content := range contents {
			if content.Kind == llmprotocol.ContentUnmodeled && content.Unmodeled != nil {
				return content.Unmodeled.Type
			}
			if content.ToolResult != nil {
				if kind := find(content.ToolResult.Content); kind != "" {
					return kind
				}
			}
		}
		return ""
	}
	for _, block := range request.Instructions {
		if kind := find(block.Content); kind != "" {
			return kind
		}
	}
	for _, message := range request.Messages {
		if kind := find(message.Content); kind != "" {
			return kind
		}
	}
	return ""
}

// droppedTextBytes bounds the tokens of the text in a Messages request's
// countChargedMembers: Claude Code's billing line as system or as a system
// text block, and every string in context_management other than a type. Each
// is charged its worst-case bytes (worstCaseBytes, normalization included):
// a byte-level BPE token covers at least one byte.
func droppedTextBytes(body []byte) (int, error) {
	var request struct {
		System            any `json:"system"`
		ContextManagement any `json:"context_management"`
	}
	// Numbers stay text: one past float64's range must not fail the read.
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	if err := decoder.Decode(&request); err != nil {
		return 0, err
	}
	total := 0
	charge := func(text string) {
		quoted, _ := json.Marshal(text)
		total += int(worstCaseBytes(quoted))
	}
	billing := func(value any) {
		if text, _ := value.(string); strings.HasPrefix(text, "x-anthropic-billing-header:") {
			charge(text)
		}
	}
	billing(request.System)
	if blocks, isList := request.System.([]any); isList {
		for _, block := range blocks {
			if block, isObject := block.(map[string]any); isObject {
				billing(block["text"])
			}
		}
	}
	var addStrings func(any, string)
	addStrings = func(value any, key string) {
		switch value := value.(type) {
		case string:
			if key != "type" {
				charge(value)
			}
		case []any:
			for _, element := range value {
				addStrings(element, key)
			}
		case map[string]any:
			for member, element := range value {
				addStrings(element, member)
			}
		}
	}
	addStrings(request.ContextManagement, "")
	return total, nil
}

// translatedCountAllowance raises a count over the Chat form for what the
// runtime's own conversion of its protocol may render beyond it: the byte
// bound's headroom ratio and template margin. It is a default, not a
// measured drift; a deployment adds capacity_margin_tokens on top.
func translatedCountAllowance(tokens int) int {
	return int(math.Ceil(float64(tokens)*defaultEstimateHeadroomRatio)) + estimateTemplateMarginTokens
}

func (counter *httpCounter) exchange(ctx context.Context, method string, body []byte, result any) error {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, method, counter.url, reader)
	if err != nil {
		return err
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	if counter.credential != "" {
		request.Header.Set("Authorization", "Bearer "+counter.credential)
	}
	response, err := counter.client.Do(request)
	if err != nil {
		// url.Error quotes the request URL; report only what failed.
		var requestErr *url.Error
		if errors.As(err, &requestErr) {
			err = requestErr.Err
		}
		return fmt.Errorf("token_counter request failed: %w", err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil {
		return fmt.Errorf("token_counter: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("token_counter answered %d", response.StatusCode)
	}
	if err := json.Unmarshal(payload, result); err != nil {
		return fmt.Errorf("token_counter: %w", err)
	}
	return nil
}
