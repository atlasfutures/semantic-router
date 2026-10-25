package execution

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// LlamaCPPConfig qualifies a same-host text runtime. Model bytes are verified
// once at startup; serving build/template and file metadata are checked for
// every measurement. These pins do not attest physical KV cache residency.
type LlamaCPPConfig struct {
	ModelPath          string `json:"model_path"`
	ModelSHA256        string `json:"model_sha256"`
	BuildInfo          string `json:"build_info"`
	ChatTemplateSHA256 string `json:"chat_template_sha256"`
	ToolTemplateSHA256 string `json:"tool_template_sha256,omitempty"`
}

type llamaCPPCounter struct {
	identity   cachecontext.PromptIdentity
	config     LlamaCPPConfig
	baseURL    string
	credential string
	file       os.FileInfo
	client     *http.Client
}

func (server *Server) RegisterConfiguredPromptCounters(ctx context.Context) error {
	for _, deployment := range server.config.Deployments {
		if deployment.PromptCounter == nil {
			continue
		}
		counter, err := newLlamaCPPCounter(ctx, deployment)
		if err != nil {
			return fmt.Errorf("qualify prompt counter for %s: %w", deployment.ID, err)
		}
		if err := server.RegisterPromptCounter(deployment.ID, counter); err != nil {
			return err
		}
	}
	return nil
}

func newLlamaCPPCounter(ctx context.Context, deployment Deployment) (*llamaCPPCounter, error) {
	config := deployment.PromptCounter
	if config == nil || deployment.Kind != "local" || deployment.Protocol != llmprotocol.OpenAIChatV1 || deployment.TokenUnits == "" {
		return nil, fmt.Errorf("llama.cpp counter requires a local Chat deployment with qualified token units")
	}
	endpoint, err := url.Parse(deployment.URL)
	if err != nil {
		return nil, err
	}
	host := endpoint.Hostname()
	ip := net.ParseIP(host)
	if (host != "localhost" && (ip == nil || !ip.IsLoopback())) || endpoint.Path != "/v1/chat/completions" || endpoint.RawQuery != "" {
		return nil, fmt.Errorf("llama.cpp counter requires the same-host /v1/chat/completions endpoint")
	}
	if !filepath.IsAbs(config.ModelPath) || !validSHA256(config.ModelSHA256) || !validSHA256(config.ChatTemplateSHA256) || config.BuildInfo != "b11840-36e36e63a" || (config.ToolTemplateSHA256 != "" && !validSHA256(config.ToolTemplateSHA256)) {
		return nil, fmt.Errorf("llama.cpp counter requires absolute model path, SHA256 pins and qualified build b11840-36e36e63a")
	}
	endpoint.Path = ""
	counter := &llamaCPPCounter{identity: promptIdentity(deployment), config: *config, baseURL: endpoint.String(), client: &http.Client{Timeout: 30 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}}
	if deployment.AuthEnv != "" {
		counter.credential = os.Getenv(deployment.AuthEnv)
	}
	if counter.file, err = verifyModelFile(ctx, config.ModelPath, config.ModelSHA256); err != nil {
		return nil, err
	}
	if err := counter.verifyProperties(ctx); err != nil {
		return nil, err
	}
	if err := counter.verifyNoAssistantPrefill(ctx); err != nil {
		return nil, err
	}
	return counter, nil
}

func (counter *llamaCPPCounter) Identity() cachecontext.PromptIdentity { return counter.identity }

func (counter *llamaCPPCounter) Count(ctx context.Context, prompt cachecontext.PreparedPrompt) (cachecontext.PromptMeasurement, error) {
	if err := counter.verifyProperties(ctx); err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	if err := counter.verifyNoAssistantPrefill(ctx); err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	var body map[string]json.RawMessage
	if err := json.Unmarshal(prompt.Body, &body); err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(body["messages"], &messages); err != nil || len(messages) == 0 {
		return cachecontext.PromptMeasurement{}, fmt.Errorf("llama.cpp count requires Chat messages")
	}
	for _, raw := range messages {
		var message struct {
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(raw, &message); err != nil {
			return cachecontext.PromptMeasurement{}, err
		}
		var blocks []struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(message.Content, &blocks) == nil {
			for _, block := range blocks {
				if block.Type != "text" {
					return cachecontext.PromptMeasurement{}, fmt.Errorf("llama.cpp text counter cannot count multimodal input")
				}
			}
		}
	}
	full, err := counter.applyTemplate(ctx, body)
	if err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	tokens, err := counter.tokenize(ctx, full)
	if err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	measurement := cachecontext.PromptMeasurement{PromptTokens: len(tokens), TokenIDs: tokens, Provenance: "provider_reported"}
	// Only a new user-message boundary is relevant to multiwarm costing.
	// Adjacent user messages are one input run; exclude the entire run.
	boundary := len(messages)
	for boundary > 0 {
		var message struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(messages[boundary-1], &message); err != nil {
			return cachecontext.PromptMeasurement{}, err
		}
		if message.Role != "user" {
			break
		}
		boundary--
	}
	// A system-only preamble is not a prior conversation; Qwen's qualified
	// template requires a user query. Start boundaries carry no worker cache
	// evidence, so keep that optional prefix unknown instead of inventing a
	// synthetic user message for the tokenizer.
	hasPreviousUser := false
	for _, raw := range messages[:boundary] {
		var message struct {
			Role string `json:"role"`
		}
		if err := json.Unmarshal(raw, &message); err != nil {
			return cachecontext.PromptMeasurement{}, err
		}
		hasPreviousUser = hasPreviousUser || message.Role == "user"
	}
	if hasPreviousUser && boundary > 0 && boundary < len(messages) {
		body["messages"], _ = json.Marshal(messages[:boundary])
		body["add_generation_prompt"] = json.RawMessage("false")
		history, err := counter.applyTemplate(ctx, body)
		if err != nil {
			return cachecontext.PromptMeasurement{}, err
		}
		historyTokens, err := counter.tokenize(ctx, history)
		if err != nil {
			return cachecontext.PromptMeasurement{}, err
		}
		// Template-dependent rewrites (including earlier reasoning removal)
		// can change the history. Do not label its length as the carried prefix.
		if cachecontext.TokenPrefixLength(historyTokens, tokens) == len(historyTokens) {
			count := len(historyTokens)
			measurement.CarriedPrefixTokens = &count
		}
	}
	if err := counter.verifyProperties(ctx); err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	return measurement, nil
}

func (counter *llamaCPPCounter) applyTemplate(ctx context.Context, body any) (string, error) {
	var result struct {
		Prompt *string `json:"prompt"`
	}
	if err := counter.exchange(ctx, "/apply-template", body, &result); err != nil {
		return "", err
	}
	if result.Prompt == nil {
		return "", fmt.Errorf("llama.cpp did not return a text prompt")
	}
	return *result.Prompt, nil
}

func (counter *llamaCPPCounter) tokenize(ctx context.Context, prompt string) ([]int, error) {
	var result struct {
		Tokens *[]int `json:"tokens"`
	}
	// Match handle_completions_impl in the pinned build. /tokenize's own
	// add_special default is false and would omit the serving BOS/EOS policy.
	if err := counter.exchange(ctx, "/tokenize", map[string]any{"content": prompt, "add_special": true, "parse_special": true, "with_pieces": false}, &result); err != nil {
		return nil, err
	}
	if result.Tokens == nil {
		return nil, fmt.Errorf("llama.cpp did not return token IDs")
	}
	return *result.Tokens, nil
}

func (counter *llamaCPPCounter) verifyNoAssistantPrefill(ctx context.Context) error {
	body := map[string]any{"model": counter.identity.Model, "messages": []map[string]string{{"role": "user", "content": "qualification probe"}, {"role": "assistant", "content": "qualification answer"}}, "chat_template_kwargs": map[string]bool{"enable_thinking": false}, "add_generation_prompt": false}
	closed, err := counter.applyTemplate(ctx, body)
	if err != nil {
		return err
	}
	body["add_generation_prompt"] = true
	opened, err := counter.applyTemplate(ctx, body)
	if err != nil {
		return err
	}
	if closed == opened || !strings.HasPrefix(opened, closed) {
		return fmt.Errorf("llama.cpp template does not prove --no-prefill-assistant with an explicit generation boundary")
	}
	return nil
}

func (counter *llamaCPPCounter) verifyProperties(ctx context.Context) error {
	file, err := os.Stat(counter.config.ModelPath)
	if err != nil || !os.SameFile(counter.file, file) || file.Size() != counter.file.Size() || !file.ModTime().Equal(counter.file.ModTime()) {
		return fmt.Errorf("qualified llama.cpp model file changed")
	}
	var props struct {
		ModelAlias    string                       `json:"model_alias"`
		ModelPath     string                       `json:"model_path"`
		BuildInfo     string                       `json:"build_info"`
		ChatTemplate  string                       `json:"chat_template"`
		ToolTemplate  string                       `json:"chat_template_tool_use"`
		EndpointProps bool                         `json:"endpoint_props"`
		Modalities    struct{ Vision, Audio bool } `json:"modalities"`
	}
	if err := counter.exchange(ctx, "/props", nil, &props); err != nil {
		return err
	}
	servingFile, err := os.Stat(props.ModelPath)
	if err != nil || !os.SameFile(counter.file, servingFile) || props.ModelAlias != counter.identity.Model || props.BuildInfo != counter.config.BuildInfo || props.EndpointProps || props.Modalities.Vision || props.Modalities.Audio || fingerprint([]byte(props.ChatTemplate)) != counter.config.ChatTemplateSHA256 {
		return fmt.Errorf("llama.cpp serving model, build or template differs from the qualified text runtime")
	}
	if (props.ToolTemplate == "" && counter.config.ToolTemplateSHA256 != "") || (props.ToolTemplate != "" && fingerprint([]byte(props.ToolTemplate)) != counter.config.ToolTemplateSHA256) {
		return fmt.Errorf("llama.cpp tool template differs from its qualification")
	}
	return nil
}

func (counter *llamaCPPCounter) exchange(ctx context.Context, path string, value, result any) error {
	method := http.MethodGet
	var body io.Reader
	if value != nil {
		encoded, err := json.Marshal(value)
		if err != nil {
			return err
		}
		body, method = bytes.NewReader(encoded), http.MethodPost
	}
	request, err := http.NewRequestWithContext(ctx, method, counter.baseURL+path, body)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	if counter.credential != "" {
		request.Header.Set("Authorization", "Bearer "+counter.credential)
	}
	response, err := counter.client.Do(request)
	if err != nil {
		return fmt.Errorf("llama.cpp %s request failed: %w", path, err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("llama.cpp %s returned status %d", path, response.StatusCode)
	}
	decoder := json.NewDecoder(io.LimitReader(response.Body, 64<<20))
	if err := decoder.Decode(result); err != nil {
		return fmt.Errorf("decode llama.cpp %s: %w", path, err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return fmt.Errorf("llama.cpp %s returned trailing or oversized data", path)
	}
	return nil
}

func verifyModelFile(ctx context.Context, path, expected string) (os.FileInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	before, err := file.Stat()
	if err != nil || !before.Mode().IsRegular() {
		return nil, fmt.Errorf("llama.cpp model must be a regular file")
	}
	hash := sha256.New()
	buffer := make([]byte, 1<<20)
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n, err := file.Read(buffer)
		_, _ = hash.Write(buffer[:n])
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
	}
	after, err := file.Stat()
	if err != nil || before.Size() != after.Size() || !before.ModTime().Equal(after.ModTime()) || hex.EncodeToString(hash.Sum(nil)) != expected {
		return nil, fmt.Errorf("llama.cpp model bytes do not match their SHA256 pin")
	}
	return after, nil
}
