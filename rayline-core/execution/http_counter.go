package execution

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/atlasfutures/semantic-router/rayline-core/cachecontext"
	"github.com/atlasfutures/semantic-router/rayline-core/llmprotocol"
)

// TokenCounterConfig names a same-host HTTP counter for one local Chat
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
type TokenCounterConfig struct {
	URL     string `json:"url"`
	AuthEnv string `json:"auth_env,omitempty"`
}

// The counter returns no token IDs, so it cannot identify a carried prefix;
// prefix overlap stays unknown and only capacity uses its count.
type httpCounter struct {
	identity   cachecontext.PromptIdentity
	url        string
	credential string
	client     *http.Client
}

const httpCounterTimeout = 30 * time.Second

func newHTTPCounter(ctx context.Context, deployment Deployment, config Config) (*httpCounter, error) {
	counterConfig := deployment.TokenCounter
	if counterConfig == nil || deployment.Kind != "local" || deployment.Protocol != llmprotocol.OpenAIChatV1 || deployment.TokenUnits == "" {
		return nil, fmt.Errorf("token_counter requires a local Chat deployment with token_units")
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
	var result struct {
		PromptTokens *int   `json:"prompt_tokens"`
		Units        string `json:"units"`
	}
	if err := counter.exchange(ctx, http.MethodPost, prompt.Body, &result); err != nil {
		return cachecontext.PromptMeasurement{}, err
	}
	if result.PromptTokens == nil || *result.PromptTokens < 0 {
		return cachecontext.PromptMeasurement{}, fmt.Errorf("token_counter returned no prompt_tokens")
	}
	if result.Units != counter.identity.Units {
		return cachecontext.PromptMeasurement{}, fmt.Errorf("token_counter changed units to %q", result.Units)
	}
	return cachecontext.PromptMeasurement{PromptTokens: *result.PromptTokens, Provenance: "local_measured"}, nil
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
		return fmt.Errorf("token_counter: %w", err)
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
