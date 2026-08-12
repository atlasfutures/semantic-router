package raylineremote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"
)

const maxRemoteRetries = 2

// WorkerContract is the VSR-owned dispatch identity expected at Pathfinder
// readiness. Backend is optional only when the provider type cannot be
// expressed by the current VSR endpoint surface.
type WorkerContract struct {
	ID             string
	Model          string
	Backend        string
	ThinkingMode   string
	AllowFallbacks bool
	Prices         *WorkerPrices
}

type WorkerPrices struct {
	Input      float64
	CacheRead  float64
	CacheWrite float64
	Output     float64
}

type ClientConfig struct {
	BaseURL       string
	BundleVersion string
	// AllowInsecureTransport permits a plaintext BaseURL. Prepare sends the
	// caller's whole message and tool set, so http is refused by default.
	AllowInsecureTransport bool
	APIKey                 string
	ConnectTimeout         time.Duration
	RequestTimeout         time.Duration
	LeaseTTL               time.Duration
	MaxRetries             int
	Workers                []WorkerContract
}

type Client struct {
	config     ClientConfig
	baseURL    *url.URL
	httpClient *http.Client
	now        func() time.Time
}

func NewClient(config ClientConfig) (*Client, error) {
	dialer := &net.Dialer{Timeout: config.ConnectTimeout}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.DialContext = dialer.DialContext
	return newClient(config, &http.Client{Transport: transport})
}

func newClient(
	config ClientConfig,
	httpClient *http.Client,
) (*Client, error) {
	baseURL, err := parseBaseURL(
		config.BaseURL,
		config.AllowInsecureTransport,
	)
	if err != nil {
		return nil, err
	}
	if err := validateClientConfig(config, httpClient); err != nil {
		return nil, err
	}
	config.Workers = append([]WorkerContract(nil), config.Workers...)
	return &Client{
		config:     config,
		baseURL:    baseURL,
		httpClient: httpClient,
		now:        time.Now,
	}, nil
}

func parseBaseURL(
	raw string,
	allowInsecureTransport bool,
) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") ||
		parsed.User != nil || parsed.RawQuery != "" ||
		parsed.Fragment != "" {
		return nil, newFailure(
			FailureRequest,
			"initialize",
			"invalid_base_url",
			0,
		)
	}
	if parsed.Scheme == "http" && !allowInsecureTransport {
		return nil, initializeFailure("insecure_base_url")
	}
	return parsed, nil
}

func validateClientConfig(
	config ClientConfig,
	httpClient *http.Client,
) error {
	if httpClient == nil {
		return initializeFailure("http_client_required")
	}
	if err := validateClientIdentity(config); err != nil {
		return err
	}
	if err := validateClientTiming(config); err != nil {
		return err
	}
	return validateClientWorkers(config.Workers)
}

func validateClientIdentity(config ClientConfig) error {
	if strings.TrimSpace(config.BundleVersion) == "" ||
		len(config.BundleVersion) > 512 ||
		strings.TrimSpace(config.APIKey) == "" ||
		len(config.APIKey) > 4_096 {
		return initializeFailure("invalid_identity")
	}
	return nil
}

func validateClientTiming(config ClientConfig) error {
	if config.ConnectTimeout <= 0 ||
		config.RequestTimeout <= 0 ||
		config.ConnectTimeout > config.RequestTimeout ||
		config.LeaseTTL < 5*time.Second ||
		config.RequestTimeout*2 > config.LeaseTTL {
		return initializeFailure("invalid_timeout")
	}
	if config.MaxRetries < 0 || config.MaxRetries > maxRemoteRetries {
		return initializeFailure("invalid_retries")
	}
	return nil
}

func validateClientWorkers(workers []WorkerContract) error {
	if len(workers) < 2 || len(workers) > maxCandidates {
		return initializeFailure("invalid_workers")
	}
	ids := make(map[string]bool, len(workers))
	models := make(map[string]bool, len(workers))
	for _, worker := range workers {
		if !validWorkerContract(worker) ||
			ids[worker.ID] ||
			models[worker.Model] {
			return initializeFailure("invalid_workers")
		}
		ids[worker.ID] = true
		models[worker.Model] = true
	}
	return nil
}

func validWorkerContract(worker WorkerContract) bool {
	return strings.TrimSpace(worker.ID) != "" &&
		len(worker.ID) <= 256 &&
		strings.TrimSpace(worker.Model) != "" &&
		len(worker.Model) <= 512 &&
		strings.TrimSpace(worker.ThinkingMode) != "" &&
		len(worker.ThinkingMode) <= 64 &&
		(worker.Prices == nil || validWorkerPrices(workerPriceWire{
			Input:      worker.Prices.Input,
			CacheRead:  worker.Prices.CacheRead,
			CacheWrite: worker.Prices.CacheWrite,
			Output:     worker.Prices.Output,
		}))
}

func initializeFailure(reason string) error {
	return newFailure(FailureRequest, "initialize", reason, 0)
}

func (client *Client) endpoint(path string) string {
	endpoint := *client.baseURL
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + path
	endpoint.RawPath = ""
	return endpoint.String()
}

type prepareWireRequest struct {
	SchemaVersion string      `json:"schema_version"`
	DecisionID    string      `json:"decision_id"`
	EpisodeKey    string      `json:"episode_key"`
	BundleVersion string      `json:"bundle_version"`
	Candidates    []string    `json:"candidates"`
	Request       ChatRequest `json:"request"`
}

type prepareWireResponse struct {
	SchemaVersion  string `json:"schema_version"`
	DecisionID     string `json:"decision_id"`
	Receipt        string `json:"receipt"`
	State          string `json:"state"`
	SelectedWorker string `json:"selected_worker"`
	Policy         string `json:"policy"`
	RouteCallIndex int    `json:"route_call_index"`
	BundleVersion  string `json:"bundle_version"`
	LeaseExpiresAt string `json:"lease_expires_at"`
}

// Prepare selects one configured worker without advancing Pathfinder's
// committed episode state.
func (client *Client) Prepare(
	ctx context.Context,
	input PrepareInput,
) (*Transaction, PrepareResult, error) {
	if client == nil ||
		!decisionIDPattern.MatchString(input.DecisionID) ||
		!episodeKeyPattern.MatchString(input.EpisodeKey) ||
		!client.validCandidates(input.Candidates) ||
		validateChatRequest(input.Request) != nil {
		return nil, PrepareResult{}, newFailure(
			FailureRequest,
			"prepare",
			"invalid_prepare",
			0,
		)
	}
	request := prepareWireRequest{
		SchemaVersion: TransactionSchemaVersion,
		DecisionID:    input.DecisionID,
		EpisodeKey:    input.EpisodeKey,
		BundleVersion: client.config.BundleVersion,
		Candidates:    append([]string(nil), input.Candidates...),
		Request:       input.Request,
	}
	var response prepareWireResponse
	if err := client.post(
		ctx,
		"prepare",
		"/v1/route/prepare",
		request,
		&response,
	); err != nil {
		return nil, PrepareResult{}, err
	}
	expiresAt, err := client.validatePrepareResponse(input, response)
	if err != nil {
		return nil, PrepareResult{}, err
	}
	transaction := newTransaction(
		client,
		response.DecisionID,
		response.Receipt,
		response.SelectedWorker,
		response.RouteCallIndex,
		expiresAt,
	)
	return transaction, PrepareResult{
		SelectedWorker: response.SelectedWorker,
		RouteCallIndex: response.RouteCallIndex,
		Policy:         response.Policy,
	}, nil
}

func (client *Client) validCandidates(candidates []string) bool {
	if len(candidates) < 2 ||
		len(candidates) != len(client.config.Workers) {
		return false
	}
	expected := make([]string, 0, len(client.config.Workers))
	for _, worker := range client.config.Workers {
		expected = append(expected, worker.ID)
	}
	return slices.Equal(candidates, expected)
}

func (client *Client) validatePrepareResponse(
	input PrepareInput,
	response prepareWireResponse,
) (time.Time, error) {
	if !client.validPrepareMetadata(input, response) {
		return time.Time{}, newFailure(
			FailureContract,
			"prepare",
			"invalid_response",
			0,
		)
	}
	expiresAt, err := time.Parse(time.RFC3339Nano, response.LeaseExpiresAt)
	if err != nil || !expiresAt.After(client.now()) {
		return time.Time{}, newFailure(
			FailureLease,
			"prepare",
			"invalid_lease",
			0,
		)
	}
	return expiresAt, nil
}

func (client *Client) validPrepareMetadata(
	input PrepareInput,
	response prepareWireResponse,
) bool {
	return response.SchemaVersion == TransactionSchemaVersion &&
		response.DecisionID == input.DecisionID &&
		response.BundleVersion == client.config.BundleVersion &&
		response.State == "prepared" &&
		receiptPattern.MatchString(response.Receipt) &&
		response.RouteCallIndex >= 0 &&
		response.RouteCallIndex <= 1_000_000_000 &&
		strings.TrimSpace(response.Policy) != "" &&
		len(response.Policy) <= 256 &&
		slices.Contains(input.Candidates, response.SelectedWorker)
}

func (client *Client) get(
	ctx context.Context,
	operation string,
	path string,
	response any,
) error {
	return client.doJSON(ctx, operation, http.MethodGet, path, nil, response)
}

func (client *Client) post(
	ctx context.Context,
	operation string,
	path string,
	request any,
	response any,
) error {
	payload, err := json.Marshal(request)
	if err != nil || len(payload) == 0 || len(payload) > maxWireBytes {
		return newFailure(
			FailureRequest,
			operation,
			"encode",
			0,
		)
	}
	return client.doJSON(
		ctx,
		operation,
		http.MethodPost,
		path,
		payload,
		response,
	)
}

func (client *Client) doJSON(
	parent context.Context,
	operation string,
	method string,
	path string,
	payload []byte,
	response any,
) error {
	if client == nil {
		return newFailure(
			FailureRequest,
			operation,
			"client_required",
			0,
		)
	}
	ctx, cancel := context.WithTimeout(
		parent,
		client.config.RequestTimeout,
	)
	defer cancel()
	for attempt := 0; ; attempt++ {
		httpResponse, err := client.doAttempt(
			ctx,
			method,
			path,
			payload,
		)
		if err != nil {
			closeResponse(httpResponse)
			if ctx.Err() != nil {
				return newFailure(
					FailureTimeout,
					operation,
					"request_timeout",
					0,
				)
			}
			if attempt < client.config.MaxRetries {
				continue
			}
			return newFailure(
				FailureTransport,
				operation,
				"pre_response",
				0,
			)
		}
		if httpResponse.StatusCode >= http.StatusOK &&
			httpResponse.StatusCode < http.StatusMultipleChoices {
			return consumeJSONResponse(
				httpResponse,
				operation,
				response,
			)
		}
		if retryableStatus(httpResponse.StatusCode) &&
			attempt < client.config.MaxRetries {
			closeResponse(httpResponse)
			continue
		}
		return consumeStatusFailure(httpResponse, operation)
	}
}

func (client *Client) doAttempt(
	ctx context.Context,
	method string,
	path string,
	payload []byte,
) (*http.Response, error) {
	request, err := http.NewRequestWithContext(
		ctx,
		method,
		client.endpoint(path),
		bytes.NewReader(payload),
	)
	if err != nil {
		return nil, err
	}
	request.Header.Set("accept", "application/json")
	request.Header.Set("authorization", "Bearer "+client.config.APIKey)
	if method == http.MethodPost {
		request.Header.Set("content-type", "application/json")
	}
	return client.httpClient.Do(request)
}

func retryableStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusTooManyRequests,
		http.StatusBadGateway,
		http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func closeResponse(response *http.Response) {
	if response == nil || response.Body == nil {
		return
	}
	_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4_096))
	_ = response.Body.Close()
}

func consumeJSONResponse(
	response *http.Response,
	operation string,
	target any,
) error {
	defer func() {
		_ = response.Body.Close()
	}()
	body, err := io.ReadAll(io.LimitReader(
		response.Body,
		maxWireBytes+1,
	))
	if err != nil {
		return newFailure(
			FailureDecode,
			operation,
			"read",
			0,
		)
	}
	if len(body) > maxWireBytes {
		return newFailure(
			FailureDecode,
			operation,
			"response_limit",
			0,
		)
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return newFailure(
			FailureDecode,
			operation,
			"json",
			0,
		)
	}
	if err := requireJSONEOF(decoder); err != nil {
		return newFailure(
			FailureDecode,
			operation,
			"json_trailing",
			0,
		)
	}
	return nil
}

func requireJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	return errors.New("unexpected trailing JSON")
}

func consumeStatusFailure(
	response *http.Response,
	operation string,
) error {
	defer func() {
		_ = response.Body.Close()
	}()
	body, _ := io.ReadAll(io.LimitReader(response.Body, 64_001))
	code := "http_status"
	if len(body) <= 64_000 {
		var envelope struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &envelope) == nil &&
			errorCodePattern.MatchString(envelope.Error.Code) {
			code = envelope.Error.Code
		}
	}
	return newFailure(
		FailureStatus,
		operation,
		code,
		response.StatusCode,
	)
}
