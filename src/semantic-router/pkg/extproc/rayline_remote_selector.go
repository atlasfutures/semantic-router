package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylineremote"
)

type raylineRemoteRuntime struct {
	client        *raylineremote.Client
	hmacKey       []byte
	workerIDs     []string
	modelByWorker map[string]string
	bundleHash    string
	failureClass  string
}

type raylineRemoteSelector struct {
	runtimes map[string]*raylineRemoteRuntime
}

type raylineRemoteSelectionFailure struct {
	class string
}

func (failure *raylineRemoteSelectionFailure) Error() string {
	return "Rayline remote selection failed (class=" + failure.class + ")"
}

func createRaylineRemoteSelector(
	cfg *config.RouterConfig,
) (selection.Selector, string) {
	decisions := configuredRaylineRemoteDecisions(cfg)
	if len(decisions) == 0 {
		return nil, ""
	}
	selector := &raylineRemoteSelector{
		runtimes: make(map[string]*raylineRemoteRuntime, len(decisions)),
	}
	overallFailure := ""
	for _, decision := range decisions {
		runtime, failureClass := createRaylineRemoteRuntime(
			cfg,
			decision,
		)
		if failureClass != "" {
			runtime = &raylineRemoteRuntime{
				failureClass: failureClass,
			}
			if overallFailure == "" {
				overallFailure = failureClass
			}
		}
		selector.runtimes[decision.Name] = runtime
	}
	return selector, overallFailure
}

func createRaylineRemoteRuntime(
	cfg *config.RouterConfig,
	decision *config.Decision,
) (*raylineRemoteRuntime, string) {
	if cfg == nil || decision == nil ||
		decision.Algorithm == nil ||
		decision.Algorithm.RaylineRemote == nil {
		return nil, "configuration"
	}
	remoteConfig := decision.Algorithm.RaylineRemote
	apiKey, ok := nonemptyEnvironmentSecret(remoteConfig.APIKeyEnv)
	if !ok {
		return nil, "api_auth"
	}
	hmacSecret, ok := nonemptyEnvironmentSecret(
		remoteConfig.EpisodeHMACKeyEnv,
	)
	if !ok || len(hmacSecret) < 32 {
		return nil, "episode_auth"
	}
	workers, modelByWorker, failureClass := raylineRemoteWorkerContracts(cfg, decision)
	if failureClass != "" {
		return nil, failureClass
	}
	client, err := raylineremote.NewClient(
		raylineremote.ClientConfig{
			BaseURL:       remoteConfig.BaseURL,
			BundleVersion: remoteConfig.BundleVersion,
			APIKey:        apiKey,
			ConnectTimeout: time.Duration(
				remoteConfig.ConnectTimeoutMS,
			) * time.Millisecond,
			RequestTimeout: time.Duration(
				remoteConfig.RequestTimeoutMS,
			) * time.Millisecond,
			LeaseTTL: time.Duration(
				remoteConfig.LeaseTTLSeconds,
			) * time.Second,
			MaxRetries: remoteConfig.MaxRetries,
			Workers:    workers,
		},
	)
	if err != nil {
		return nil, "client_config"
	}
	if err := client.CheckReadiness(context.Background()); err != nil {
		return nil, remoteFailureClass(err, "readiness")
	}
	workerIDs := make([]string, 0, len(workers))
	for _, worker := range workers {
		workerIDs = append(workerIDs, worker.ID)
	}
	return &raylineRemoteRuntime{
		client:        client,
		hmacKey:       []byte(hmacSecret),
		workerIDs:     workerIDs,
		modelByWorker: modelByWorker,
		bundleHash:    boundedIdentityHash(remoteConfig.BundleVersion),
	}, ""
}

func configuredRaylineRemoteDecisions(
	cfg *config.RouterConfig,
) []*config.Decision {
	if cfg == nil {
		return nil
	}
	all := cfg.AllRoutingDecisions()
	decisions := make([]*config.Decision, 0)
	for index := range all {
		decision := &all[index]
		if decision.Algorithm != nil &&
			decision.Algorithm.Type ==
				config.RaylineRemoteAlgorithmType {
			decisions = append(decisions, decision)
		}
	}
	return decisions
}

func nonemptyEnvironmentSecret(name string) (string, bool) {
	value, ok := os.LookupEnv(name)
	return value, ok && strings.TrimSpace(value) != ""
}

func raylineRemoteWorkerContracts(
	cfg *config.RouterConfig,
	decision *config.Decision,
) (
	[]raylineremote.WorkerContract,
	map[string]string,
	string,
) {
	remoteConfig := decision.Algorithm.RaylineRemote
	modelRefs := make(map[string]config.ModelRef, len(decision.ModelRefs))
	for _, modelRef := range decision.ModelRefs {
		modelRefs[modelRef.Model] = modelRef
	}
	workers := make(
		[]raylineremote.WorkerContract,
		0,
		len(remoteConfig.Workers),
	)
	modelByWorker := make(
		map[string]string,
		len(remoteConfig.Workers),
	)
	for _, configured := range remoteConfig.Workers {
		modelRef, ok := modelRefs[configured.Model]
		if !ok ||
			cfg.GetModelAPIFormat(configured.Model) !=
				config.APIFormatOpenAI {
			return nil, nil, "dispatch_model"
		}
		providerModel, backend, ok := raylineRemoteProviderIdentity(cfg, configured.Model)
		if !ok {
			return nil, nil, "dispatch_endpoint"
		}
		prices, ok := raylineRemotePrices(cfg, configured.Model)
		if !ok {
			return nil, nil, "dispatch_pricing"
		}
		thinkingMode := "disabled"
		if modelRef.UseReasoning != nil && *modelRef.UseReasoning {
			thinkingMode = strings.TrimSpace(
				modelRef.ReasoningEffort,
			)
			if thinkingMode == "" {
				thinkingMode = "enabled"
			}
		}
		workers = append(workers, raylineremote.WorkerContract{
			ID:             configured.ID,
			Model:          providerModel,
			Backend:        backend,
			ThinkingMode:   thinkingMode,
			AllowFallbacks: false,
			Prices:         &prices,
		})
		modelByWorker[configured.ID] = configured.Model
	}
	return workers, modelByWorker, ""
}

func raylineRemoteProviderIdentity(
	cfg *config.RouterConfig,
	model string,
) (string, string, bool) {
	endpoints := cfg.GetEndpointsForModel(model)
	if len(endpoints) == 0 {
		return "", "", false
	}
	providerModel := ""
	backend := ""
	for _, endpoint := range endpoints {
		candidateModel := cfg.ResolveExternalModelID(
			model,
			endpoint.Name,
		)
		candidateBackend := endpoint.Type
		if candidateBackend == "" {
			candidateBackend = "vllm"
		}
		if providerModel == "" {
			providerModel = candidateModel
			backend = candidateBackend
			continue
		}
		if candidateModel != providerModel ||
			candidateBackend != backend {
			return "", "", false
		}
	}
	return providerModel, backend, providerModel != "" && backend != ""
}

func raylineRemotePrices(
	cfg *config.RouterConfig,
	model string,
) (raylineremote.WorkerPrices, bool) {
	pricing, ok := cfg.GetFullModelPricing(model)
	if !ok || (pricing.Currency != "" && pricing.Currency != "USD") {
		return raylineremote.WorkerPrices{}, false
	}
	cacheWrite := pricing.PromptPer1M
	if pricing.CacheWritePer1M != nil {
		cacheWrite = *pricing.CacheWritePer1M
	}
	const scale = 1_000_000
	return raylineremote.WorkerPrices{
		Input:      pricing.PromptPer1M / scale,
		CacheRead:  pricing.CachedInputPer1M / scale,
		CacheWrite: cacheWrite / scale,
		Output:     pricing.CompletionPer1M / scale,
	}, true
}

func (selector *raylineRemoteSelector) Select(
	ctx context.Context,
	selCtx *selection.SelectionContext,
) (*selection.SelectionResult, error) {
	runtime, remoteContext, err := selector.prepareSelection(selCtx)
	if err != nil {
		return nil, err
	}
	episodeKey, err := raylineremote.DeriveEpisodeKey(
		remoteContext.RawEpisodeID,
		runtime.hmacKey,
	)
	if err != nil {
		return nil, remoteSelectionFailure(
			remoteFailureClass(err, "episode"),
		)
	}
	chatRequest, err := raylineremote.ChatRequestFromBody(
		remoteContext.RequestBody,
	)
	if err != nil {
		return nil, remoteSelectionFailure(
			remoteFailureClass(err, "request"),
		)
	}
	started := time.Now()
	transaction, prepared, err := runtime.client.Prepare(
		ctx,
		raylineremote.PrepareInput{
			DecisionID: remoteContext.DecisionID,
			EpisodeKey: episodeKey,
			Candidates: append([]string(nil), runtime.workerIDs...),
			Request:    chatRequest,
		},
	)
	latency := time.Since(started)
	if err != nil {
		return nil, remoteSelectionFailure(
			remoteFailureClass(err, "prepare"),
		)
	}
	model, ok := runtime.modelByWorker[prepared.SelectedWorker]
	if !ok {
		abortRemoteTransaction(transaction)
		return nil, remoteSelectionFailure("worker_mapping")
	}
	selectedIndex := candidateModelIndex(selCtx.CandidateModels, model)
	if selectedIndex < 0 {
		abortRemoteTransaction(transaction)
		return nil, remoteSelectionFailure("candidate_mapping")
	}
	if remoteContext.Transaction != nil {
		abortRemoteTransaction(transaction)
		return nil, remoteSelectionFailure("duplicate_transaction")
	}
	if err := transaction.StartLeaseKeeper(
		context.Background(),
	); err != nil {
		abortRemoteTransaction(transaction)
		return nil, remoteSelectionFailure(
			remoteFailureClass(err, "lease"),
		)
	}
	remoteContext.Transaction = transaction
	selected := selCtx.CandidateModels[selectedIndex]
	return &selection.SelectionResult{
		SelectedModel: selected.Model,
		LoRAName:      selected.LoRAName,
		Score:         1,
		Confidence:    1,
		Method:        selection.MethodRaylineRemote,
		Tier:          selection.TierExperimental,
		Reasoning:     "Pathfinder-owned Rayline policy",
		AllScores:     nil,
		RaylineRemote: &selection.RaylineRemoteTrace{
			BundleHash:       runtime.bundleHash,
			SelectedIndex:    selectedIndex,
			RouteCallIndex:   prepared.RouteCallIndex,
			SelectionLatency: latency,
		},
	}, nil
}

func (selector *raylineRemoteSelector) prepareSelection(
	selCtx *selection.SelectionContext,
) (
	*raylineRemoteRuntime,
	*selection.RaylineRemoteSelectionContext,
	error,
) {
	if selector == nil || selCtx == nil ||
		selCtx.RaylineRemote == nil {
		return nil, nil, remoteSelectionFailure("missing_context")
	}
	remoteContext := selCtx.RaylineRemote
	if remoteContext.PreparationFailure != "" {
		return nil, nil, remoteSelectionFailure(
			remoteContext.PreparationFailure,
		)
	}
	runtime := selector.runtimes[selCtx.DecisionName]
	if runtime == nil {
		return nil, nil, remoteSelectionFailure("unknown_decision")
	}
	if runtime.failureClass != "" || runtime.client == nil {
		class := runtime.failureClass
		if class == "" {
			class = "not_ready"
		}
		return nil, nil, remoteSelectionFailure(class)
	}
	if len(selCtx.CandidateModels) != len(runtime.modelByWorker) {
		return nil, nil, remoteSelectionFailure("candidate_count")
	}
	for _, model := range runtime.modelByWorker {
		if candidateModelIndex(selCtx.CandidateModels, model) < 0 {
			return nil, nil, remoteSelectionFailure("candidate_mapping")
		}
	}
	return runtime, remoteContext, nil
}

func candidateModelIndex(
	candidates []config.ModelRef,
	model string,
) int {
	for index := range candidates {
		if candidates[index].Model == model {
			return index
		}
	}
	return -1
}

func abortRemoteTransaction(
	transaction *raylineremote.Transaction,
) {
	if transaction == nil {
		return
	}
	ctx, cancel := context.WithTimeout(
		context.Background(),
		5*time.Second,
	)
	defer cancel()
	_ = transaction.Abort(ctx, raylineremote.AbortSelectionFailed)
}

func remoteFailureClass(err error, fallback string) string {
	var failure *raylineremote.Failure
	if errors.As(err, &failure) {
		return string(failure.Class)
	}
	return fallback
}

func remoteSelectionFailure(
	class string,
) *raylineRemoteSelectionFailure {
	if class == "" {
		class = "selection"
	}
	return &raylineRemoteSelectionFailure{class: class}
}

func boundedIdentityHash(value string) string {
	digest := sha256.Sum256([]byte(value))
	return hex.EncodeToString(digest[:8])
}

func (selector *raylineRemoteSelector) Method() selection.SelectionMethod {
	return selection.MethodRaylineRemote
}

func (selector *raylineRemoteSelector) UpdateFeedback(
	context.Context,
	*selection.Feedback,
) error {
	return errors.New(
		"rayline remote does not accept online selector feedback",
	)
}

func (selector *raylineRemoteSelector) Tier() selection.AlgorithmTier {
	return selection.TierExperimental
}

func (selector *raylineRemoteSelector) ExternalDependencies() []selection.Dependency {
	return []selection.Dependency{
		{
			Name:        "Pathfinder Rayline router",
			Type:        selection.DependencyExternalService,
			Description: "Version-pinned transactional selection service",
			Required:    true,
		},
	}
}

func bindRaylineRemoteDispatchContract(
	selCtx *selection.SelectionContext,
	algorithm *config.AlgorithmConfig,
	result *selection.SelectionResult,
	selectedModelRef *config.ModelRef,
	ctx *RequestContext,
) error {
	if !raylineRemoteSelection(algorithm) {
		return nil
	}
	var transaction *raylineremote.Transaction
	if selCtx != nil && selCtx.RaylineRemote != nil {
		transaction = selCtx.RaylineRemote.Transaction
	}
	abortOnFailure := func() error {
		abortRemoteTransaction(transaction)
		return selectionFailureForAlgorithm(
			algorithm,
			"dispatch_contract",
		)
	}
	if result == nil ||
		result.RaylineRemote == nil ||
		selectedModelRef == nil ||
		transaction == nil ||
		ctx == nil {
		return abortOnFailure()
	}
	selectedIndex := candidateModelIndex(
		selCtx.CandidateModels,
		selectedModelRef.Model,
	)
	if selectedIndex < 0 ||
		result.RaylineRemote.SelectedIndex != selectedIndex {
		return abortOnFailure()
	}
	if err := bindRaylineRemoteSelectionTransaction(
		ctx,
		transaction,
	); err != nil {
		return abortOnFailure()
	}
	return nil
}

func abortPreparedRaylineRemote(
	selCtx *selection.SelectionContext,
) {
	if selCtx == nil || selCtx.RaylineRemote == nil {
		return
	}
	abortRemoteTransaction(selCtx.RaylineRemote.Transaction)
}

func observeRaylineRemoteSelection(
	ctx *RequestContext,
	trace *selection.RaylineRemoteTrace,
) {
	if ctx == nil || trace == nil {
		return
	}
	metrics.RecordRaylineRemoteSelection(
		trace.SelectedIndex,
		trace.SelectionLatency,
	)
	logging.ComponentEvent(
		"extproc",
		"rayline_remote_selection",
		map[string]interface{}{
			"request_id":       ctx.RequestID,
			"bundle_hash":      trace.BundleHash,
			"candidate_index":  trace.SelectedIndex,
			"route_call_index": trace.RouteCallIndex,
			"latency_millis": trace.SelectionLatency.
				Milliseconds(),
		},
	)
}
