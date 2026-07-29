//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (s *ClassificationAPIServer) handleModelsInfo(w http.ResponseWriter, _ *http.Request) {
	response := s.buildModelsInfoResponse()
	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleEmbeddingModelsInfo handles GET /api/v1/embeddings/models
// Returns ONLY embedding models information
func (s *ClassificationAPIServer) handleEmbeddingModelsInfo(w http.ResponseWriter, r *http.Request) {
	embeddingModels := s.getEmbeddingModelsInfo(s.loadModelsRuntimeState())

	response := map[string]interface{}{
		"models": embeddingModels,
		"count":  len(embeddingModels),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleClassifierInfo returns live classifier/runtime config.
// Access requires config.read; plaintext secrets require secret_view (otherwise redacted).
func (s *ClassificationAPIServer) handleClassifierInfo(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
			"status": "no_config",
			"config": nil,
		})
		return
	}

	s.writeJSONResponse(w, http.StatusOK, map[string]interface{}{
		"status": "config_loaded",
		"config": s.maybeRedactConfigView(r, classifierConfigView(cfg)),
	})
}

// classifierConfigView renders the router config the way /info/classifier
// publishes it. RouterConfig no longer stores decisions inline — config.Recipes
// owns them — but the top-level `Decisions` key is a published wire contract
// (the runtime-config end-to-end testcase decodes it), so the view keeps
// emitting the default recipe's decisions under that key.
func classifierConfigView(cfg *config.RouterConfig) interface{} {
	normalized := jsonCompatibleValue(cfg)
	view, ok := normalized.(map[string]interface{})
	if !ok {
		return normalized
	}
	view["Decisions"] = jsonCompatibleValue(cfg.DefaultDecisions())
	return view
}

type classifierModelAvailability struct {
	core                   bool
	factCheck              bool
	hallucination          bool
	hallucinationExplainer bool
	feedback               bool
}

// buildModelsInfoResponse builds the models info response
func (s *ClassificationAPIServer) buildModelsInfoResponse() ModelsInfoResponse {
	runtimeState := s.loadModelsRuntimeState()
	models := s.getClassifierModelsInfo(s.classifierModelAvailability(), runtimeState)

	// Add embedding models information
	embeddingModels := s.getEmbeddingModelsInfo(runtimeState)
	models = append(models, embeddingModels...)

	// Get system information
	systemInfo := s.getSystemInfo()

	return ModelsInfoResponse{
		Models:  models,
		Summary: buildModelsInfoSummary(runtimeState, models),
		System:  systemInfo,
	}
}
