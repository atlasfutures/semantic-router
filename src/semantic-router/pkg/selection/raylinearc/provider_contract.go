/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package raylinearc

import (
	"fmt"
	"net/url"
	"strings"
)

func validateWorkerProviderContract(worker *WorkerManifest) error {
	if strings.TrimSpace(worker.APIKeyEnv) == "" {
		return fmt.Errorf("worker %q api_key_env is required", worker.ID)
	}
	switch worker.EffectiveDispatchBackend() {
	case DispatchOpenRouter:
		return validateWorkerOpenRouterContract(worker)
	case DispatchOpenAICompat:
		return validateWorkerOpenAICompatibleContract(worker)
	default:
		return fmt.Errorf(
			"worker %q dispatch backend %q is unsupported",
			worker.ID,
			worker.DispatchBackend,
		)
	}
}

func validateWorkerOpenRouterContract(worker *WorkerManifest) error {
	required := map[string]string{
		"provider slug":           worker.OpenRouterProviderSlug,
		"provider name":           worker.OpenRouterProviderName,
		"provider pricing source": worker.OpenRouterPricingSource,
	}
	for label, value := range required {
		if strings.TrimSpace(value) == "" {
			return fmt.Errorf("worker %q %s is required", worker.ID, label)
		}
	}
	if len(worker.OpenRouterProviderOrder) == 0 ||
		worker.OpenRouterProviderOrder[0] != worker.OpenRouterProviderSlug {
		return fmt.Errorf(
			"worker %q provider order must start with its provider slug",
			worker.ID,
		)
	}
	seenProviders := make(map[string]struct{}, len(worker.OpenRouterProviderOrder))
	for _, provider := range worker.OpenRouterProviderOrder {
		normalizedProvider := strings.TrimSpace(provider)
		if normalizedProvider == "" || normalizedProvider != provider {
			return fmt.Errorf(
				"worker %q provider order contains an invalid slug %q",
				worker.ID,
				provider,
			)
		}
		if _, exists := seenProviders[provider]; exists {
			return fmt.Errorf(
				"worker %q provider order contains duplicate slug %q",
				worker.ID,
				provider,
			)
		}
		seenProviders[provider] = struct{}{}
	}
	if worker.OpenRouterAllowFallbacks {
		return fmt.Errorf("worker %q must disable provider fallbacks", worker.ID)
	}
	if !worker.OpenRouterRequireParameters {
		return fmt.Errorf("worker %q must require provider parameters", worker.ID)
	}
	return nil
}

func validateWorkerOpenAICompatibleContract(worker *WorkerManifest) error {
	if !validProviderBaseURL(worker.ProviderBaseURL) {
		return fmt.Errorf(
			"worker %q openai-compatible provider_base_url is invalid",
			worker.ID,
		)
	}
	if hasOpenRouterProviderFields(worker) {
		return fmt.Errorf(
			"worker %q openai-compatible dispatch cannot declare OpenRouter provider fields",
			worker.ID,
		)
	}
	return nil
}

func validProviderBaseURL(value string) bool {
	parsed, err := url.Parse(value)
	return err == nil && parsed.Host != "" &&
		(parsed.Scheme == "http" || parsed.Scheme == "https") &&
		parsed.User == nil && parsed.RawQuery == "" && parsed.Fragment == ""
}

func hasOpenRouterProviderFields(worker *WorkerManifest) bool {
	return worker.OpenRouterProviderSlug != "" ||
		worker.OpenRouterProviderName != "" ||
		len(worker.OpenRouterProviderOrder) != 0 ||
		worker.OpenRouterAllowFallbacks ||
		worker.OpenRouterRequireParameters ||
		worker.OpenRouterPricingSource != ""
}
