package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ValidateModelPaths validates that all discovered paths contain valid model files.
func ValidateModelPaths(paths *ModelPaths) error {
	if paths == nil {
		return fmt.Errorf("model paths is nil")
	}

	if paths.HasLoRAModels() {
		loraChecks := map[string]string{
			"LoRA Intent classifier":   paths.LoRAIntentClassifier,
			"LoRA PII classifier":      paths.LoRAPIIClassifier,
			"LoRA Security classifier": paths.LoRASecurityClassifier,
		}

		for name, path := range loraChecks {
			if path == "" {
				return fmt.Errorf("%s model not found", name)
			}
			if err := validateModelDirectory(path, name); err != nil {
				return err
			}
		}
		return nil
	}

	if paths.HasLegacyModels() {
		legacyChecks := map[string]string{
			"ModernBERT base":     paths.ModernBertBase,
			"Intent classifier":   paths.IntentClassifier,
			"PII classifier":      paths.PIIClassifier,
			"Security classifier": paths.SecurityClassifier,
		}

		for name, path := range legacyChecks {
			if path == "" {
				return fmt.Errorf("%s model not found", name)
			}
			if err := validateModelDirectory(path, name); err != nil {
				return err
			}
		}
		return nil
	}

	return fmt.Errorf("no valid models found (neither LoRA nor legacy)")
}

// validateModelDirectory checks if a directory contains valid model files.
func validateModelDirectory(path, modelName string) error {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return fmt.Errorf("%s model directory does not exist: %s", modelName, path)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s model path is not a directory: %s", modelName, path)
	}

	commonModelFiles := []string{
		"config.json",
		"pytorch_model.bin",
		"model.safetensors",
		"tokenizer.json",
		"vocab.txt",
	}

	hasModelFile := false
	for _, filename := range commonModelFiles {
		if _, err := os.Stat(filepath.Join(path, filename)); err == nil {
			hasModelFile = true
			break
		}
	}

	if !hasModelFile {
		return fmt.Errorf("%s model directory appears to be empty or invalid: %s", modelName, path)
	}

	return checkPickleOnlyWeights(path, modelName)
}

// allowPthWeightsEnv enables loading PyTorch pickle weights (pytorch_model.bin).
// candle-binding enforces the same opt-in; see candle-binding/src/core/weights.rs.
const allowPthWeightsEnv = "SR_ALLOW_PTH_WEIGHTS"

func pthWeightsAllowed() bool {
	v := strings.TrimSpace(os.Getenv(allowPthWeightsEnv))
	return v == "1" || strings.EqualFold(v, "true")
}

// checkPickleOnlyWeights fails a model directory whose only weights are
// pytorch_model.bin unless SR_ALLOW_PTH_WEIGHTS is set. The candle loaders
// would refuse it at load time; this reports it at startup with the fix.
func checkPickleOnlyWeights(path, modelName string) error {
	if _, err := os.Stat(filepath.Join(path, "model.safetensors")); err == nil {
		return nil
	}
	if _, err := os.Stat(filepath.Join(path, "pytorch_model.bin")); err != nil {
		return nil
	}
	if pthWeightsAllowed() {
		return nil
	}
	return fmt.Errorf("%s model at %s has only PyTorch pickle weights (pytorch_model.bin), "+
		"which are disabled by default: convert them to model.safetensors or set %s=1",
		modelName, path, allowPthWeightsEnv)
}
