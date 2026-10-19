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

package config

import (
	"fmt"
	"math"
)

// RaylineARCExplorationConfig is a cell's Stage-1 exploration request member,
// selection.exploration (pathfinder#5436). Absent, the request carries none
// and the decision is greedy. Present, it is sent only on a turn whose user is
// eligible: the gateway must state that the request is not in privacy mode
// and that its user is not opted out of training-data collection, and the
// cell must not be an eval cell. Anything unknown is treated as excluded.
type RaylineARCExplorationConfig struct {
	// Tau is the Stage-1 softmax temperature, from the calibrated rate in
	// Basis. 0 asks for the greedy decision.
	Tau *float64 `yaml:"tau"`
	// TauMax is the highest tau this cell may send: the calibration's
	// tau_max, which the service's ARC_EXPLORE_TAU_MAX should equal. A tau
	// above it is refused at load rather than by the service on every turn.
	TauMax *float64 `yaml:"tau_max"`
	// Basis records how Tau was calibrated. Its package_sha256 must be this
	// cell's package_sha256: tau is calibrated per package.
	Basis RaylineARCExplorationBasis `yaml:"basis"`
	// Salt seeds the service's draw together with the boundary's decision
	// key, so the same boundary draws the same model.
	Salt string `yaml:"salt"`
}

// RaylineARCExplorationBasis is selection.exploration.basis.
type RaylineARCExplorationBasis struct {
	// R is the calibrated exploration rate, in [0, 1].
	R *float64 `yaml:"r"`
	// PackageSHA256 is the package the rate was calibrated on.
	PackageSHA256 string `yaml:"package_sha256"`
	// ReadoutPin is the encoder readout the scores were calibrated with.
	ReadoutPin string `yaml:"readout_pin"`
}

// maxRaylineARCExplorationStringLength bounds the salt and readout pin, which
// are logged with every exploration record.
const maxRaylineARCExplorationStringLength = 128

func validateRaylineARCExplorationConfig(cfg *RaylineARCExplorationConfig, packageSHA256 string) error {
	if cfg == nil {
		return nil
	}
	finite := func(value *float64) bool {
		return value != nil && !math.IsNaN(*value) && !math.IsInf(*value, 0) && *value >= 0
	}
	if !finite(cfg.TauMax) {
		return fmt.Errorf("exploration tau_max must be a finite number >= 0")
	}
	if !finite(cfg.Tau) || *cfg.Tau > *cfg.TauMax {
		return fmt.Errorf("exploration tau must be a finite number between 0 and tau_max (%g)", *cfg.TauMax)
	}
	if !finite(cfg.Basis.R) || *cfg.Basis.R > 1 {
		return fmt.Errorf("exploration basis.r must be a finite number between 0 and 1")
	}
	if !raylineARCSHA256Pattern.MatchString(cfg.Basis.PackageSHA256) {
		return fmt.Errorf("exploration basis.package_sha256 must be 64 lowercase hex characters")
	}
	if cfg.Basis.PackageSHA256 != packageSHA256 {
		return fmt.Errorf("exploration basis.package_sha256 must be the cell's package_sha256: tau is calibrated per package")
	}
	if cfg.Basis.ReadoutPin == "" || len(cfg.Basis.ReadoutPin) > maxRaylineARCExplorationStringLength {
		return fmt.Errorf("exploration basis.readout_pin must be 1 to %d bytes", maxRaylineARCExplorationStringLength)
	}
	if cfg.Salt == "" || len(cfg.Salt) > maxRaylineARCExplorationStringLength {
		return fmt.Errorf("exploration salt must be 1 to %d bytes", maxRaylineARCExplorationStringLength)
	}
	return nil
}
