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

// RaylineARCExplorationTemperatureConfig is a cell's hosted Stage-1
// exploration in the temperature shape (pathfinder#5637): T is a true softmax
// temperature over per-decision normalized scores, so one T means about the
// same randomness at any V and on any package, and no per-package basis is
// sent. The service enforces the hosted cap (ARC_EXPLORE_T_MAX_HOSTED,
// clamping a caller's T and refusing a default above it) and the new-model
// floor; VSR decides who may explore and states it.
//
// Fail closed: a turn is sent the member only on a cell that is not an eval
// cell and trusts the gateway's turn-signal headers, on a main turn whose
// gateway states x-rayline-privacy-mode: false and x-rayline-training-data:
// allowed. Any other turn -- privacy mode, ZDR, opted out, or any of them
// unstated -- sends none, so it is greedy and never floor-sampled.
type RaylineARCExplorationTemperatureConfig struct {
	// Temperature is the cell's default T, the calibrated internal default
	// (r about 0.05), sent when the gateway states no caller T. 0 sends the
	// member at T 0, under which only the service's floor explores.
	Temperature *float64 `yaml:"temperature"`
	// TemperatureMax is the cell's hosted cap, the calibration's
	// temperature_max, which the service's ARC_EXPLORE_T_MAX_HOSTED should
	// equal. A default above it is refused at load; a caller's T above it is
	// sent as stated and clamped by the service, which logs both.
	TemperatureMax *float64 `yaml:"temperature_max"`
	// CallerTemperature honours the end caller's own T on an eligible
	// hosted turn: the body's routing.temperature, on a cell whose
	// policy_service reads generation routing (pathfinder#5664). Off, the
	// default is sent. No request header carries a caller T.
	CallerTemperature bool `yaml:"caller_temperature,omitempty"`
	// Salt seeds the service's draw with the boundary's decision key.
	Salt string `yaml:"salt"`
	// Calibration is the default T's provenance (r, package, readout pin),
	// logged by the service and never checked. Optional.
	Calibration *RaylineARCExplorationBasis `yaml:"calibration,omitempty"`
}

func validateRaylineARCExplorationTemperatureConfig(cfg *RaylineARCExplorationTemperatureConfig) error {
	if cfg == nil {
		return nil
	}
	finite := func(value *float64) bool {
		return value != nil && !math.IsNaN(*value) && !math.IsInf(*value, 0) && *value >= 0
	}
	if !finite(cfg.TemperatureMax) {
		return fmt.Errorf("exploration_temperature temperature_max must be a finite number >= 0")
	}
	if !finite(cfg.Temperature) || *cfg.Temperature > *cfg.TemperatureMax {
		return fmt.Errorf("exploration_temperature temperature must be a finite number between 0 and temperature_max (%g)", *cfg.TemperatureMax)
	}
	if cfg.Salt == "" || len(cfg.Salt) > maxRaylineARCExplorationStringLength {
		return fmt.Errorf("exploration_temperature salt must be 1 to %d bytes", maxRaylineARCExplorationStringLength)
	}
	if basis := cfg.Calibration; basis != nil {
		if !finite(basis.R) || *basis.R > 1 {
			return fmt.Errorf("exploration_temperature calibration.r must be a finite number between 0 and 1")
		}
		if !raylineARCSHA256Pattern.MatchString(basis.PackageSHA256) {
			return fmt.Errorf("exploration_temperature calibration.package_sha256 must be 64 lowercase hex characters")
		}
		if basis.ReadoutPin == "" || len(basis.ReadoutPin) > maxRaylineARCExplorationStringLength {
			return fmt.Errorf("exploration_temperature calibration.readout_pin must be 1 to %d bytes", maxRaylineARCExplorationStringLength)
		}
	}
	return nil
}
