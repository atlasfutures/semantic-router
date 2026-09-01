/*
Copyright 2026 vLLM Semantic Router.

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

package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

func TestParseControllerCommandOptions(t *testing.T) {
	lookup := func(name string) (string, bool) {
		values := map[string]string{
			controllerConfigEnv:         "/config/router.yaml",
			controllerDecisionEnv:       "arc-prod",
			controllerCredentialNameEnv: "ARC_CONTROLLER_PASSWORD",
			controllerRedisUsernameEnv:  "arc-controller",
		}
		value, exists := values[name]
		return value, exists
	}
	options, err := parseControllerCommandOptions(
		[]string{
			"drain",
			"--replica-id", "encoder-a",
			"--timeout", "3s",
		},
		lookup,
	)
	if err != nil {
		t.Fatalf("parse options: %v", err)
	}
	if options.configPath != "/config/router.yaml" ||
		options.decisionName != "arc-prod" ||
		options.replicaID != "encoder-a" ||
		options.passwordEnvironment != "ARC_CONTROLLER_PASSWORD" ||
		options.redisUsername != "arc-controller" ||
		options.operationTimeout != 3*time.Second {
		t.Fatalf("unexpected options: %#v", options)
	}
}

func TestParseControllerCommandOptionsRejectsUnsafeInputs(t *testing.T) {
	tests := [][]string{
		{},
		{"unknown"},
		{"register"},
		{"register", "--replica-id", "encoder-c"},
		{"drain"},
		{"run", "--interval", "0s"},
		{"status", "unexpected"},
	}
	for _, args := range tests {
		if _, err := parseControllerCommandOptions(args, nil); err == nil {
			t.Fatalf("expected %v to fail", args)
		}
	}
}

func TestParseControllerRegisterOptions(t *testing.T) {
	options, err := parseControllerCommandOptions(
		[]string{
			"register",
			"--replica-id", "encoder-c",
			"--base-url", "https://encoder-c.example/v1",
		},
		nil,
	)
	if err != nil {
		t.Fatalf("parse register options: %v", err)
	}
	if options.replicaID != "encoder-c" ||
		options.replicaBaseURL != "https://encoder-c.example/v1" {
		t.Fatalf("unexpected register options: %#v", options)
	}
}

func TestSelectDynamicRaylineARCConfig(t *testing.T) {
	first := dynamicDecision("first")
	second := dynamicDecision("second")
	routerConfig := &config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Decisions: []config.Decision{first, second},
		},
	}
	if _, err := selectDynamicRaylineARCConfig(routerConfig, ""); err == nil {
		t.Fatal("expected ambiguous config to require a decision")
	}
	selected, err := selectDynamicRaylineARCConfig(routerConfig, "second")
	if err != nil {
		t.Fatalf("select decision: %v", err)
	}
	if selected != second.Algorithm.RaylineARC {
		t.Fatal("selected the wrong dynamic ARC config")
	}
}

func TestMembershipStatusOmitsEndpointAndSecretData(t *testing.T) {
	now := time.Now().UTC()
	status := newMembershipStatus(
		"status",
		raylinearc.EncoderMembershipSnapshot{
			Revision: 4,
			Replicas: []raylinearc.EncoderMembershipReplica{
				{
					ID:             "encoder-a",
					BaseURL:        "https://private-encoder.example/v1",
					State:          raylinearc.EncoderReplicaDraining,
					DrainStartedAt: &now,
				},
				{
					ID:      "encoder-b",
					BaseURL: "https://private-survivor.example/v1",
					State:   raylinearc.EncoderReplicaActive,
				},
			},
		},
	)
	payload, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("marshal status: %v", err)
	}
	encoded := string(payload)
	for _, protected := range []string{"private-encoder", "private-survivor", "base_url"} {
		if strings.Contains(encoded, protected) {
			t.Fatalf("status leaked %q: %s", protected, encoded)
		}
	}
	if status.Active != 1 || status.Draining != 1 || len(status.Members) != 2 {
		t.Fatalf("unexpected status: %#v", status)
	}
}

func TestControllerRedisPasswordUsesOnlyNamedEnvironment(t *testing.T) {
	lookup := func(name string) (string, bool) {
		if name != "ARC_CONTROLLER_PASSWORD" {
			t.Fatalf("unexpected environment lookup: %s", name)
		}
		return "private-secret", true
	}
	password, err := controllerRedisPassword("ARC_CONTROLLER_PASSWORD", lookup)
	if err != nil || password != "private-secret" {
		t.Fatalf("password=%q err=%v", password, err)
	}
	if _, err := controllerRedisPassword("MISSING", func(string) (string, bool) {
		return "", false
	}); err == nil {
		t.Fatal("expected missing password environment to fail")
	}
}

func dynamicDecision(name string) config.Decision {
	return config.Decision{
		Name: name,
		Algorithm: &config.AlgorithmConfig{
			Type: config.RaylineARCAlgorithmType,
			RaylineARC: &config.RaylineARCAlgorithmConfig{
				Encoder: config.RaylineARCEncoderConfig{
					Membership: config.RaylineARCEncoderMembershipConfig{
						SchemaVersion: config.RaylineARCEncoderMembershipV1,
					},
				},
			},
		},
	}
}
