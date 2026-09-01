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
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type membershipController interface {
	RegisterActive(context.Context, string, string) (raylinearc.EncoderMembershipSnapshot, error)
	BeginDrain(context.Context, string) (raylinearc.EncoderMembershipSnapshot, error)
	Reconcile(context.Context) (raylinearc.EncoderMembershipDrainReport, error)
}

type controllerRuntime struct {
	source     raylinearc.EncoderMembershipReader
	controller membershipController
	close      func() error
}

func openControllerRuntime(
	options controllerCommandOptions,
	lookup environmentLookup,
) (*controllerRuntime, error) {
	routerConfig, err := config.Parse(options.configPath)
	if err != nil {
		return nil, errors.New("controller config is invalid")
	}
	arcConfig, err := selectDynamicRaylineARCConfig(
		routerConfig,
		options.decisionName,
	)
	if err != nil {
		return nil, err
	}
	redisConfig := arcConfig.Episode.Redis
	passwordEnvironment := redisConfig.PasswordEnv
	if options.passwordEnvironment != "" {
		passwordEnvironment = options.passwordEnvironment
	}
	password, err := controllerRedisPassword(passwordEnvironment, lookup)
	if err != nil {
		return nil, err
	}
	store, err := raylinearc.NewRedisEpisodeStore(
		raylinearc.RedisEpisodeStoreConfig{
			Address:   redisConfig.Address,
			DB:        redisConfig.DB,
			Username:  options.redisUsername,
			Password:  password,
			UseTLS:    redisConfig.UseTLS,
			PoolSize:  redisConfig.PoolSize,
			KeyPrefix: arcConfig.Episode.KeyPrefix,
			LeaseTTL: time.Duration(
				arcConfig.Episode.LeaseTTLSeconds,
			) * time.Second,
			IdleTTL: time.Duration(
				arcConfig.Episode.IdleTTLSeconds,
			) * time.Second,
		},
	)
	if err != nil {
		return nil, errors.New("controller Redis configuration is invalid")
	}
	source, err := raylinearc.NewRedisEncoderMembershipSource(store)
	if err != nil {
		_ = store.Close()
		return nil, errors.New("controller membership source is unavailable")
	}
	controller, err := raylinearc.NewEncoderMembershipController(
		raylinearc.EncoderMembershipControllerConfig{
			Source:     source,
			References: store,
			IdleTTL: time.Duration(
				arcConfig.Episode.IdleTTLSeconds,
			) * time.Second,
		},
	)
	if err != nil {
		_ = source.Close()
		return nil, errors.New("controller configuration is incomplete")
	}
	return &controllerRuntime{
		source:     source,
		controller: controller,
		close:      source.Close,
	}, nil
}

func selectDynamicRaylineARCConfig(
	routerConfig *config.RouterConfig,
	decisionName string,
) (*config.RaylineARCAlgorithmConfig, error) {
	if routerConfig == nil {
		return nil, errors.New("controller config is unavailable")
	}
	matches := make([]*config.RaylineARCAlgorithmConfig, 0, 1)
	for index := range routerConfig.Decisions {
		decision := &routerConfig.Decisions[index]
		if decisionName != "" && decision.Name != decisionName {
			continue
		}
		if decision.Algorithm == nil ||
			decision.Algorithm.Type != config.RaylineARCAlgorithmType ||
			decision.Algorithm.RaylineARC == nil ||
			decision.Algorithm.RaylineARC.Encoder.Membership.SchemaVersion == "" {
			continue
		}
		matches = append(matches, decision.Algorithm.RaylineARC)
	}
	if len(matches) == 0 {
		if decisionName != "" {
			return nil, fmt.Errorf(
				"decision %q is not a dynamic Rayline ARC decision",
				decisionName,
			)
		}
		return nil, errors.New("config has no dynamic Rayline ARC decision")
	}
	if len(matches) > 1 {
		return nil, errors.New("multiple dynamic Rayline ARC decisions require --decision")
	}
	return matches[0], nil
}

func controllerRedisPassword(
	passwordEnvironment string,
	lookup environmentLookup,
) (string, error) {
	passwordEnvironment = strings.TrimSpace(passwordEnvironment)
	if passwordEnvironment == "" {
		return "", nil
	}
	if lookup == nil {
		return "", errors.New("controller Redis password environment is unavailable")
	}
	password, exists := lookup(passwordEnvironment)
	if !exists {
		return "", errors.New("controller Redis password environment is unset")
	}
	return password, nil
}
