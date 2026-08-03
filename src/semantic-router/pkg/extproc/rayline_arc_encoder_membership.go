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

package extproc

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/raylinearc"
)

type raylineARCReadyEncoder interface {
	raylineARCEncoder
	Probe(context.Context, string) error
	Close()
}

func createRaylineARCEncoder(
	arcConfig *config.RaylineARCAlgorithmConfig,
) (raylineARCReadyEncoder, string) {
	clientConfig, err := raylineARCEncoderClientConfig(arcConfig)
	if err != nil {
		return nil, "encoder_auth"
	}
	if len(arcConfig.Encoder.Replicas) == 0 {
		encoder, clientErr := raylinearc.NewEncoderClient(clientConfig)
		if clientErr != nil {
			return nil, "encoder_config"
		}
		return encoder, ""
	}
	replicas := make(
		[]raylinearc.EncoderReplica,
		0,
		len(arcConfig.Encoder.Replicas),
	)
	closeClients := func() {
		for index := range replicas {
			replicas[index].Client.Close()
		}
	}
	for _, configured := range arcConfig.Encoder.Replicas {
		replicaConfig := clientConfig
		replicaConfig.BaseURL = configured.BaseURL
		client, clientErr := raylinearc.NewEncoderClient(replicaConfig)
		if clientErr != nil {
			closeClients()
			return nil, "encoder_config"
		}
		replicas = append(replicas, raylinearc.EncoderReplica{
			ID:     configured.ID,
			State:  raylinearc.EncoderReplicaState(configured.State),
			Client: client,
		})
	}
	pool, err := raylinearc.NewEncoderPool(
		replicas,
		raylinearc.EncoderPoolConfig{
			SchemaVersion: arcConfig.Encoder.Failover.SchemaVersion,
			UnavailableStatusCodes: append(
				[]int(nil),
				arcConfig.Encoder.Failover.UnavailableStatusCodes...,
			),
			UnavailableCooldown: time.Duration(
				arcConfig.Encoder.Failover.UnavailableCooldownSeconds,
			) * time.Second,
			MaxRemaps: arcConfig.Encoder.Failover.MaxRemaps,
		},
	)
	if err != nil {
		closeClients()
		return nil, "encoder_config"
	}
	return pool, ""
}
