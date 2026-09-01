#!/usr/bin/env python3
"""The router config the compose-free acceptance stack renders.

One decision, two synthetic workers, and the ARC algorithm block. Everything
the hermetic run does not need -- embeddings, classifiers, caches -- is
switched off explicitly rather than left to a default, so a config change
upstream cannot quietly pull a model download into this run.
"""

from __future__ import annotations

CONFIG_TEMPLATE = """version: v0.3

listeners:
  - name: rayline-arc-acceptance
    address: 127.0.0.1
    port: {listener_port}
    timeout: 30s

providers:
  defaults:
    default_model: worker-a
  models:
    - name: worker-a
      provider_model_id: synthetic/provider-a
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 1
        cached_input_per_1m: 0.5
        cache_write_per_1m: 1.5
        completion_per_1m: 2
      backend_refs:
        - name: synthetic-a
          base_url: http://127.0.0.1:{provider_port}
          provider: openai
          type: openai
          api_key_env: SYNTHETIC_API_KEY
    - name: worker-b
      provider_model_id: synthetic/provider-b
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 2
        cached_input_per_1m: 1
        cache_write_per_1m: 2.5
        completion_per_1m: 4
      backend_refs:
        - name: synthetic-b
          base_url: http://127.0.0.1:{provider_port}
          provider: openai
          type: openai
          api_key_env: SYNTHETIC_API_KEY

routing:
  modelCards:
    - name: worker-a
      modality: text
    - name: worker-b
      modality: text
  decisions:
    - name: rayline-arc-acceptance
      description: Compose-free synthetic ARC acceptance route
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: worker-a
          use_reasoning: false
        - model: worker-b
          use_reasoning: true
      adaptations:
        mode: bypass
      plugins:
        # ARC episode requests must never be persisted; the validator rejects
        # a rayline_arc decision whose effective replay config is enabled.
        - type: router_replay
          configuration:
            enabled: false
      algorithm:
        type: rayline_arc
        on_error: fail_closed
        rayline_arc:
          artifact_dir: {artifact_dir}
          artifact_revision: {artifact_revision}
          encoder:
            base_url: http://127.0.0.1:{encoder_port}
            replicas: []
            membership: {{}}
            failover: {{}}
            model: Qwen/Qwen3.5-0.8B
            model_revision: 2fc06364715b967f1860aea9cf38778875588b17
            expected_build_id: vllm@public-rayline-e2e-build
            expected_io_plugin_version: rayline-arc-io@0.1.0
            serializer_version: mtrouter-token-blocks-v2
            serving_rung: B
            required_pooling_capabilities:
              - chunked_causal_mean
            modal_key_env: RAYLINE_ARC_MODAL_KEY
            modal_secret_env: RAYLINE_ARC_MODAL_SECRET
            connect_timeout_seconds: 5
            total_timeout_seconds: 10
            max_retries: 0
            max_inflight_encoder_calls: {max_inflight}
          episode:
            id_header: x-rayline-session
            close_header: ""
            backend: {episode_backend}
            key_prefix: "vsr:rayline-arc-acceptance:"
            acquire_timeout_seconds: {acquire_timeout}
            lease_ttl_seconds: 3
            idle_ttl_seconds: 900
            max_in_memory_episodes: 128
            development_mode: {development_mode}
            redis:
              address: {redis_address}
              db: 0
              password_env: RAYLINE_ARC_REDIS_PASSWORD
              use_tls: false
              pool_size: 8

global:
  stores:
    semantic_cache:
      enabled: false
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: ""
        qwen3_model_path: ""
        gemma_model_path: ""
        bert_model_path: ""
        multimodal_model_path: ""
    modules:
      prompt_guard:
        enabled: false
        model_ref: ""
        model_id: ""
        jailbreak_mapping_path: ""
        use_mmbert_32k: false
      classifier:
        domain:
          model_ref: ""
          model_id: ""
          category_mapping_path: ""
          use_mmbert_32k: false
        pii:
          model_ref: ""
          model_id: ""
          pii_mapping_path: ""
          use_mmbert_32k: false
      feedback_detector:
        enabled: false
        model_ref: ""
        model_id: ""
        use_mmbert_32k: false
"""
