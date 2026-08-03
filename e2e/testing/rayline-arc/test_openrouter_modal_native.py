# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import yaml

import openrouter_modal_native_benchmark as benchmark
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_modal_native_fixture import (
    CHECKPOINT_REMOTE_PATH,
    DECISION_LOG_REMOTE_PATH,
    router_config_text,
)


def test_native_config_is_the_agt013_pool_with_local_kv_encoding() -> None:
    config = yaml.safe_load(router_config_text())
    router = config["router"]
    assert router["checkpoint_path"] == f"/artifacts/{CHECKPOINT_REMOTE_PATH}"
    assert router["log_path"] == f"/artifacts/{DECISION_LOG_REMOTE_PATH}"
    assert router["mtrouter_device"] == "cuda"
    assert router["mtrouter_incremental_encode"] is True
    assert [worker["id"] for worker in config["workers"]] == list(WORKERS)
    assert {worker["id"]: worker["model"] for worker in config["workers"]} == WORKERS
    assert all(
        worker["openrouter_allow_fallbacks"] is False for worker in config["workers"]
    )
    assert all(worker["max_completion_tokens"] == 96 for worker in config["workers"])


def test_native_benchmark_has_a_fixed_small_request_envelope() -> None:
    assert benchmark.EXPECTED_NATIVE_REQUESTS == 63
    assert benchmark.EXPECTED_DIRECT_REQUESTS == 13
    assert benchmark.MAX_PROVIDER_REQUESTS == 76
    assert benchmark.MAX_COVERAGE_REQUESTS == 24
    assert benchmark.CONCURRENCY == 4
    assert benchmark.REPETITIONS == 2


def _result(request_id: str, path: str, total: float) -> dict[str, object]:
    return {
        "path": path,
        "case_id": "agentic-00",
        "scenario": "code_patch",
        "request_id": request_id,
        "selected_worker": "worker-a",
        "response_model": WORKERS["worker-a"],
        "time_to_first_event_seconds": total - 0.01,
        "time_to_first_token_seconds": total - 0.005,
        "total_seconds": total,
        "data_events": 3,
    }


def _decision(request_id: str, *, embedding_ms: float) -> dict[str, object]:
    return {
        "request_id": request_id,
        "error": "",
        "selected_worker": "worker-a",
        "worker_model": WORKERS["worker-a"],
        "served_model": WORKERS["worker-a"],
        "served_provider": PROVIDER_NAMES["worker-a"][0],
        "input_tokens": 100,
        "output_tokens": 10,
        "decision_latency_ms": embedding_ms + 1,
        "features": {
            "embedding_latency_ms": embedding_ms,
            "q_latency_ms": 1,
            "encode_mode": "full" if embedding_ms else "not_applicable",
            "serialized_tokens": 100,
        },
        "transport": {"attempts": [{"status": "success"}]},
        "settlement_cost_basis": {"provider_charged_usd": 0.001},
    }


def test_final_report_joins_client_and_decision_planes(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "EXPECTED_NATIVE_REQUESTS", 4)
    monkeypatch.setattr(benchmark, "EXPECTED_DIRECT_REQUESTS", 2)
    monkeypatch.setattr(benchmark, "MAX_PROVIDER_REQUESTS", 6)
    static = _result("static", "modal_static", 2.0)
    arc = _result("arc", "modal_arc", 4.0)
    stratified = _result("stratified", "modal_static", 1.0)
    endpoint = _result("endpoint", "modal_static", 1.0)
    direct = {
        "path": "direct",
        "case_id": "agentic-00",
        "scenario": "code_patch",
        "selected_worker": "worker-a",
        "response_model": WORKERS["worker-a"],
        "provider": PROVIDER_NAMES["worker-a"][0],
        "external_attempts": 1,
        "time_to_first_event_seconds": 0.5,
        "time_to_first_token_seconds": 0.5,
        "total_seconds": 1.0,
        "envoy_upstream_service_seconds": None,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "cost_usd": 0.001,
        "data_events": 3,
    }
    client = {
        "run_id": "test",
        "key_readiness": {"external_attempts": 1, "cost_usd": 0.001},
        "endpoint_probes": [endpoint],
        "coverage": [],
        "selected_cases": [
            {
                "case_id": "agentic-00",
                "scenario": "code_patch",
                "expected_worker": "worker-a",
            }
        ],
        "natural_phases": [
            {"path": "modal_static", "wall_seconds": 2.0, "results": [static]},
            {"path": "modal_arc", "wall_seconds": 4.0, "results": [arc]},
        ],
        "stratified_phases": [
            {"path": "direct", "wall_seconds": 1.0, "results": [direct]},
            {
                "path": "modal_static",
                "wall_seconds": 1.0,
                "results": [stratified],
            },
        ],
    }
    report = benchmark.finalize_report(
        client_report=client,
        decisions=[
            _decision("endpoint", embedding_ms=0),
            _decision("static", embedding_ms=0),
            _decision("arc", embedding_ms=1500),
            _decision("stratified", embedding_ms=0),
        ],
        actual_openrouter_cost_usd=0.006,
        deployment={"gpu": "L40S"},
        checkpoint={"checkpoint_sha256": "a" * 64},
    )
    assert report["status"] == "passed"
    assert report["actual_provider_requests"] == 6
    assert report["natural_comparison"]["arc_to_static_throughput_ratio"] == 0.5
    assert (
        report["natural_paths"]["modal_arc"]["embedding_latency"]["p50_seconds"] == 1.5
    )
    assert report["natural_paths"]["modal_arc"]["providers"] == ["Baidu"]
