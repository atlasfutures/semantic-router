# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import importlib.util
import itertools
import json
import math
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Any

if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub

import openrouter_agentic_stage_metrics as stage_metrics
import openrouter_kv_cache_artifact_fixture as artifact_fixture
import openrouter_kv_cache_successor_benchmark as successor_benchmark
import openrouter_kv_cache_successor_contract as successor_contract
import openrouter_kv_cache_successor_remote_gate as remote_gate
import openrouter_kv_cache_successor_workload as workload
import openrouter_launch_authority as launch_authority
import pytest
import run_openrouter_kv_cache_native as native_launcher
import yaml
from openrouter_agentic_workload import WORKERS
from openrouter_fullstack_packets import packet_catalog
from openrouter_fullstack_state import EncoderDeployment

EXPECTED_REQUESTS_PER_DEPLOYMENT = 36
EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS = 156
EXPECTED_ENCODER_STATES = len(workload.SEQUENCE_IDS) * workload.STEPS
EXPECTED_PROBE_SESSIONS = len(workload.SEQUENCE_IDS)
EXPECTED_STARTED_SESSIONS_ON_FAILURE = 2


def _embedding(axis: float) -> tuple[float, ...]:
    values = [0.0] * workload.ENCODER_DIMENSION
    values[workload.ROUTING_AXIS_INDEX] = axis
    values[0] = math.sqrt(1.0 - axis**2)
    return tuple(values)


class _ParityClient:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls = 0
        self.fail_at = fail_at
        self.revisions: dict[str, int] = {}
        self.closed: list[str] = []

    def encode_with_embedding(
        self,
        episode_id: str,
        _turns: list[dict[str, str]],
    ) -> tuple[dict[str, Any], tuple[float, ...]]:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic parity failure")
        revision = self.revisions.get(episode_id, 0) + 1
        self.revisions[episode_id] = revision
        sequence_index = (self.calls - 1) // workload.STEPS
        worker = ("worker-c", "worker-a", "worker-b")[sequence_index]
        axis = {"worker-a": 0.005, "worker-b": 0.0, "worker-c": -0.005}[worker]
        serialized = 100 + (revision - 1) * 50
        retained = 0 if revision == 1 else serialized - 50
        return (
            {
                "action": "created" if revision == 1 else "appended",
                "revision": revision,
                "serialized_tokens": serialized,
                "retained_prefix_tokens": retained,
                "appended_tokens": serialized - retained,
                "latency_seconds": 0.01,
            },
            _embedding(axis),
        )

    def close_if_present(self, episode_id: str) -> bool:
        self.closed.append(episode_id)
        return episode_id in self.revisions


def test_successor_histories_are_public_strict_prefixes() -> None:
    sequences = workload.history_sequences()
    assert [sequence["sequence_id"] for sequence in sequences] == list(
        workload.SEQUENCE_IDS
    )
    assert workload.EXPECTED_REQUESTS_PER_DEPLOYMENT == EXPECTED_REQUESTS_PER_DEPLOYMENT
    for sequence in sequences:
        states = sequence["states"]
        assert len(states) == workload.STEPS
        assert [state["expected_worker"] for state in states] == sequence[
            "expected_selection_trace"
        ]
        for previous, current in itertools.pairwise(states):
            prior_messages = previous["messages"]
            current_messages = current["messages"]
            assert current_messages[: len(prior_messages)] == prior_messages
            assert len(current_messages) == len(prior_messages) + 2


def test_successor_prompts_have_no_model_specific_routing_anchors() -> None:
    encoded = json.dumps(
        [
            {
                "messages": state["messages"],
                "tools": state["tools"],
            }
            for sequence in workload.history_sequences()
            for state in sequence["states"]
        ]
    ).lower()
    forbidden = [
        *[worker.lower() for worker in WORKERS],
        *[model.lower() for model in WORKERS.values()],
        "deepseek-v4",
        "mimo-v2.5",
        "tencent/hy3",
    ]
    assert not any(value in encoded for value in forbidden)


def test_normalized_turns_preserve_the_agentic_tool_history() -> None:
    case = workload.history_sequences()[0]["states"][0]
    turns = workload.normalized_turns(case)
    assert [turn["role"] for turn in turns] == [
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert "[tool_call search_workspace]" in turns[1]["text"]
    assert turns[2]["text"].startswith("[tool_result search_workspace]\n")
    assert not any(turn["role"] == "system" for turn in turns)


def test_axis_classifier_covers_all_workers_with_margin() -> None:
    observations = {
        "worker-a": workload.classify_axis(0.005),
        "worker-b": workload.classify_axis(0.0),
        "worker-c": workload.classify_axis(-0.005),
    }
    assert {selected for selected, _gap in observations.values()} == set(WORKERS)
    assert all(
        gap > workload.MINIMUM_TOP_TWO_SCORE_GAP
        for _selected, gap in observations.values()
    )


def test_successor_contract_is_source_closed_and_exactly_bounded() -> None:
    contract = successor_contract.validate()
    assert contract["source_closed"] is True
    assert contract["launch_authorized"] is False
    assert contract["requires_new_budget_authority"] is True
    assert contract["logical_provider_requests"] == {
        "provider_preflight": 6,
        "semantic_cache_measurement": 72,
        "maximum_total": 78,
    }
    assert contract["maximum_external_attempts"] == EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    assert contract["report_schema_version"] == (
        "rayline.openrouter-kv-cache-comparison.v3"
    )
    assert "remote_encoder_trace_matches_offline_trace" in contract["acceptance_gates"]


def test_successor_remote_packet_is_source_closed_before_resource_use(tmp_path) -> None:
    encoder = EncoderDeployment(
        app_name="test-encoder",
        class_name="SessionEncoder",
        build_id="test-build",
        deployment_source_commit="test-source",
        plugin_source_digest="test-plugin",
    )
    packet = packet_catalog(
        tmp_path,
        encoder,
        tmp_path / "service.py",
        canary_key_limit_usd=0.25,
        maximum_canary_seconds=60,
    )["kv-cache-flashinfer-agt018"]

    assert packet.expected_run_id == successor_contract.RUN_ID
    assert packet.key_limit_usd == 0
    assert packet.maximum_seconds == 0
    assert packet.driver.name == "openrouter_kv_cache_successor_remote.py"
    assert packet.artifact_revision == successor_contract.ARTIFACT_REVISION
    assert (
        artifact_fixture.SUCCESSOR_ARTIFACT_REVISION
        == successor_contract.ARTIFACT_REVISION
    )
    with pytest.raises(SystemExit, match="source-closed"):
        launch_authority.verify_source_authority(
            "kv-cache-flashinfer-agt018",
            {},
            repo_root=tmp_path,
        )


def test_successor_native_mode_switches_identity_and_stays_source_closed(
    tmp_path,
) -> None:
    try:
        native_launcher._configure_generation("agt018")
        assert native_launcher.RUN_ID == successor_contract.RUN_ID
        assert native_launcher.APP_NAME == successor_contract.NATIVE_APP_NAME
        assert native_launcher.ARTIFACT_REVISION == successor_contract.ARTIFACT_REVISION
        assert (
            native_launcher.BENCHMARK.name
            == "openrouter_kv_cache_successor_benchmark.py"
        )
        assert native_launcher.KEY_LIMIT_USD == 0
        assert native_launcher.MAXIMUM_PAID_SECONDS == 0
        with pytest.raises(RuntimeError, match="authority is source-closed"):
            native_launcher._verify_authority(tmp_path)
    finally:
        native_launcher._configure_generation("agt017")


def test_successor_remote_client_requires_the_flashinfer_build(monkeypatch) -> None:
    monkeypatch.setenv(stage_metrics.ENCODER_BASE_URL_ENV, "https://encoder.invalid")
    monkeypatch.setenv(stage_metrics.ENCODER_KEY_ENV, "test-key")
    monkeypatch.setenv(stage_metrics.ENCODER_SECRET_ENV, "test-secret")
    monkeypatch.setenv(
        stage_metrics.ENCODER_BUILD_ID_ENV,
        successor_contract.REMOTE_ENGINE_BUILD_ID,
    )
    client = stage_metrics.encoder_client_from_environment(1.0)
    assert client.expected_engine_build_id == successor_contract.REMOTE_ENGINE_BUILD_ID


def test_successor_modal_app_is_registered_as_flashinfer() -> None:
    service = (
        Path(__file__).resolve().parents[3]
        / "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
    ).read_text()
    assert f'"{successor_contract.REMOTE_APP_NAME}": "flashinfer"' in service


def test_successor_artifact_revision_flows_through_fixture_and_config(
    tmp_path,
) -> None:
    artifact_fixture.generate(tmp_path, successor_contract.ARTIFACT_REVISION)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["artifact_id"] == successor_contract.ARTIFACT_REVISION

    config_path = (
        Path(__file__).resolve().parents[3]
        / "deploy/compose/rayline-arc/config-openrouter-kv-cache.yaml"
    )
    config = yaml.safe_load(config_path.read_text())
    revision = config["routing"]["decisions"][0]["algorithm"]["rayline_arc"][
        "artifact_revision"
    ]
    assert revision == "${RAYLINE_ARC_ARTIFACT_REVISION}"


def test_remote_encoder_gate_covers_nine_states_and_emits_no_payloads() -> None:
    client = _ParityClient()
    report = remote_gate.verify_remote_encoder(client, successor_contract.RUN_ID)

    assert report["status"] == "passed"
    assert report["states"] == EXPECTED_ENCODER_STATES
    assert report["sessions_closed"] == EXPECTED_PROBE_SESSIONS
    assert report["selected_workers"] == sorted(WORKERS)
    assert len(client.closed) == EXPECTED_PROBE_SESSIONS
    encoded = json.dumps(report)
    assert '"embedding":' not in encoded
    assert '"messages":' not in encoded
    assert '"turns":' not in encoded


def test_remote_encoder_gate_closes_started_sessions_after_failure() -> None:
    client = _ParityClient(fail_at=4)
    with pytest.raises(RuntimeError, match="synthetic parity failure"):
        remote_gate.verify_remote_encoder(client, successor_contract.RUN_ID)
    assert len(client.closed) == EXPECTED_STARTED_SESSIONS_ON_FAILURE


def _args(deployment: str) -> Namespace:
    return Namespace(
        deployment=deployment,
        base_url="https://gateway.invalid",
        metrics_url=("https://metrics.invalid" if deployment == "remote_vllm" else ""),
        run_id=successor_contract.RUN_ID,
        timeout_seconds=1.0,
        journal="",
    )


def test_remote_parity_gate_runs_before_any_routed_provider_cell(monkeypatch) -> None:
    events: list[str] = []
    client = object()
    monkeypatch.setattr(
        successor_benchmark,
        "encoder_client_from_environment",
        lambda _timeout: client,
    )
    monkeypatch.setattr(
        successor_benchmark,
        "verify_remote_encoder",
        lambda actual, _run_id: events.append("gate") or {"status": "passed"},
    )

    def fake_cell(**kwargs: Any) -> dict[str, Any]:
        events.append("provider")
        case = kwargs["case"]
        return {
            "sequence_id": kwargs["sequence_id"],
            "mode": kwargs["mode"],
            "episode": kwargs["episode"],
            "step": kwargs["step"],
            "selected_worker": case["expected_worker"],
            "expected_worker": case["expected_worker"],
        }

    monkeypatch.setattr(successor_benchmark, "_run_recorded_cell", fake_cell)
    report = successor_benchmark.run(_args("remote_vllm"))

    assert events[0] == "gate"
    assert events.count("provider") == EXPECTED_REQUESTS_PER_DEPLOYMENT
    assert report["pre_provider_remote_encoder_parity"] == {"status": "passed"}


def test_native_successor_benchmark_skips_remote_gate(monkeypatch) -> None:
    monkeypatch.setattr(
        successor_benchmark,
        "verify_remote_encoder",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected gate")),
    )

    def fake_cell(**kwargs: Any) -> dict[str, Any]:
        case = kwargs["case"]
        return {
            "sequence_id": kwargs["sequence_id"],
            "mode": kwargs["mode"],
            "episode": kwargs["episode"],
            "step": kwargs["step"],
            "selected_worker": case["expected_worker"],
            "expected_worker": case["expected_worker"],
        }

    monkeypatch.setattr(successor_benchmark, "_run_recorded_cell", fake_cell)
    report = successor_benchmark.run(_args("native_modal"))
    assert report["pre_provider_remote_encoder_parity"] is None


def test_successor_cell_rejects_natural_selection_divergence(monkeypatch) -> None:
    case = workload.history_sequences()[0]["states"][0]
    monkeypatch.setattr(
        successor_benchmark.base,
        "_run_cell",
        lambda **_kwargs: {"selected_worker": "worker-a"},
    )
    with pytest.raises(RuntimeError, match="natural selection trace diverged"):
        successor_benchmark._run_recorded_cell(
            args=_args("native_modal"),
            encoder_client=None,
            journal_path=None,
            sequence_id="code_patch",
            case=case,
            mode="retained",
            episode=0,
            step=0,
            ordinal=1,
        )
