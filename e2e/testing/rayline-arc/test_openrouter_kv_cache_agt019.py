# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub

import openrouter_kv_cache_agt019_contract as agt019_contract
import openrouter_kv_cache_artifact_fixture as artifact_fixture
import openrouter_kv_cache_matched_pair as matched_pair
import openrouter_launch_authority as launch_authority
import openrouter_modal_native_fixture as native_fixture
import pytest
import run_openrouter_kv_cache_native as native_launcher
import yaml
from openrouter_agentic_workload import PROVIDER_NAMES, PROVIDER_SLUGS, WORKERS
from openrouter_fullstack_packets import packet_catalog
from openrouter_fullstack_state import EncoderDeployment

EXPECTED_REQUESTS_PER_DEPLOYMENT = 36
EXPECTED_PREFLIGHT_REQUESTS = 6
EXPECTED_MEASUREMENT_REQUESTS = 72
EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS = 78
EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS = 156
EXPECTED_MAXIMUM_COMPLETION_TOKENS = 24
EXPECTED_AUTHORIZED_CUMULATIVE_USD = 154.31282402
MAXIMUM_EXPECTED_PACKET_USD = 9.54
LUNA_PROMPT_COST = 0.0000002
LUNA_CACHE_READ_COST = 0.00000002
LUNA_CACHE_WRITE_COST = 0.00000025
LUNA_COMPLETION_COST = 0.0000012
TOKEN_SCALE = 1_000_000
PRICE_FIELDS = (
    ("prompt_per_1m", "estimated_input_cost_per_token"),
    ("cached_input_per_1m", "estimated_cache_read_cost_per_token"),
    ("cache_write_per_1m", "estimated_cache_write_cost_per_token"),
    ("completion_per_1m", "estimated_output_cost_per_token"),
)
# raylineARCPriceEqual's tolerance, mirrored exactly. pytest.approx defaults to
# a relative 1e-6, which is a thousand times looser than the router allows and
# would let a config drift past this test but still fail the gate closed.
PRICE_ABSOLUTE_FLOOR = 1e-12
PRICE_RELATIVE_TOLERANCE = 1e-9


def _router_price_equal(configured: float, artifact: float) -> bool:
    """Replicate `raylineARCPriceEqual` in rayline_arc_readiness.go."""

    if not (math.isfinite(configured) and math.isfinite(artifact)):
        return False
    tolerance = max(PRICE_ABSOLUTE_FLOOR, abs(artifact) * PRICE_RELATIVE_TOLERANCE)
    return abs(configured - artifact) <= tolerance


SEQUENCE_WORKERS = (
    ("code_patch", "worker-a"),
    ("doc_edit", "worker-b"),
    ("data_query", "worker-c"),
)
STEPS_PER_SEQUENCE = 2
MODE = "retained"
EPISODE = 0

# Since the 2026-08-07 luna amendment, worker-b is pinned to OpenAI alone and
# can no longer be served by different providers on the two arms. The policy is
# therefore exercised through worker-c, whose frozen order still holds three.
DIVERGED_WORKER = "worker-c"
BASE_PROVIDER = "Tencent"
DIVERGED_REMOTE_PROVIDER = "Novita"
BASE_COMPLETION_TOKENS = 18
DIVERGED_REMOTE_COMPLETION_TOKENS = 11
NATIVE_SECONDS = 2.0
REMOTE_SECONDS = 1.0

EXPECTED_TOTAL_PAIRS = len(SEQUENCE_WORKERS) * STEPS_PER_SEQUENCE
PAIRS_PER_WORKER = STEPS_PER_SEQUENCE
EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER = (
    EXPECTED_TOTAL_PAIRS - PAIRS_PER_WORKER
)
NO_PAIRS = 0
FULL_COVERAGE = 1.0
EXPECTED_DIVERGED_COVERAGE = (
    EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER / EXPECTED_TOTAL_PAIRS
)
EXPECTED_REMOTE_TO_NATIVE_RATIO = REMOTE_SECONDS / NATIVE_SECONDS


def _arm_rows(
    *,
    seconds: float,
    diverged_worker: str = "",
    drop_field: str = "",
) -> list[dict[str, Any]]:
    """Build one arm's measurement cells for the three-worker workload."""

    rows: list[dict[str, Any]] = []
    for sequence_id, worker in SEQUENCE_WORKERS:
        diverged = worker == diverged_worker
        for step in range(STEPS_PER_SEQUENCE):
            row: dict[str, Any] = {
                "sequence_id": sequence_id,
                "mode": MODE,
                "episode": EPISODE,
                "step": step,
                "selected_worker": worker,
                "provider": (DIVERGED_REMOTE_PROVIDER if diverged else BASE_PROVIDER),
                "completion_tokens": (
                    DIVERGED_REMOTE_COMPLETION_TOKENS
                    if diverged
                    else BASE_COMPLETION_TOKENS
                ),
                "total_seconds": seconds,
            }
            if drop_field:
                row.pop(drop_field)
            rows.append(row)
    return rows


def test_agt019_contract_is_bound_with_the_matched_pair_gate() -> None:
    contract = agt019_contract.validate()

    assert agt019_contract.PREREGISTRATION_COMMIT == (
        "2b8a39b129a36a6b2e79aeb765195937ba7643f3"
    )
    assert agt019_contract.AUTHORIZATION_COMMIT == (
        "91af3bb17441bdd0c540f3d54b0217e43c82852d"
    )
    assert agt019_contract.SOURCE_CLOSED_KEY_LIMIT_USD_PER_ARM == (
        agt019_contract.AUTHORIZED_KEY_LIMIT_USD_PER_ARM
    )
    assert agt019_contract.SOURCE_CLOSED_MAXIMUM_PAID_WALL_SECONDS == (
        agt019_contract.AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS
    )
    assert contract["source_closed"] is False
    assert contract["launch_authorized"] is True
    assert contract["requires_new_budget_authority"] is False
    assert contract["run_id"] == "rayline-openrouter-kv-cache-agt019-20260805"
    assert contract["report_schema_version"] == (
        "rayline.openrouter-kv-cache-comparison.v4"
    )
    assert "matched_pair_comparability_policy" in contract["acceptance_gates"]
    assert "matched_completion_policy" not in contract["acceptance_gates"]
    assert contract["logical_provider_requests"] == {
        "provider_preflight": EXPECTED_PREFLIGHT_REQUESTS,
        "semantic_cache_measurement": EXPECTED_MEASUREMENT_REQUESTS,
        "maximum_total": EXPECTED_MAXIMUM_LOGICAL_PROVIDER_REQUESTS,
    }
    assert contract["maximum_external_attempts"] == EXPECTED_MAXIMUM_EXTERNAL_ATTEMPTS
    assert contract["maximum_completion_tokens"] == EXPECTED_MAXIMUM_COMPLETION_TOKENS
    assert (
        agt019_contract.EXPECTED_SEMANTIC_REQUESTS_PER_DEPLOYMENT
        == EXPECTED_REQUESTS_PER_DEPLOYMENT
    )
    assert contract["budget"]["authorized_cumulative_usd"] == (
        EXPECTED_AUTHORIZED_CUMULATIVE_USD
    )


def test_agt019_budget_preserves_the_frozen_reserve() -> None:
    receipt = agt019_contract.agt019_budget_receipt()

    assert receipt["authorized_cumulative_usd"] == EXPECTED_AUTHORIZED_CUMULATIVE_USD
    assert receipt["maximum_complete_packet_usd"] < MAXIMUM_EXPECTED_PACKET_USD
    assert receipt["reserve_after_complete_envelope_usd"] >= (
        agt019_contract.REQUIRED_FINAL_RESERVE_USD
    )


def test_identical_arms_are_fully_matched_everywhere() -> None:
    pairs = matched_pair.pair_cells(
        _arm_rows(seconds=NATIVE_SECONDS),
        _arm_rows(seconds=REMOTE_SECONDS),
    )
    report = matched_pair.comparability_report(pairs)

    assert report["total_pairs"] == EXPECTED_TOTAL_PAIRS
    assert report["fully_matched_pairs"] == EXPECTED_TOTAL_PAIRS
    assert report["fully_matched_coverage_fraction"] == FULL_COVERAGE
    assert report["inadmissible_workers"] == []
    assert report["admissible_lane"]["pairs"] == EXPECTED_TOTAL_PAIRS
    assert report["admissible_lane"]["remote_to_native_e2e_mean_ratio"] == (
        EXPECTED_REMOTE_TO_NATIVE_RATIO
    )
    assert all(
        lane["cross_deployment_e2e_admissible"] is True
        for lane in report["per_worker"].values()
    )
    assert matched_pair.policy_gate(report, EXPECTED_TOTAL_PAIRS) is True


def test_provider_fallthrough_labels_one_worker_inadmissible_without_failing() -> None:
    pairs = matched_pair.pair_cells(
        _arm_rows(seconds=NATIVE_SECONDS),
        _arm_rows(seconds=REMOTE_SECONDS, diverged_worker=DIVERGED_WORKER),
    )
    report = matched_pair.comparability_report(pairs)
    diverged_lane = report["per_worker"][DIVERGED_WORKER]

    assert report["inadmissible_workers"] == [DIVERGED_WORKER]
    assert diverged_lane["cross_deployment_e2e_admissible"] is False
    assert diverged_lane["total_pairs"] == PAIRS_PER_WORKER
    assert diverged_lane["fully_matched"]["pairs"] == NO_PAIRS
    assert diverged_lane["completion_matched"]["pairs"] == NO_PAIRS
    assert diverged_lane["native_providers"] == [BASE_PROVIDER]
    assert diverged_lane["remote_providers"] == [DIVERGED_REMOTE_PROVIDER]
    assert report["fully_matched_pairs"] == (
        EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER
    )
    assert report["fully_matched_coverage_fraction"] == EXPECTED_DIVERGED_COVERAGE
    assert report["admissible_lane"]["pairs"] == (
        EXPECTED_MATCHED_PAIRS_WITH_ONE_DIVERGED_WORKER
    )
    assert matched_pair.policy_gate(report, EXPECTED_TOTAL_PAIRS) is True


@pytest.mark.parametrize("field", ["provider", "completion_tokens"])
def test_missing_evidence_fields_are_defects_not_outcomes(field: str) -> None:
    with pytest.raises(RuntimeError, match="AGT019 matched-pair join"):
        matched_pair.pair_cells(
            _arm_rows(seconds=NATIVE_SECONDS),
            _arm_rows(seconds=REMOTE_SECONDS, drop_field=field),
        )


def test_unpaired_cells_are_rejected() -> None:
    remote = _arm_rows(seconds=REMOTE_SECONDS)[:-1]
    with pytest.raises(RuntimeError, match="unpaired cells"):
        matched_pair.pair_cells(_arm_rows(seconds=NATIVE_SECONDS), remote)


def test_duplicated_cells_are_rejected() -> None:
    native = _arm_rows(seconds=NATIVE_SECONDS)
    native.append(dict(native[0]))
    with pytest.raises(RuntimeError, match="duplicate cells"):
        matched_pair.pair_cells(native, _arm_rows(seconds=REMOTE_SECONDS))


def test_selection_divergence_is_rejected() -> None:
    remote = _arm_rows(seconds=REMOTE_SECONDS)
    remote[0] = {**remote[0], "selected_worker": "worker-c"}
    with pytest.raises(RuntimeError, match="selection divergence"):
        matched_pair.pair_cells(_arm_rows(seconds=NATIVE_SECONDS), remote)


def test_agt019_remote_packet_carries_bound_authority_limits(tmp_path) -> None:
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
    )["kv-cache-flashinfer-agt019"]

    assert packet.expected_run_id == agt019_contract.RUN_ID
    assert packet.key_limit_usd == (agt019_contract.AUTHORIZED_KEY_LIMIT_USD_PER_ARM)
    assert packet.maximum_seconds == (
        agt019_contract.AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS
    )
    assert packet.driver.name == "openrouter_kv_cache_successor_remote.py"
    assert packet.artifact_revision == agt019_contract.ARTIFACT_REVISION
    assert packet.encoder is not None
    assert packet.encoder.app_name == agt019_contract.REMOTE_APP_NAME
    with pytest.raises(subprocess.CalledProcessError):
        launch_authority.verify_source_authority(
            "kv-cache-flashinfer-agt019",
            {},
            repo_root=tmp_path,
        )


def test_agt019_native_mode_switches_identity_with_bound_authority(
    tmp_path,
) -> None:
    try:
        native_launcher._configure_generation("agt019")
        assert native_launcher.RUN_ID == agt019_contract.RUN_ID
        assert native_launcher.APP_NAME == agt019_contract.NATIVE_APP_NAME
        assert native_launcher.WEBHOOK_LABEL == (agt019_contract.NATIVE_WEBHOOK_LABEL)
        assert native_launcher.ARTIFACT_REVISION == (agt019_contract.ARTIFACT_REVISION)
        assert native_launcher.RUN_LABEL == "AGT019"
        assert native_launcher.TRAINING_STAGE == "openrouter_kv_cache_agt019"
        assert native_launcher.APP_TITLE == "Rayline AGT019"
        assert native_launcher.CONTEXT_SCHEMA_VERSION == (
            "rayline-router.modal-native-agt019.v1"
        )
        assert (
            native_launcher.BENCHMARK.name
            == "openrouter_kv_cache_successor_benchmark.py"
        )
        assert native_launcher.KEY_LIMIT_USD == (
            agt019_contract.AUTHORIZED_KEY_LIMIT_USD_PER_ARM
        )
        assert native_launcher.MAXIMUM_PAID_SECONDS == (
            agt019_contract.AUTHORIZED_MAXIMUM_PAID_WALL_SECONDS
        )
        with pytest.raises(subprocess.CalledProcessError):
            native_launcher._verify_authority(tmp_path)
    finally:
        native_launcher._configure_generation("agt017")


def test_dishonest_labeling_fails_the_policy_gate() -> None:
    pairs = matched_pair.pair_cells(
        _arm_rows(seconds=NATIVE_SECONDS),
        _arm_rows(seconds=REMOTE_SECONDS, diverged_worker=DIVERGED_WORKER),
    )
    mislabeled = matched_pair.comparability_report(pairs)
    mislabeled["inadmissible_workers"] = sorted(mislabeled["per_worker"])

    assert matched_pair.policy_gate(mislabeled, EXPECTED_TOTAL_PAIRS) is False

    undercounted = matched_pair.comparability_report(pairs)
    undercounted["total_pairs"] = EXPECTED_TOTAL_PAIRS - 1

    assert matched_pair.policy_gate(undercounted, EXPECTED_TOTAL_PAIRS) is False


def test_agt019_modal_app_is_registered_as_flashinfer() -> None:
    service = (
        Path(__file__).resolve().parents[3]
        / "src/vllm-plugins/rayline_arc_io/modal_session_service.py"
    ).read_text()
    assert f'"{agt019_contract.REMOTE_APP_NAME}": "flashinfer"' in service


def test_v7_artifact_pins_the_luna_lane_to_a_single_openai_provider(
    tmp_path,
) -> None:
    artifact_fixture.generate(tmp_path, artifact_fixture.AGT019_LUNA_ARTIFACT_REVISION)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    workers = {worker["id"]: worker for worker in manifest["workers"]}
    luna = workers["worker-b"]

    assert manifest["artifact_id"] == agt019_contract.ARTIFACT_REVISION
    assert luna["model"] == "openai/gpt-5.6-luna"
    assert luna["openrouter_provider_order"] == ["openai"]
    assert luna["openrouter_provider_name"] == "OpenAI"
    assert luna["openrouter_allow_fallbacks"] is False
    # Nothing is relaxed to accommodate the new lane: the capability filter
    # stays on, and the unsupported parameter is omitted instead.
    assert luna["openrouter_require_parameters"] is True
    assert "temperature" not in luna
    # Per-field maxima across OpenAI's three gpt-5.6-luna endpoint tags, so the
    # rate can only be over-stated. Unlike the other workers OpenAI prices
    # cache reads and writes separately, so these are not flat.
    assert luna["estimated_input_cost_per_token"] == LUNA_PROMPT_COST
    assert luna["estimated_cache_read_cost_per_token"] == LUNA_CACHE_READ_COST
    assert luna["estimated_cache_write_cost_per_token"] == LUNA_CACHE_WRITE_COST
    assert luna["estimated_output_cost_per_token"] == LUNA_COMPLETION_COST
    # The other two lanes are untouched by the amendment.
    assert workers["worker-a"]["temperature"] == 0
    assert workers["worker-c"]["temperature"] == 0


def test_compose_pricing_matches_the_v7_artifact(tmp_path) -> None:
    """The router's price-identity gate fails closed on any mismatch here.

    Without this check a divergence between the compose config and the artifact
    is only discovered by a paid launch aborting mid-flight.
    """

    artifact_fixture.generate(tmp_path, artifact_fixture.AGT019_LUNA_ARTIFACT_REVISION)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    artifact_workers = {worker["id"]: worker for worker in manifest["workers"]}
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[3]
            / "deploy/compose/rayline-arc/config-openrouter-kv-cache.yaml"
        ).read_text()
    )

    for model in config["providers"]["models"]:
        worker = artifact_workers[model["name"]]
        pricing = model["pricing"]
        assert model["provider_model_id"] == worker["model"]
        # The gate also refuses anything that is not USD or omits cache-write.
        assert pricing["currency"] == "USD"
        assert "cache_write_per_1m" in pricing
        for configured_key, artifact_key in PRICE_FIELDS:
            assert _router_price_equal(
                pricing[configured_key], worker[artifact_key] * TOKEN_SCALE
            ), f"{model['name']}.{configured_key} diverges from the artifact"


def test_both_arms_read_the_same_luna_worker_set() -> None:
    assert native_fixture.WORKERS is artifact_fixture.AGT019_LUNA_WORKERS
    assert WORKERS["worker-b"] == "openai/gpt-5.6-luna"
    assert PROVIDER_SLUGS["worker-b"] == ("openai",)
    assert PROVIDER_NAMES["worker-b"] == ("OpenAI",)
