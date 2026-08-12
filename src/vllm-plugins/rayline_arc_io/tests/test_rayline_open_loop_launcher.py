# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

launcher = importlib.import_module("rayline_open_loop_launcher")


def test_unregistered_open_loop_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id="unregistered-run",
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="no Rayline open-loop sweep"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []


def test_launcher_has_no_provider_or_qualification_path() -> None:
    source = (SCRIPT_DIR / "rayline_open_loop_launcher.py").read_text()

    assert launcher.OPEN_LOOP_ARMS == ("rayline_remote", "rayline_arc")
    assert "modal_inprocess" not in source
    assert "openrouter" not in source.lower()
    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source


def test_probe_cell_can_share_session_namespace_without_changing_receipt_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    packet_dir = tmp_path / "packet"
    (packet_dir / "cells/r030").mkdir(parents=True)
    context = SimpleNamespace(
        contract=SimpleNamespace(run_id="run"),
        packet_dir=packet_dir,
        semantic_root=REPO_ROOT,
    )
    cell = SimpleNamespace(label="r030")
    seen: list[str] = []

    def fake_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        seen.extend(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"status":"ok"}\n')
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(launcher, "_run", fake_run)
    receipt = launcher._probe_cell(
        context,
        cell,
        "rayline_arc",
        "http://example.test",
        output_dir,
        10.0,
        logical_arm="treatment",
        session_namespace="shared-affinity",
    )

    assert seen[seen.index("--run-id") + 1] == "run:r030:shared-affinity"
    assert (output_dir / "treatment.json").exists()
    assert receipt == {"status": "ok"}


def _evidence_context(tmp_path: Path, contract: SimpleNamespace) -> SimpleNamespace:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return SimpleNamespace(contract=contract, output_dir=output_dir)


def _flashinfer_contract() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run",
        encoder_app_name="rayline-arc-session-encoder-flashinfer-perf031",
        encoder_build_id=(
            "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca+gdn-flashinfer-eager"
        ),
        encoder_gdn_prefill_backend="flashinfer",
        encoder_gpu="H100",
    )


def _encoder(contract: SimpleNamespace, response: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        app_name=contract.encoder_app_name,
        base_url="https://encoder.test",
        client=SimpleNamespace(request=lambda _method, _path: response),
    )


def test_deployment_evidence_records_the_observed_engine_sizing(
    tmp_path: Path,
) -> None:
    contract = _flashinfer_contract()
    context = _evidence_context(tmp_path, contract)
    lines = ["Setting attention block size to 544 tokens"]
    encoder = _encoder(
        contract,
        {
            "schema_version": "rayline.arc.session-startup-log-response.v1",
            "engine_build_id": contract.encoder_build_id,
            "captured": True,
            "lines": lines,
        },
    )

    evidence = launcher._write_deployment_evidence(context, encoder)

    written = json.loads((context.output_dir / "deployment-evidence.json").read_text())
    assert written == evidence
    assert evidence["gdn_prefill_backend"] == "flashinfer"
    assert evidence["encoder_app_name"] == contract.encoder_app_name
    assert evidence["startup_log_captured"] is True
    assert evidence["startup_log"] == lines


def test_deployment_evidence_never_claims_an_uncaptured_startup_log(
    tmp_path: Path,
) -> None:
    contract = _flashinfer_contract()
    context = _evidence_context(tmp_path, contract)
    encoder = _encoder(
        contract,
        {
            "schema_version": "rayline.arc.session-startup-log-response.v1",
            "engine_build_id": contract.encoder_build_id,
            "captured": False,
            "lines": [],
        },
    )

    evidence = launcher._write_deployment_evidence(context, encoder)

    # An empty capture is evidence that nothing was observed, not evidence that
    # the engine reported nothing. It must not stop the run either.
    assert evidence["startup_log_captured"] is False
    assert evidence["startup_log"] == []


def test_deployment_evidence_fails_closed_on_a_divergent_engine_identity(
    tmp_path: Path,
) -> None:
    contract = _flashinfer_contract()
    context = _evidence_context(tmp_path, contract)
    encoder = _encoder(
        contract,
        {
            "schema_version": "rayline.arc.session-startup-log-response.v1",
            "engine_build_id": "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca",
            "captured": True,
            "lines": [],
        },
    )

    with pytest.raises(launcher.LaunchError, match="build id differs"):
        launcher._write_deployment_evidence(context, encoder)

    assert not (context.output_dir / "deployment-evidence.json").exists()


def test_expected_arc_requests_comes_from_the_contract() -> None:
    """The count that aborts a cell must not be a launcher constant.

    `EXPECTED_ARC_REQUESTS` was `MEASURED_CASES + WARMUP_CASES` read from a
    module the launcher does not write, and it is only ever compared after a
    cell has already burned paid GPU time. A packet with any other corpus size
    would have aborted the run at the most expensive possible moment. The
    count now comes from the contract whose packet preflight already checked.
    """

    contract = SimpleNamespace(measured_cases=12, warmup_cases=3)

    assert launcher._expected_arc_requests(contract) == 15
    assert not hasattr(launcher, "EXPECTED_ARC_REQUESTS")
