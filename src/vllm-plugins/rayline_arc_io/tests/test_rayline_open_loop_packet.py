# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

concurrency_test = importlib.import_module("test_rayline_concurrency_packet")
concurrency_packet = importlib.import_module("rayline_concurrency_packet")
open_packet = importlib.import_module("rayline_open_loop_packet")
open_probe = importlib.import_module("rayline_open_loop_probe")


def _source_packet(root: Path) -> Path:
    directional = concurrency_test._directional_packet(root)
    source = root / "concurrency"
    concurrency_packet.build_sweep_packet(
        source_packet_dir=directional, output_dir=source
    )
    return source


def test_open_loop_packet_preserves_corpus_and_freezes_rate_identity(
    tmp_path: Path,
) -> None:
    source = _source_packet(tmp_path)
    output = tmp_path / "open-loop"

    manifest = open_packet.build_open_loop_packet(
        source_packet_dir=source, output_dir=output
    )

    assert manifest["measured_cases"] == open_packet.MEASURED_CASES
    assert manifest["corpus_sha256"] == open_packet._sha256(source / "corpus.json")
    assert list(manifest["cells"]) == ["r015", "r030", "r045"]
    assert len({cell["workload_sha256"] for cell in manifest["cells"].values()}) == len(
        open_packet.OFFERED_RATES
    )
    for rate in open_packet.OFFERED_RATES:
        label = open_packet.rate_label(rate)
        warmup, measured, identity, _worker_map, workload = (
            open_probe.load_open_loop_packet(
                arm="rayline_arc",
                corpus_path=output / "corpus.json",
                workload_path=output / f"cells/{label}/workload.json",
                topology_path=output / "topology.json",
                identity_path=output / f"cells/{label}/identity.json",
            )
        )
        assert len(warmup) == open_packet.WARMUP_CASES
        assert len(measured) == open_packet.MEASURED_CASES
        assert identity["measurement_scope"] == open_packet.MEASUREMENT_SCOPE
        assert workload["offered_rate_rps"] == rate


def test_open_loop_packet_refuses_overwrite(tmp_path: Path) -> None:
    source = _source_packet(tmp_path)
    output = tmp_path / "open-loop"
    output.mkdir()

    with pytest.raises(open_packet.OpenLoopPacketError, match="already exists"):
        open_packet.build_open_loop_packet(source_packet_dir=source, output_dir=output)
