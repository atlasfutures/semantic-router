# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCHER_PATH = REPO_ROOT / "e2e/testing/rayline-arc/run_modal_fullstack.py"
COMPOSE_PATH = REPO_ROOT / "deploy/compose/rayline-arc/compose.yaml"
CONFIG_PATH = REPO_ROOT / "deploy/compose/rayline-arc/config.yaml"
REAL_WORKER_ENVOY_PATH = (
    REPO_ROOT / "deploy/compose/rayline-arc/envoy-real-workers.yaml"
)
EXPECTED_STARTUP_SECONDS = 240
EXPECTED_CANARY_SECONDS = 15 * 60
EXPECTED_MODAL_VERSION = "1.5.1"
EXPECTED_TLS_CLUSTERS = 2


def _tree() -> ast.Module:
    return ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"))


def _integer_value(node: ast.expr) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _integer_value(node.left) * _integer_value(node.right)
    raise AssertionError("assignment is not a static integer expression")


def _assignment_value(name: str) -> object:
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return _integer_value(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_real_worker_launcher_pins_real_encoder_and_global_deadlines() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert _assignment_value("MAX_STARTUP_SECONDS") == EXPECTED_STARTUP_SECONDS
    assert _assignment_value("MAX_CANARY_SECONDS") == EXPECTED_CANARY_SECONDS
    assert "rayline-arc-session-encoder-sessionenc-2d82ac.modal.run" in source
    assert "vllm@b1049f6dd95c27d2e1b052eebc3b1a7f9f41195f" in source
    assert 'ROUTER_HEALTH_URL = "http://127.0.0.1:18082/health"' in source
    assert "_wait_http(ROUTER_HEALTH_URL)" in source
    assert "timeout=MAX_CANARY_SECONDS" in source
    assert 'ENCODER_APP_ID = "ap-rs3UkEn5XUnWjrZOXYbkuB"' in source
    assert '"container", "stop", container_id, "--yes"' in source
    assert "_stop_encoder_containers(modal_command, environment)" in source


def test_launcher_uses_one_pinned_modal_sdk_for_api_and_cli() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert f'REQUIRED_MODAL_VERSION = "{EXPECTED_MODAL_VERSION}"' in source
    assert "modal.__version__ != REQUIRED_MODAL_VERSION" in source
    assert 'modal_command = [sys.executable, "-m", "modal"]' in source
    assert 'shutil.which("modal")' not in source


def test_launcher_selects_the_bounded_benchmark_driver_without_a_paid_bulk_flag() -> (
    None
):
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert (
        '"benchmark": Path(__file__).with_name("modal_fullstack_benchmark.py")'
        in source
    )
    assert (
        'parser.add_argument("--mode", choices=sorted(DRIVERS), default="canary")'
        in source
    )
    assert "execute-paid-1000" not in source


def test_launcher_selects_the_bounded_stage_diagnostic() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert (
        '"diagnostic": Path(__file__).with_name("modal_fullstack_diagnostic.py")'
        in source
    )


def test_encoder_identity_is_dynamic_but_timeouts_remain_typed() -> None:
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    config = CONFIG_PATH.read_text(encoding="utf-8")
    assert "RAYLINE_ARC_ENCODER_BASE_URL:" in compose
    assert "RAYLINE_ARC_ENCODER_BUILD_ID:" in compose
    assert "base_url: ${RAYLINE_ARC_ENCODER_BASE_URL}" in config
    assert "expected_build_id: ${RAYLINE_ARC_ENCODER_BUILD_ID}" in config
    assert "connect_timeout_seconds: 10" in config
    assert "total_timeout_seconds: 180" in config


def test_paid_launcher_selects_dedicated_tls_worker_routes() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    compose = COMPOSE_PATH.read_text(encoding="utf-8")
    envoy = REAL_WORKER_ENVOY_PATH.read_text(encoding="utf-8")

    assert "REAL_WORKER_ENVOY_FILE = (" in launcher
    assert (
        '"RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH": str(REAL_WORKER_ENVOY_FILE)' in launcher
    )
    assert "${RAYLINE_ARC_E2E_ENVOY_CONFIG_PATH:-./envoy.yaml}" in compose
    assert "cluster: worker_a" in envoy
    assert "cluster: worker_b" in envoy
    assert (
        "atlasfutures-dev--rayline-arc-generation-workers-worker-a.modal.run" in envoy
    )
    assert (
        "atlasfutures-dev--rayline-arc-generation-workers-worker-b.modal.run" in envoy
    )
    assert envoy.count("envoy.transport_sockets.tls") == EXPECTED_TLS_CLUSTERS
    assert envoy.count("/etc/ssl/certs/ca-certificates.crt") == EXPECTED_TLS_CLUSTERS
    assert "fake-provider" not in envoy
    assert "authorization" not in envoy.lower()
