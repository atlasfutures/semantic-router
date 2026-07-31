# SPDX-License-Identifier: Apache-2.0

import ast
import os
from pathlib import Path

import pytest

SERVICE_PATH = Path(__file__).resolve().parents[1] / "modal_generation_workers.py"
EXPECTED_WORKER_COUNT = 2


def _tree() -> ast.Module:
    return ast.parse(SERVICE_PATH.read_text(encoding="utf-8"))


def _assignment_value(name: str) -> object:
    for node in _tree().body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment: {name}")


def test_generation_workers_are_revision_pinned_and_bounded() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    assert _assignment_value("MODEL_ID") == "Qwen/Qwen3.5-0.8B"
    assert _assignment_value("MODEL_REVISION") == (
        "2fc06364715b967f1860aea9cf38778875588b17"
    )
    assert _assignment_value("MAX_CONTAINERS") == 1
    assert "gpu=GPU_TYPE" in source
    assert source.count("max_containers=MAX_CONTAINERS") == EXPECTED_WORKER_COUNT
    assert source.count("@modal.web_server(PORT") == EXPECTED_WORKER_COUNT


def test_generation_command_requires_auth_and_disables_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace: dict[str, object] = {"os": os}
    module = _tree()
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "server_command"
    )
    assignments = [
        node
        for node in module.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name)
            and target.id
            in {
                "MODEL_ID",
                "MODEL_REVISION",
                "PORT",
                "MAX_MODEL_LEN",
                "MAX_CONCURRENT_INPUTS",
            }
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    isolated = ast.Module(body=[*assignments, function], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(isolated), str(SERVICE_PATH), "exec"),
        namespace,
    )
    server_command = namespace["server_command"]

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_API_KEY is required"):
        server_command("synthetic/provider-a")  # type: ignore[operator]

    monkeypatch.setenv("VLLM_API_KEY", "public-test-only-key")
    command = server_command("synthetic/provider-b")  # type: ignore[operator]
    assert "--api-key" in command
    assert "public-test-only-key" in command
    assert "--no-enable-prefix-caching" in command
    assert command[command.index("--served-model-name") + 1] == "synthetic/provider-b"
