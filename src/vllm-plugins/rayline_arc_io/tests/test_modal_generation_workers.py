# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _service_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "modal_generation_workers", SERVICE_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_generation_worker_secret_dependency_is_unconditional() -> None:
    source = SERVICE_PATH.read_text(encoding="utf-8")
    module = _tree()
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_worker_secret"
            for target in node.targets
        )
    )

    assert isinstance(assignment.value, ast.Call)
    assert source.count("secrets=[_worker_secret]") == EXPECTED_WORKER_COUNT
    assert "_worker_secrets" not in source


def test_generation_command_requires_auth_and_disables_prefix_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAYLINE_ARC_WORKER_API_KEY", raising=False)
    secret_payloads: list[dict[str, str]] = []

    class ModalChain:
        def entrypoint(self, *_args: object, **_kwargs: object) -> ModalChain:
            return self

        def uv_pip_install(self, *_args: object, **_kwargs: object) -> ModalChain:
            return self

        def env(self, *_args: object, **_kwargs: object) -> ModalChain:
            return self

    class ModalApp:
        def function(self, *_args: object, **_kwargs: object) -> object:
            return lambda function: function

    def identity_decorator(*_args: object, **_kwargs: object) -> object:
        return lambda function: function

    def modal_app(*_args: object, **_kwargs: object) -> ModalApp:
        return ModalApp()

    def secret_from_dict(payload: dict[str, str]) -> object:
        secret_payloads.append(payload)
        return object()

    modal_stub = SimpleNamespace(
        Secret=SimpleNamespace(from_dict=secret_from_dict),
        Image=SimpleNamespace(from_registry=lambda *_args, **_kwargs: ModalChain()),
        App=modal_app,
        Volume=SimpleNamespace(from_name=lambda *_args, **_kwargs: object()),
        concurrent=identity_decorator,
        web_server=identity_decorator,
    )
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    server_command = _service_module().server_command
    assert secret_payloads == [{"VLLM_API_KEY": ""}]

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VLLM_API_KEY is required"):
        server_command("synthetic/provider-a")  # type: ignore[operator]

    monkeypatch.setenv("VLLM_API_KEY", "public-test-only-key")
    command = server_command("synthetic/provider-b")  # type: ignore[operator]
    assert "--api-key" in command
    assert "public-test-only-key" in command
    assert "--no-enable-prefix-caching" in command
    assert command[command.index("--served-model-name") + 1] == "synthetic/provider-b"
