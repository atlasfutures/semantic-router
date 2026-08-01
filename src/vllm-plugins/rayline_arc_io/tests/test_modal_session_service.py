# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "modal_session_service.py"
MAX_CONCURRENT_INPUTS = 32
MAX_CONTAINERS = 1


def source() -> str:
    return SERVICE_PATH.read_text()


def decorator_call(node: ast.AST, suffix: str) -> ast.Call:
    for decorator in node.decorator_list:  # type: ignore[attr-defined]
        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func).endswith(
            suffix
        ):
            return decorator
    raise AssertionError(f"missing {suffix} decorator")


def test_session_service_is_authenticated_and_bounded() -> None:
    module = ast.parse(source())
    service = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SessionEncoder"
    )
    function = decorator_call(service, "app.cls")
    concurrent = decorator_call(service, "concurrent")
    web = next(
        node
        for node in service.body
        if isinstance(node, ast.FunctionDef) and node.name == "web"
    )
    asgi = decorator_call(web, "asgi_app")

    function_keywords = {keyword.arg for keyword in function.keywords}
    concurrency_keywords = {
        keyword.arg: keyword.value for keyword in concurrent.keywords
    }
    web_keywords = {keyword.arg: keyword.value for keyword in asgi.keywords}

    assert {
        "gpu",
        "cpu",
        "memory",
        "timeout",
        "scaledown_window",
        "max_containers",
        "volumes",
    } <= function_keywords
    function_keyword_values = {
        keyword.arg: keyword.value for keyword in function.keywords
    }
    assert ast.literal_eval(function_keyword_values["max_containers"]) == MAX_CONTAINERS
    assert ast.literal_eval(concurrency_keywords["max_inputs"]) == MAX_CONCURRENT_INPUTS
    assert ast.literal_eval(web_keywords["requires_proxy_auth"]) is True


def test_session_service_freezes_the_proven_retained_vllm_runtime() -> None:
    service_source = source()
    for expected in (
        'VLLM_COMMIT = "77a901d233499ef588370f93056f82dae15bcb93"',
        'VLLM_REPOSITORY = "https://github.com/atlasfutures/vllm.git"',
        'GPU_TYPE = "H100"',
        "MAX_SESSIONS = 8",
        "MAX_RESIDENT_TOKENS = MAX_SESSIONS * MAX_SERIALIZED_TOKENS",
        "IDLE_TTL_SECONDS = 5 * 60",
        '"vllm/outputs.py"',
        '"vllm/v1/engine/output_processor.py"',
        '"vllm/v1/engine/pooling_session.py"',
        "python3 -m py_compile",
        "enable_prefix_caching=False",
        'gdn_prefill_backend="torch_reference"',
        "VLLMRetainedPoolingBackendFactory",
        "VLLMSessionEngineMetricsProvider",
        "self._engine.get_scheduler_load",
        "self._coordinator.append_metrics_snapshot",
        "SessionCoordinator",
        "create_session_app",
    ):
        assert expected in service_source
