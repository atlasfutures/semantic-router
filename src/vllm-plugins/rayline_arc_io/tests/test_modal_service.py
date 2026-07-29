# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "modal_service.py"


def _source() -> str:
    return SERVICE_PATH.read_text()


def _decorator_call(function: ast.FunctionDef, suffix: str) -> ast.Call:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call) and ast.unparse(decorator.func).endswith(
            suffix
        ):
            return decorator
    raise AssertionError(f"missing {suffix} decorator")


def test_modal_service_is_proxy_authenticated_and_single_input() -> None:
    module = ast.parse(_source())
    serve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "serve"
    )
    web_server = _decorator_call(serve, "web_server")
    concurrent = _decorator_call(serve, "concurrent")
    function = _decorator_call(serve, "function")

    web_keywords = {keyword.arg: keyword.value for keyword in web_server.keywords}
    concurrency_keywords = {
        keyword.arg: keyword.value for keyword in concurrent.keywords
    }
    function_keywords = {keyword.arg for keyword in function.keywords}

    assert ast.literal_eval(web_keywords["requires_proxy_auth"]) is True
    assert ast.literal_eval(concurrency_keywords["max_inputs"]) == 1
    assert {"gpu", "cpu", "memory", "timeout", "scaledown_window", "volumes"} <= (
        function_keywords
    )


def test_modal_service_freezes_rung_b_vllm_contract() -> None:
    source = _source()
    for expected in (
        'MODEL_ID = "Qwen/Qwen3.5-0.8B"',
        'MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"',
        'VLLM_COMMIT = "918a2d159b718ab7f50d3fba87578e310034593d"',
        "CHUNK_SCHEDULE_TOKENS = 8_192",
        '"task": "embed"',
        '"pooling_type": "MEAN"',
        '"use_activation": True',
        '"--runner",',
        '"pooling",',
        '"--io-processor-plugin",',
        '"rayline_arc_io",',
        '"--no-enable-prefix-caching",',
        '"--no-enable-log-requests",',
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/pooling_params.py",
        "@sha256:",
        "RAYLINE_ARC_PLUGIN_SOURCE_DIGEST",
        "fetch --depth 1 origin",
    ):
        assert expected in source
