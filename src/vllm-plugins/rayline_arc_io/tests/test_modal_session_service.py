# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "modal_session_service.py"
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
    # The encoder is deliberately unpinned: a region= pin costs a 1.75x Modal
    # narrow-region multiplier and PERF011/PERF014 measured it as slower, not
    # faster. Re-pin only with a measurement that clears a placement gate.
    assert "region" not in function_keywords
    assert ast.literal_eval(function_keyword_values["max_containers"]) == MAX_CONTAINERS
    # The ingress cap is app-conditional (PERF034 widens it), so the decorator
    # must reference the module constant whose definition the freeze test pins.
    assert ast.unparse(concurrency_keywords["max_inputs"]) == "MAX_CONCURRENT_INPUTS"
    assert ast.literal_eval(web_keywords["requires_proxy_auth"]) is True


def test_session_service_freezes_the_proven_retained_vllm_runtime() -> None:
    service_source = source()
    for expected in (
        'VLLM_COMMIT = "9f5ea81ca0aa570aea46baf82311a1139c1267ca"',
        'VLLM_REPOSITORY = "https://github.com/atlasfutures/vllm.git"',
        'GPU_TYPE = "H100"',
        "MAX_SESSIONS = 32 if APP_NAME in PERF034_APP_PROFILES else 8",
        "MAX_CONCURRENT_INPUTS = 64 if APP_NAME in PERF034_APP_PROFILES else 32",
        "MAX_RESIDENT_TOKENS = MAX_SESSIONS * MAX_SERIALIZED_TOKENS",
        "IDLE_TTL_SECONDS = 5 * 60",
        '"vllm/outputs.py"',
        '"vllm/v1/engine/output_processor.py"',
        '"vllm/v1/engine/pooling_session.py"',
        "python3 -m py_compile",
        "enable_prefix_caching=False",
        "enable_logging_iteration_details=True",
        "gdn_prefill_backend=runtime_backend",
        "VLLMRetainedPoolingBackendFactory",
        "VLLMSessionEngineMetricsProvider",
        "self._engine.get_scheduler_load",
        "self._coordinator.append_metrics_snapshot",
        "SessionCoordinator",
        "create_session_app",
    ):
        assert expected in service_source


def test_session_service_captures_engine_sizing_around_the_engine_build() -> None:
    service_source = source()

    # The capture must wrap exactly the engine build; sizing lines are emitted
    # nowhere else, and a wider scope would retain unrelated request logging.
    assert "with capture_startup_log() as startup_capture:" in service_source
    assert (
        "            self._engine = AsyncLLM.from_engine_args(engine_args)"
        in service_source
    )
    assert "self._startup_log = tuple(startup_capture.lines)" in service_source
    assert "startup_log=self._startup_log," in service_source


def test_session_service_allows_only_the_frozen_scaleout_app_names() -> None:
    service_source = source()

    assert 'DEFAULT_APP_NAME = "rayline-arc-session-encoder"' in service_source
    assert '"rayline-arc-session-encoder-a"' in service_source
    assert '"rayline-arc-session-encoder-b"' in service_source
    assert '"rayline-arc-session-encoder-c"' in service_source
    assert 'os.environ.get("RAYLINE_ARC_SESSION_APP_NAME", DEFAULT_APP_NAME)' in (
        service_source
    )
    assert "unsupported Rayline ARC session app name" in service_source


def test_session_service_confines_perf030_backends_to_exact_app_names() -> None:
    service_source = source()

    assert '"rayline-arc-session-encoder-reference-perf030": "torch_reference"' in (
        service_source
    )
    assert '"rayline-arc-session-encoder-flashinfer-perf030": "flashinfer"' in (
        service_source
    )
    assert 'EXPERIMENT_APP_PROFILES.get(APP_NAME, "torch_reference")' in service_source
    assert "if APP_NAME in EXPERIMENT_APP_PROFILES" in service_source
    assert '"RAYLINE_ARC_SESSION_APP_NAME": APP_NAME' in service_source
    assert "runtime_backend, runtime_build_id = _runtime_profile()" in service_source
    assert "engine_build_id=runtime_build_id," in service_source


def test_session_service_confines_agt017_flashinfer_to_its_exact_app_name() -> None:
    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-agt017": "flashinfer"'
        in service_source
    )
    assert "EXPERIMENT_APP_PROFILES =" in service_source


def test_session_service_confines_agt018_flashinfer_to_its_exact_app_name() -> None:
    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-agt018": "flashinfer"'
        in service_source
    )
    assert "**AGT018_APP_PROFILES" in service_source


def test_session_service_confines_agt019_flashinfer_to_its_exact_app_name() -> None:
    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-agt019": "flashinfer"'
        in service_source
    )
    assert "**AGT019_APP_PROFILES" in service_source


def test_session_service_confines_perf031_flashinfer_to_its_exact_app_name() -> None:
    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-perf031": "flashinfer"'
        in service_source
    )
    assert "**PERF031_APP_PROFILES" in service_source
    # PERF031 arm 0 is the negative control and must stay unprofiled so its
    # engine build id remains PERF021's bare `vllm@...` identity.
    assert '"rayline-arc-session-encoder-reference-perf031"' not in service_source


def test_session_service_confines_perf034_cap_raise_to_its_exact_app_name() -> None:
    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-perf034": "flashinfer"'
        in service_source
    )
    assert "**PERF034_APP_PROFILES" in service_source
    # The cap raise is scoped to the PERF034 app: every other app, including
    # the default live encoder, must keep 8 sessions and 32 ingress inputs.
    assert "MAX_SESSIONS = 32 if APP_NAME in PERF034_APP_PROFILES else 8" in (
        service_source
    )
    assert "MAX_CONCURRENT_INPUTS = 64 if APP_NAME in PERF034_APP_PROFILES else 32" in (
        service_source
    )


def test_allowed_app_names_extend_with_every_registered_experiment() -> None:
    module = ast.parse(source())
    allowed = next(
        node.value
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ALLOWED_APP_NAMES"
            for target in node.targets
        )
    )
    starred = {
        ast.unparse(element.value)
        for element in allowed.elts  # type: ignore[attr-defined]
        if isinstance(element, ast.Starred)
    }

    # A new experiment profile must become launchable by registration alone;
    # nothing may have to remember to also edit the allow-list.
    assert "EXPERIMENT_APP_PROFILES" in starred
    assert "SCALEOUT_APP_NAMES" in starred
