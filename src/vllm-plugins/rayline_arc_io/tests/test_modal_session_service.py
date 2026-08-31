# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

SERVICE_PATH = Path(__file__).resolve().parents[1] / "modal_session_service.py"
MAX_CONTAINERS = 1
# The complete GPU routing, frozen as one block so no branch can move without
# this file noticing. PERF035 owns the L4 and the RTX PRO 6000 set owns the
# other Cloud Run card; every other app name, including every closed run's,
# stays H100.
GPU_TYPE_CONDITIONAL = (
    "if APP_NAME in PERF035_APP_PROFILES:\n"
    '    GPU_TYPE = "L4"\n'
    "elif APP_NAME in RTX6000_APP_PROFILES:\n"
    '    GPU_TYPE = "RTX-PRO-6000"\n'
    "else:\n"
    '    GPU_TYPE = "H100"'
)
# The two cross-cutting membership sets, frozen with their exact contents.
# A packet joins a card or a cap by naming itself in one of these; nothing
# else routes, so an app that names itself nowhere gets H100 and eight lanes.
CARD_AND_CAP_SETS = (
    "RTX6000_APP_PROFILES = {**PERF036_APP_PROFILES, **PERF037_APP_PROFILES}",
    "CAP_RAISED_APP_PROFILES = {**PERF034_APP_PROFILES, **PERF037_APP_PROFILES}",
)
MAX_SESSIONS_CONDITIONAL = (
    "MAX_SESSIONS = 32 if APP_NAME in CAP_RAISED_APP_PROFILES else 8"
)
MAX_INPUTS_CONDITIONAL = (
    "MAX_CONCURRENT_INPUTS = 64 if APP_NAME in CAP_RAISED_APP_PROFILES else 32"
)


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
        GPU_TYPE_CONDITIONAL,
        *CARD_AND_CAP_SETS,
        MAX_SESSIONS_CONDITIONAL,
        MAX_INPUTS_CONDITIONAL,
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
    # The cap raise is scoped to the apps that name themselves in the raised
    # set -- PERF034 and PERF037 only. Every other app, including the default
    # live encoder, must keep 8 sessions and 32 ingress inputs.
    assert MAX_SESSIONS_CONDITIONAL in service_source
    assert MAX_INPUTS_CONDITIONAL in service_source
    assert (
        "CAP_RAISED_APP_PROFILES = "
        "{**PERF034_APP_PROFILES, **PERF037_APP_PROFILES}" in service_source
    )


def test_session_service_confines_perf035_l4_to_its_exact_app_name() -> None:
    """The GPU class is per-app, because every closed run's evidence claims H100.

    PERF035 measures the deployment target's silicon, so its app -- and only
    its app -- deploys on an L4. The cap raise must not follow it: a 24 GB card
    cannot hold the 32-lane corpus at all, so PERF035 stays on the 8/32 caps
    every non-PERF034 app keeps.
    """

    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-perf035-l4": "flashinfer"'
        in service_source
    )
    assert "**PERF035_APP_PROFILES" in service_source
    assert GPU_TYPE_CONDITIONAL in service_source
    # The L4 app is not in the cap-raise profile set, so the conditionals the
    # PERF034 test pins already hold it at 8 sessions and 32 ingress inputs.
    assert "PERF035" not in service_source.split("MAX_SESSIONS = ")[1].split("\n")[0]
    assert (
        "PERF035"
        not in service_source.split("MAX_CONCURRENT_INPUTS = ")[1].split("\n")[0]
    )


def test_session_service_confines_perf036_rtx6000_to_its_exact_app_name() -> None:
    """The RTX PRO 6000 arm owns one app name, and the caps do not follow it.

    PERF036 measures Cloud Run's other GPU class, so its app -- and only its
    app -- deploys on the 96 GB card. Unlike the L4, eight lanes fit this card
    by the derived cap alone (24 GiB worst case against a ~87 GB pool), but
    the caps still stay at 8/32: the packet's one variable is the silicon.
    """

    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-perf036-rtx6000": "flashinfer"'
        in service_source
    )
    assert "**PERF036_APP_PROFILES" in service_source
    assert GPU_TYPE_CONDITIONAL in service_source
    # Not in the cap-raise profile set, so the conditionals the PERF034 test
    # pins already hold it at 8 sessions and 32 ingress inputs.
    assert "PERF036" not in service_source.split("MAX_SESSIONS = ")[1].split("\n")[0]
    assert (
        "PERF036"
        not in service_source.split("MAX_CONCURRENT_INPUTS = ")[1].split("\n")[0]
    )


def test_session_service_gives_perf037_both_the_card_and_the_cap_raise() -> None:
    """The first app that needs two earlier packets' deviations at once.

    A burst claim for the deployment target has to be measured on the target's
    silicon, and it cannot be offered at all on eight lanes -- eight is the
    most that can ever be in flight there, so a burst above eight concurrent
    decisions would register as client-side lateness rather than as load on
    the encoder. PERF037 therefore carries PERF036's card and PERF034's 32/64
    caps together, and is the only app that does.
    """

    service_source = source()

    assert (
        '"rayline-arc-session-encoder-flashinfer-perf037-rtx6000-32lane"'
        ': "flashinfer"' in service_source
    )
    assert "**PERF037_APP_PROFILES" in service_source
    for expected in CARD_AND_CAP_SETS:
        assert expected in service_source
    assert GPU_TYPE_CONDITIONAL in service_source
    assert MAX_SESSIONS_CONDITIONAL in service_source
    assert MAX_INPUTS_CONDITIONAL in service_source


def test_only_perf037_holds_membership_of_both_cross_cutting_sets() -> None:
    """Two deviations at once is the exception, so it may not spread quietly.

    The card set and the cap set intersect in exactly one app name today. A
    packet that joins both without saying so in its own contract would deploy
    on silicon its evidence does not claim, at a lane count its packet does
    not carry -- the two failure modes these sets exist to prevent.
    """

    module = ast.parse(source())
    profiles = {
        target.id: {ast.literal_eval(key) for key in node.value.keys if key is not None}
        for node in module.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id.endswith("_APP_PROFILES")
    }

    def members(name: str) -> set[str]:
        expanded: set[str] = set()
        for node in ast.parse(source()).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                continue
            for key, value in zip(node.value.keys, node.value.values, strict=True):
                if key is None and isinstance(value, ast.Name):
                    expanded |= profiles[value.id]
                elif key is not None:
                    expanded.add(ast.literal_eval(key))
        return expanded

    card = members("RTX6000_APP_PROFILES")
    caps = members("CAP_RAISED_APP_PROFILES")
    assert card & caps == {
        "rayline-arc-session-encoder-flashinfer-perf037-rtx6000-32lane"
    }
    assert "rayline-arc-session-encoder-flashinfer-perf036-rtx6000" in card - caps
    assert "rayline-arc-session-encoder-flashinfer-perf034" in caps - card


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
