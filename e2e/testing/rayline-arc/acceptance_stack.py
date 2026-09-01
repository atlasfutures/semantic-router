#!/usr/bin/env python3
"""Compose-free process harness for the Rayline ARC acceptance checks.

The ported hermetic stack drove Envoy, Redis and a docker compose project.
None of that is reachable on a developer laptop without Docker, and none of it
is what CP4 needs to see. This harness keeps the same three moving parts --
the synthetic artifact, the contract-faithful fake encoder, the fake provider
-- and starts the real router binary next to them as an ordinary child
process. Requests go to the decision-only management endpoint, which runs the
whole ARC path (episode lease, turn projection, encoder call, policy) and
stops exactly where dispatch would begin.

Episode state is in memory by default, so a run needs nothing but Python and
the router binary. Set REDIS_ADDR to run the same checks against a real Redis
episode store instead.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import artifact_fixture
from hermetic_config import CONFIG_TEMPLATE

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

# The artifact policy the ported fixture ships keeps the stay margin at 0.05,
# which is inert against the fixture's own +/-1 routing axis. The hermetic run
# has no episode store to read back, so "the second turn saw the first turn's
# arm" has to be visible in the answer itself. A margin wider than the axis
# makes it so: the second turn stays on the first turn's worker even when the
# encoder points the other way, and a fresh episode with the same prompt still
# goes the other way. See test_acceptance.py::previous_arm_is_read_back.
STAY_MARGIN = 4.0

MODAL_KEY = "public-e2e-modal-key"
MODAL_SECRET = "public-e2e-modal-secret"
PROVIDER_KEY = "public-e2e-provider-key"

READY_METRIC = 'llm_rayline_arc_component_ready{component="artifact_head_encoder"}'
STARTUP_TIMEOUT_SECONDS = 60


def free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _library_path() -> str:
    parts = [
        str(REPO_ROOT / binding / "target" / "release")
        for binding in ("candle-binding", "ml-binding", "nlp-binding")
    ]
    return ":".join(parts)


class Stack:
    """One running acceptance stack: encoder, provider, router."""

    def __init__(self, *, max_inflight: int = 0, acquire_timeout: int = 5) -> None:
        self.max_inflight = max_inflight
        self.acquire_timeout = acquire_timeout
        self.encoder_port = free_port()
        self.provider_port = free_port()
        self.api_port = free_port()
        self.metrics_port = free_port()
        self.extproc_port = free_port()
        self.listener_port = free_port()
        self.redis_address = os.getenv("REDIS_ADDR", "").strip()
        self._workdir = TemporaryDirectory(prefix="rayline-arc-acceptance-")
        self._processes: list[subprocess.Popen[bytes]] = []
        self._router_log: Any = None

    # ---- lifecycle -----------------------------------------------------

    def start(self) -> None:
        root = Path(self._workdir.name)
        artifact_dir = root / "artifact"
        # The fixture's default worker binding is OpenRouter, and readiness
        # then insists the configured base URL is openrouter.ai. The
        # openai_compatible binding is the same fixture pointed at a local
        # provider, which is the only way a hermetic run can satisfy the
        # artifact dispatch contract.
        provider_base_url = f"http://127.0.0.1:{self.provider_port}"
        os.environ["RAYLINE_ARC_E2E_DISPATCH_BACKEND"] = "openai_compatible"
        os.environ["RAYLINE_ARC_E2E_WORKER_A_BASE_URL"] = provider_base_url
        os.environ["RAYLINE_ARC_E2E_WORKER_B_BASE_URL"] = provider_base_url
        artifact_fixture.generate(artifact_dir)
        self._widen_stay_margin(artifact_dir / "manifest.json")

        self._spawn_mock("encoder", self.encoder_port)
        self._spawn_mock("provider", self.provider_port)
        self._wait_http(f"http://127.0.0.1:{self.encoder_port}/health")
        self._wait_http(f"http://127.0.0.1:{self.provider_port}/health")

        config_path = root / "config.yaml"
        config_path.write_text(self._render_config(artifact_dir))
        self._spawn_router(root, config_path)
        self._wait_ready()

    def stop(self) -> None:
        for process in reversed(self._processes):
            if process.poll() is None:
                process.terminate()
        for process in reversed(self._processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        if self._router_log is not None:
            self._router_log.close()
        self._workdir.cleanup()

    def __enter__(self) -> Stack:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    # ---- request helpers -----------------------------------------------

    def route(
        self,
        session: str,
        marker: str,
        *,
        route_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        timeout: float = 20,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        """One decision-only consult. The body is an Anthropic Messages request.

        /v1/route decodes as AnthropicMessagesV1, which is what the deployed
        caller sends, so the acceptance body has to be that shape too.
        """
        headers = {"x-rayline-session": session}
        if route_id is not None:
            headers["x-rayline-route-id"] = route_id
        if messages is None:
            messages = [{"role": "user", "content": f"acceptance {marker}"}]
        return self.request(
            self.api_port,
            "/v1/route",
            method="POST",
            body={"model": "claude-sonnet-4", "messages": messages, "max_tokens": 1},
            headers=headers,
            timeout=timeout,
        )

    def request(
        self,
        port: int,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 10,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=payload,
            method=method,
            headers={"content-type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
                return (
                    response.status,
                    json.loads(raw) if raw else {},
                    {key.lower(): value for key, value in response.headers.items()},
                )
        except urllib.error.HTTPError as error:
            raw = error.read()
            parsed: dict[str, Any] = {}
            if raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {"body": raw.decode(errors="replace")}
            return (
                error.code,
                parsed,
                {key.lower(): value for key, value in error.headers.items()},
            )

    def metrics_text(self) -> str:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.metrics_port}/metrics", timeout=5
        ) as response:
            return response.read().decode()

    def metric(self, name: str, **labels: str) -> float:
        text = self.metrics_text()
        if labels:
            rendered = ",".join(
                f'{key}="{value}"' for key, value in sorted(labels.items())
            )
            prefix = f"{name}{{{rendered}}} "
        else:
            prefix = f"{name} "
        for line in text.splitlines():
            if line.startswith(prefix):
                return float(line[len(prefix) :])
        return 0.0

    def encoder_stats(self) -> dict[str, Any]:
        _, body, _ = self.request(self.encoder_port, "/stats")
        return body

    def provider_requests(self) -> list[dict[str, Any]]:
        _, body, _ = self.request(self.provider_port, "/observed")
        return body.get("requests", [])

    def router_log(self) -> str:
        return (Path(self._workdir.name) / "router.log").read_text(errors="replace")

    # ---- internals ------------------------------------------------------

    def _widen_stay_margin(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text())
        manifest["policy"]["previous_worker_stay_margin"] = STAY_MARGIN
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    def _render_config(self, artifact_dir: Path) -> str:
        using_redis = bool(self.redis_address)
        return CONFIG_TEMPLATE.format(
            listener_port=self.listener_port,
            provider_port=self.provider_port,
            encoder_port=self.encoder_port,
            artifact_dir=artifact_dir,
            artifact_revision=artifact_fixture.ARTIFACT_ID,
            max_inflight=self.max_inflight,
            acquire_timeout=self.acquire_timeout,
            episode_backend="redis" if using_redis else "memory",
            development_mode="false" if using_redis else "true",
            redis_address=self.redis_address or "127.0.0.1:6379",
        )

    def _child_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "RAYLINE_ARC_MODAL_KEY": MODAL_KEY,
                "RAYLINE_ARC_MODAL_SECRET": MODAL_SECRET,
                "SYNTHETIC_API_KEY": PROVIDER_KEY,
                "RAYLINE_ARC_REDIS_PASSWORD": os.getenv("REDIS_PASSWORD", ""),
            }
        )
        library_path = _library_path()
        for name in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
            existing = env.get(name, "")
            env[name] = f"{library_path}:{existing}" if existing else library_path
        return env

    def _spawn_mock(self, service: str, port: int) -> None:
        env = self._child_env()
        env["RAYLINE_ARC_E2E_ENCODER_ID"] = "encoder-a"
        self._processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "mock_servers.py"),
                    service,
                    "--port",
                    str(port),
                ],
                cwd=str(HERE),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    def _spawn_router(self, root: Path, config_path: Path) -> None:
        binary = os.getenv("RAYLINE_ARC_ROUTER_BIN", str(REPO_ROOT / "bin" / "router"))
        if not Path(binary).exists():
            raise AssertionError(
                f"router binary not found at {binary}; "
                "build it or set RAYLINE_ARC_ROUTER_BIN"
            )
        self._router_log = (root / "router.log").open("wb")
        self._processes.append(
            subprocess.Popen(
                [
                    binary,
                    "-config",
                    str(config_path),
                    "-port",
                    str(self.extproc_port),
                    "-api-port",
                    str(self.api_port),
                    "-api-bind",
                    "127.0.0.1",
                    "-metrics-port",
                    str(self.metrics_port),
                    "-management-auth-mode",
                    "disabled",
                ],
                cwd=str(REPO_ROOT),
                env=self._child_env(),
                stdout=self._router_log,
                stderr=subprocess.STDOUT,
            )
        )

    def _wait_http(self, url: str, timeout: float = 20) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=1):
                    return
            except (urllib.error.URLError, OSError, urllib.error.HTTPError):
                time.sleep(0.1)
        raise AssertionError(f"{url} never became reachable")

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._processes[-1].poll() is not None:
                raise AssertionError(
                    "router exited during startup:\n" + self._tail_router_log()
                )
            try:
                if f"{READY_METRIC} 1" in self.metrics_text():
                    return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.2)
        raise AssertionError(
            "router never reported the ARC component ready:\n" + self._tail_router_log()
        )

    def _tail_router_log(self) -> str:
        try:
            return "\n".join(self.router_log().splitlines()[-40:])
        except OSError:
            return "(no router log)"
