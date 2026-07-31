# SPDX-License-Identifier: Apache-2.0

"""HTTP continuation handling for bounded Modal Web Function canaries."""

from __future__ import annotations

import http.client
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlparse

HTTP_SEE_OTHER = 303
MAX_RESULT_REDIRECTS = 2


def connection_for_url(url: str, timeout_seconds: float) -> tuple[Any, str]:
    """Create one bounded HTTP(S) connection and normalized path prefix."""
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError("URL omitted hostname")
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    return (
        connection_type(parsed.hostname, parsed.port, timeout=timeout_seconds),
        parsed.path.rstrip("/"),
    )


def _origin(url: str) -> tuple[str, str | None, int | None]:
    parsed = urlparse(url)
    return parsed.scheme, parsed.hostname, parsed.port


def request_following_result_redirects(
    *,
    connection_factory: Callable[[str, float], tuple[Any, str]],
    method: str,
    url: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[Any, Any]:
    """Follow only bounded same-origin Modal result redirects."""
    expected_origin = _origin(url)
    current_url = url
    current_method = method
    current_body = body
    for redirect_count in range(MAX_RESULT_REDIRECTS + 1):
        connection, path = connection_factory(current_url, timeout_seconds)
        parsed = urlparse(current_url)
        request_target = path or "/"
        if parsed.query:
            request_target = f"{request_target}?{parsed.query}"
        connection.request(
            current_method,
            request_target,
            body=current_body,
            headers=headers,
        )
        response = connection.getresponse()
        if response.status != HTTP_SEE_OTHER:
            return connection, response
        location = response.getheader("location", "")
        response.read()
        connection.close()
        if redirect_count == MAX_RESULT_REDIRECTS:
            raise RuntimeError("Modal result redirect limit exceeded")
        if not location:
            raise RuntimeError("Modal result redirect omitted Location")
        redirected_url = urljoin(current_url, location)
        if _origin(redirected_url) != expected_origin:
            raise RuntimeError("refusing cross-origin Modal result redirect")
        current_url = redirected_url
        current_method = "GET"
        current_body = None
    raise AssertionError("unreachable redirect loop")
