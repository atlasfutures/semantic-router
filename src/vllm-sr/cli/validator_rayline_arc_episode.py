"""Base Rayline ARC episode storage and HTTP-header validation."""

import re

from cli.validation_error import ValidationError

_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9a-z-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_EPISODE_KEY_PREFIX_BYTES = 128
_MAX_NETWORK_PORT = 65_535


def validate_episode_contract(prefix, episode) -> list[ValidationError]:
    """Validate the episode state contract independent of encoder mode."""

    return _validate_headers(prefix, episode) + _validate_storage(prefix, episode)


def _validate_headers(prefix, episode) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if not _HEADER_NAME.fullmatch(episode.id_header):
        errors.append(
            ValidationError(
                "id_header must be a nonempty lowercase HTTP field name",
                field=f"{prefix}.id_header",
            )
        )
    if episode.close_header is not None:
        if not _HEADER_NAME.fullmatch(episode.close_header):
            errors.append(
                ValidationError(
                    "close_header must be a lowercase HTTP field name",
                    field=f"{prefix}.close_header",
                )
            )
        elif episode.close_header == episode.id_header:
            errors.append(
                ValidationError(
                    "close_header must differ from id_header",
                    field=f"{prefix}.close_header",
                )
            )
    return errors


def _validate_storage(prefix, episode) -> list[ValidationError]:
    errors: list[ValidationError] = []
    if (
        not episode.key_prefix.strip()
        or _byte_length(episode.key_prefix) > _MAX_EPISODE_KEY_PREFIX_BYTES
    ):
        errors.append(
            ValidationError(
                "key_prefix must contain between 1 and 128 bytes",
                field=f"{prefix}.key_prefix",
            )
        )
    if episode.idle_ttl_seconds < episode.lease_ttl_seconds:
        errors.append(
            ValidationError(
                "idle_ttl_seconds cannot be less than lease_ttl_seconds",
                field=f"{prefix}.idle_ttl_seconds",
            )
        )
    if episode.backend == "memory":
        if not episode.development_mode:
            errors.append(
                ValidationError(
                    "backend=memory requires development_mode=true",
                    field=f"{prefix}.development_mode",
                )
            )
        if episode.max_in_memory_episodes <= 0:
            errors.append(
                ValidationError(
                    "max_in_memory_episodes must be positive for backend=memory",
                    field=f"{prefix}.max_in_memory_episodes",
                )
            )
    if episode.backend == "redis":
        if episode.redis is None or not _valid_host_port(episode.redis.address):
            errors.append(
                ValidationError(
                    "redis.address must be a host:port pair",
                    field=f"{prefix}.redis.address",
                )
            )
        elif episode.redis.password_env and not _ENV_NAME.fullmatch(
            episode.redis.password_env
        ):
            errors.append(
                ValidationError(
                    "redis.password_env must be a valid environment variable name",
                    field=f"{prefix}.redis.password_env",
                )
            )
    return errors


def _valid_host_port(value: str | None) -> bool:
    if not value or ":" not in value:
        return False
    host, port = value.rsplit(":", 1)
    if not host or not port.isascii() or not port.isdigit():
        return False
    if not 0 < int(port) <= _MAX_NETWORK_PORT:
        return False
    if host.startswith("[") and host.endswith("]"):
        return len(host) > len("[]")
    # Bare hosts must not contain ":"; IPv6 literals require brackets so the
    # Go and Python validators accept exactly the same addresses.
    return ":" not in host


def _byte_length(value: str) -> int:
    return len(value.encode())
