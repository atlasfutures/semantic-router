"""Semantic validation for the Rayline ARC config family."""

import os
import re
from urllib.parse import urlparse

from cli.rayline_arc_config import (
    RAYLINE_ARC_ENCODER_MODEL,
    RAYLINE_ARC_ENCODER_MODEL_REVISION,
    RAYLINE_ARC_SERIALIZER_VERSION,
)
from cli.validation_error import ValidationError

_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9a-z-]+$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MUTABLE_PINS = {"latest", "main", "master", "head"}
_MIN_MODEL_REFS = 2
_MAX_BOUNDED_STRING_BYTES = 512
_MAX_EPISODE_KEY_PREFIX_BYTES = 128
_MAX_NETWORK_PORT = 65_535


def validate_rayline_arc_decisions(config) -> list[ValidationError]:
    """Validate cross-field ARC contracts that Pydantic cannot express."""

    errors: list[ValidationError] = []
    for decision in config.decisions:
        if decision.algorithm is None or decision.algorithm.type != "rayline_arc":
            continue
        errors.extend(_validate_rayline_arc_decision(decision))
    return errors


def _validate_rayline_arc_decision(decision) -> list[ValidationError]:
    errors: list[ValidationError] = []
    field = f"decisions.{decision.name}.algorithm"
    algorithm = decision.algorithm
    arc = algorithm.rayline_arc

    if algorithm.on_error != "fail_closed":
        errors.append(
            ValidationError(
                f"decision '{decision.name}' algorithm.type=rayline_arc requires "
                "algorithm.on_error=fail_closed",
                field=f"{field}.on_error",
            )
        )
    if arc is None:
        errors.append(
            ValidationError(
                f"decision '{decision.name}' requires algorithm.rayline_arc when "
                "algorithm.type=rayline_arc",
                field=f"{field}.rayline_arc",
            )
        )
        return errors

    errors.extend(_validate_artifact(decision.name, arc))
    errors.extend(_validate_encoder(decision.name, arc.encoder))
    errors.extend(_validate_episode(decision.name, arc.episode))

    adaptations = decision.adaptations
    if adaptations is None or adaptations.mode != "bypass":
        errors.append(
            ValidationError(
                f"decision '{decision.name}' algorithm.type=rayline_arc requires "
                "adaptations.mode=bypass so Router Learning cannot override the "
                "artifact decision",
                field=f"decisions.{decision.name}.adaptations.mode",
            )
        )

    models = [model.model for model in decision.modelRefs]
    if len(models) < _MIN_MODEL_REFS:
        errors.append(
            ValidationError(
                f"decision '{decision.name}' algorithm.type=rayline_arc requires "
                "at least two modelRefs",
                field=f"decisions.{decision.name}.modelRefs",
            )
        )
    elif len(models) != len(set(models)):
        errors.append(
            ValidationError(
                f"decision '{decision.name}' algorithm.type=rayline_arc requires "
                "unique modelRefs in artifact order",
                field=f"decisions.{decision.name}.modelRefs",
            )
        )
    return errors


def _validate_artifact(name, arc) -> list[ValidationError]:
    errors: list[ValidationError] = []
    prefix = f"decisions.{name}.algorithm.rayline_arc"
    if (
        not os.path.isabs(arc.artifact_dir)
        or _byte_length(arc.artifact_dir) > _MAX_BOUNDED_STRING_BYTES
    ):
        errors.append(
            ValidationError(
                "artifact_dir must be an absolute path of at most 512 bytes",
                field=f"{prefix}.artifact_dir",
            )
        )
    pin_error = _immutable_pin_error("artifact_revision", arc.artifact_revision)
    if pin_error:
        errors.append(ValidationError(pin_error, field=f"{prefix}.artifact_revision"))
    return errors


def _validate_encoder(name, encoder) -> list[ValidationError]:
    errors: list[ValidationError] = []
    prefix = f"decisions.{name}.algorithm.rayline_arc.encoder"
    parsed = urlparse(encoder.base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or _byte_length(encoder.base_url) > _MAX_BOUNDED_STRING_BYTES
    ):
        errors.append(
            ValidationError(
                "base_url must be an absolute HTTP(S) URL without credentials, "
                "a query, or a fragment",
                field=f"{prefix}.base_url",
            )
        )
    if encoder.model != RAYLINE_ARC_ENCODER_MODEL:
        errors.append(
            ValidationError(
                f"model must be {RAYLINE_ARC_ENCODER_MODEL!r}",
                field=f"{prefix}.model",
            )
        )
    if encoder.model_revision != RAYLINE_ARC_ENCODER_MODEL_REVISION:
        errors.append(
            ValidationError(
                f"model_revision must be {RAYLINE_ARC_ENCODER_MODEL_REVISION!r}",
                field=f"{prefix}.model_revision",
            )
        )
    for field_name, value in (
        ("expected_build_id", encoder.expected_build_id),
        ("expected_io_plugin_version", encoder.expected_io_plugin_version),
    ):
        pin_error = _immutable_pin_error(field_name, value)
        if pin_error:
            errors.append(ValidationError(pin_error, field=f"{prefix}.{field_name}"))
    if encoder.serializer_version != RAYLINE_ARC_SERIALIZER_VERSION:
        errors.append(
            ValidationError(
                f"serializer_version must be {RAYLINE_ARC_SERIALIZER_VERSION!r}",
                field=f"{prefix}.serializer_version",
            )
        )
    capabilities = encoder.required_pooling_capabilities
    if len(capabilities) != len(set(capabilities)):
        errors.append(
            ValidationError(
                "required_pooling_capabilities cannot contain duplicates",
                field=f"{prefix}.required_pooling_capabilities",
            )
        )
    if encoder.connect_timeout_seconds > encoder.total_timeout_seconds:
        errors.append(
            ValidationError(
                "connect_timeout_seconds cannot exceed total_timeout_seconds",
                field=f"{prefix}.connect_timeout_seconds",
            )
        )
    return errors


def _validate_episode(name, episode) -> list[ValidationError]:
    errors: list[ValidationError] = []
    prefix = f"decisions.{name}.algorithm.rayline_arc.episode"
    if not _HEADER_NAME.fullmatch(episode.id_header):
        errors.append(
            ValidationError(
                "id_header must be a nonempty lowercase HTTP field name",
                field=f"{prefix}.id_header",
            )
        )
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


def _immutable_pin_error(name: str, value: str) -> str | None:
    pin = value.strip()
    if not pin:
        return f"{name} cannot be empty"
    if _byte_length(pin) > _MAX_BOUNDED_STRING_BYTES or any(
        character.isspace() for character in pin
    ):
        return f"{name} must be a bounded immutable identifier without whitespace"
    if pin.lower() in _MUTABLE_PINS:
        return f"{name} cannot use mutable value {pin!r}"
    return None


def _valid_host_port(value: str | None) -> bool:
    if not value or ":" not in value:
        return False
    host, port = value.rsplit(":", 1)
    return bool(host) and port.isdigit() and 0 < int(port) <= _MAX_NETWORK_PORT


def _byte_length(value: str) -> int:
    return len(value.encode())
