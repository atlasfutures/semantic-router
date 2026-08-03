"""Semantic validation for the Rayline ARC config family."""

import os
import re

from cli.rayline_arc_config import (
    RAYLINE_ARC_ENCODER_MODEL,
    RAYLINE_ARC_ENCODER_MODEL_REVISION,
    RAYLINE_ARC_SERIALIZER_VERSION,
)
from cli.validation_error import ValidationError
from cli.validator_rayline_arc_episode import (
    _valid_host_port as _episode_valid_host_port,
)
from cli.validator_rayline_arc_episode import (
    validate_episode_contract,
)
from cli.validator_rayline_arc_membership import (
    validate_encoder_membership,
    validate_episode_membership,
)

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MUTABLE_PINS = {"latest", "main", "master", "head"}
_MIN_MODEL_REFS = 2
_MAX_BOUNDED_STRING_BYTES = 512


def _valid_host_port(value: str | None) -> bool:
    """Preserve the validator module's historical helper import seam."""

    return _episode_valid_host_port(value)


def validate_rayline_arc_decisions(config) -> list[ValidationError]:
    """Validate cross-field ARC contracts that Pydantic cannot express."""

    errors: list[ValidationError] = []
    for decision in config.decisions:
        if decision.algorithm is None or decision.algorithm.type != "rayline_arc":
            continue
        errors.extend(_validate_rayline_arc_decision(decision))
        errors.extend(_validate_rayline_arc_auto_aliases(config, decision))
        errors.extend(_validate_rayline_arc_replay(config, decision))
    return errors


def _effective_auto_model_names(config) -> set[str]:
    """Mirror the router's EffectiveAutoModelNames resolution exactly.

    Go normalizes by trimming and dropping empties, and falls back to the
    defaults when normalization leaves nothing, so a whitespace-only entry
    must not silently produce an empty alias set here.
    """
    router = (getattr(config, "global_", None) or {}).get("router") or {}
    explicit = {
        str(name).strip()
        for name in (router.get("auto_model_names") or [])
        if str(name).strip()
    }
    if explicit:
        return explicit
    configured = str(router.get("auto_model_name") or "").strip()
    return {"vllm-sr/auto", "auto", configured or "MoM"}


def _validate_rayline_arc_auto_aliases(config, decision) -> list[ValidationError]:
    auto_names = _effective_auto_model_names(config)
    return [
        ValidationError(
            f"decision '{decision.name}' algorithm.type=rayline_arc modelRef "
            f"'{model.model}' collides with an auto-routing alias, which would "
            "bypass artifact-owned dispatch",
            field=f"decisions.{decision.name}.modelRefs",
        )
        for model in decision.modelRefs
        if str(model.model).strip() in auto_names
    ]


def _validate_rayline_arc_replay(config, decision) -> list[ValidationError]:
    """Router Replay defaults to enabled, so ARC decisions must disable it."""
    services = (getattr(config, "global_", None) or {}).get("services") or {}
    if "router_replay" not in services:
        # Absent: canonical defaults enable replay.
        globally_enabled = True
    else:
        replay = services["router_replay"]
        # Explicit YAML null zeroes the Go struct, leaving Enabled false.
        globally_enabled = (
            False if replay is None else bool(replay.get("enabled", True))
        )

    for plugin in decision.plugins or []:
        if getattr(plugin, "type", None) != "router_replay":
            continue
        configuration = getattr(plugin, "configuration", None) or {}
        if not bool(configuration.get("enabled", globally_enabled)):
            return []
        globally_enabled = True
        break

    if not globally_enabled:
        return []
    return [
        ValidationError(
            f"decision '{decision.name}' algorithm.type=rayline_arc requires "
            "router_replay disabled for this decision; episode requests must "
            "not be persisted",
            field=f"decisions.{decision.name}.plugins",
        )
    ]


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
    errors.extend(_validate_episode(decision.name, arc.episode, arc.encoder))

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
    errors.extend(validate_encoder_membership(prefix, encoder))
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
    errors.extend(_validate_encoder_capabilities(prefix, encoder))
    has_modal_key = bool(encoder.modal_key_env)
    has_modal_secret = bool(encoder.modal_secret_env)
    if has_modal_key != has_modal_secret:
        errors.append(
            ValidationError(
                "modal_key_env and modal_secret_env must be configured together",
                field=f"{prefix}.modal_key_env",
            )
        )
    elif has_modal_key and (
        not _ENV_NAME.fullmatch(encoder.modal_key_env)
        or not _ENV_NAME.fullmatch(encoder.modal_secret_env)
    ):
        errors.append(
            ValidationError(
                "Modal proxy credential fields must name valid environment variables",
                field=f"{prefix}.modal_key_env",
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


def _validate_encoder_capabilities(prefix, encoder) -> list[ValidationError]:
    errors: list[ValidationError] = []
    capabilities = encoder.required_pooling_capabilities
    if len(capabilities) != len(set(capabilities)):
        errors.append(
            ValidationError(
                "required_pooling_capabilities cannot contain duplicates",
                field=f"{prefix}.required_pooling_capabilities",
            )
        )
    required_rung_capability = {
        "A": "all_plugin_mean",
        "B": "chunked_causal_mean",
    }[encoder.serving_rung]
    if required_rung_capability not in capabilities:
        errors.append(
            ValidationError(
                f"serving_rung {encoder.serving_rung!r} requires "
                f"{required_rung_capability!r}",
                field=f"{prefix}.required_pooling_capabilities",
            )
        )
    if (
        "resumable_causal_mean" in capabilities
        and "chunked_causal_mean" not in capabilities
    ):
        errors.append(
            ValidationError(
                "resumable_causal_mean requires chunked_causal_mean",
                field=f"{prefix}.required_pooling_capabilities",
            )
        )
    return errors


def _validate_episode(name, episode, encoder) -> list[ValidationError]:
    prefix = f"decisions.{name}.algorithm.rayline_arc.episode"
    return validate_episode_contract(prefix, episode) + validate_episode_membership(
        prefix, episode, encoder
    )


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


def _byte_length(value: str) -> int:
    return len(value.encode())
