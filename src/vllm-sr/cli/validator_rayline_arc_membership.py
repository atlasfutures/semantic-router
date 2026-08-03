"""Replicated and dynamic Rayline ARC encoder membership validation."""

import re
from urllib.parse import urlparse

from cli.validation_error import ValidationError

_MIN_ENCODER_REPLICAS = 2
_MAX_ENCODER_REPLICAS = 8
_MAX_ENCODER_REPLICA_ID_BYTES = 64
_MAX_BOUNDED_STRING_BYTES = 512
_MIN_UNAVAILABLE_STATUS = 400
_MAX_UNAVAILABLE_STATUS = 599
_RAYLINE_ARC_ENCODER_FAILOVER_V1 = "rayline.arc.encoder-failover.v1"
_RAYLINE_ARC_ENCODER_MEMBERSHIP_V1 = "rayline.arc.encoder-membership.v1"


def validate_encoder_membership(prefix, encoder) -> list[ValidationError]:
    """Validate one encoder transport mode and replicated failover contract."""

    errors: list[ValidationError] = []
    base_url = (encoder.base_url or "").strip()
    replicas = encoder.replicas or []
    membership = encoder.membership
    modes = sum(bool(value) for value in (base_url, replicas, membership))
    if modes != 1:
        return [
            ValidationError(
                "exactly one of base_url, replicas, or membership must be configured",
                field=prefix,
            )
        ]
    if base_url:
        if not _valid_encoder_url(base_url):
            errors.append(
                ValidationError(
                    "base_url must be an absolute HTTP(S) URL without credentials, "
                    "a query, or a fragment",
                    field=f"{prefix}.base_url",
                )
            )
        if encoder.failover is not None:
            errors.append(
                ValidationError(
                    "failover requires replicas or membership",
                    field=f"{prefix}.failover",
                )
            )
        return errors

    if (
        encoder.serving_rung != "B"
        or "resumable_causal_mean" not in encoder.required_pooling_capabilities
    ):
        errors.append(
            ValidationError(
                "replicas or membership require retained serving rung B with "
                "resumable capability",
                field=f"{prefix}.serving_rung",
            )
        )
    if encoder.max_retries != 0:
        errors.append(
            ValidationError(
                "replicas or membership require max_retries=0",
                field=f"{prefix}.max_retries",
            )
        )
    errors.extend(_validate_encoder_failover(prefix, encoder.failover))
    if replicas:
        errors.extend(_validate_encoder_replicas(prefix, replicas))
        return errors

    if membership.schema_version != _RAYLINE_ARC_ENCODER_MEMBERSHIP_V1:
        errors.append(
            ValidationError(
                f"membership.schema_version must be "
                f"{_RAYLINE_ARC_ENCODER_MEMBERSHIP_V1!r}",
                field=f"{prefix}.membership.schema_version",
            )
        )
    if membership.source != "redis":
        errors.append(
            ValidationError(
                "membership.source must be 'redis'",
                field=f"{prefix}.membership.source",
            )
        )
    return errors


def validate_episode_membership(prefix, episode, encoder) -> list[ValidationError]:
    """Validate lifecycle requirements that only replicated encoders need."""

    errors: list[ValidationError] = []
    replica_membership = bool(encoder.replicas) or encoder.membership is not None
    if replica_membership and not episode.close_header:
        errors.append(
            ValidationError(
                "close_header is required with encoder replicas or membership",
                field=f"{prefix}.close_header",
            )
        )
    if not replica_membership and episode.close_header:
        errors.append(
            ValidationError(
                "close_header requires encoder replicas or membership",
                field=f"{prefix}.close_header",
            )
        )
    if encoder.membership is not None and episode.backend != "redis":
        errors.append(
            ValidationError(
                "redis backend is required with dynamic encoder membership",
                field=f"{prefix}.backend",
            )
        )
    return errors


def _validate_encoder_failover(prefix, failover) -> list[ValidationError]:
    if failover is None:
        return [
            ValidationError(
                "failover is required with replicas or membership",
                field=f"{prefix}.failover",
            )
        ]
    errors: list[ValidationError] = []
    if failover.schema_version != _RAYLINE_ARC_ENCODER_FAILOVER_V1:
        errors.append(
            ValidationError(
                f"failover.schema_version must be "
                f"{_RAYLINE_ARC_ENCODER_FAILOVER_V1!r}",
                field=f"{prefix}.failover.schema_version",
            )
        )
    if failover.max_remaps != 1:
        errors.append(
            ValidationError(
                "failover.max_remaps must be 1",
                field=f"{prefix}.failover.max_remaps",
            )
        )
    invalid_statuses = len(failover.unavailable_status_codes) != len(
        set(failover.unavailable_status_codes)
    ) or any(
        status < _MIN_UNAVAILABLE_STATUS or status > _MAX_UNAVAILABLE_STATUS
        for status in failover.unavailable_status_codes
    )
    if invalid_statuses:
        errors.append(
            ValidationError(
                "failover unavailable statuses must be unique values between "
                "400 and 599",
                field=f"{prefix}.failover.unavailable_status_codes",
            )
        )
    return errors


def _validate_encoder_replicas(prefix, replicas) -> list[ValidationError]:
    if not _MIN_ENCODER_REPLICAS <= len(replicas) <= _MAX_ENCODER_REPLICAS:
        return [
            ValidationError(
                "replicas must contain between 2 and 8 entries",
                field=f"{prefix}.replicas",
            )
        ]
    errors: list[ValidationError] = []
    ids = [replica.id for replica in replicas]
    urls = [replica.base_url.rstrip("/") for replica in replicas]
    if len(ids) != len(set(ids)):
        errors.append(
            ValidationError("replica id is duplicated", field=f"{prefix}.replicas")
        )
    if len(urls) != len(set(urls)):
        errors.append(
            ValidationError(
                "replica base_url is duplicated",
                field=f"{prefix}.replicas",
            )
        )
    if not any(replica.state == "active" for replica in replicas):
        errors.append(
            ValidationError(
                "replicas require at least one active member",
                field=f"{prefix}.replicas",
            )
        )
    for replica in replicas:
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", replica.id)
            or _byte_length(replica.id) > _MAX_ENCODER_REPLICA_ID_BYTES
        ):
            errors.append(
                ValidationError(
                    "replica id must be a bounded stable identifier",
                    field=f"{prefix}.replicas",
                )
            )
        if not _valid_encoder_url(replica.base_url):
            errors.append(
                ValidationError(
                    "replica base_url must be an absolute HTTP(S) URL without "
                    "credentials, a query, or a fragment",
                    field=f"{prefix}.replicas",
                )
            )
    return errors


def _valid_encoder_url(value: str) -> bool:
    parsed = urlparse(value)
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and _byte_length(value) <= _MAX_BOUNDED_STRING_BYTES
    )


def _byte_length(value: str) -> int:
    return len(value.encode())
