"""Shared endpoint request validation, logging, and response helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geobase_inference.geo.input_types import ImageryInput

LOGGER_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging() -> None:
    """Configure useful container logs without replacing an existing setup."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)


class RequestValidationError(ValueError):
    """Raised when an endpoint request has invalid user input."""


@dataclass(frozen=True)
class Artifact:
    """A binary output artifact ready for persistence."""

    name: str
    body: bytes
    media_type: str


@dataclass(frozen=True)
class BaseModelRequest:
    """Fields shared by every model request."""

    imagery: ImageryInput


def require_mapping(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise RequestValidationError("Request body must be a JSON object")
    return dict(data)


def normalize_handler_input(raw: Any) -> dict[str, Any]:
    """Unwrap the Hugging Face ``inputs`` envelope while supporting direct calls."""
    outer = require_mapping(raw)
    if "inputs" not in outer:
        return outer

    inputs = outer["inputs"]
    if isinstance(inputs, str):
        data: dict[str, Any] = {"imagery": inputs}
    elif isinstance(inputs, Mapping):
        data = dict(inputs)
    else:
        raise RequestValidationError(
            "inputs must be a JSON object or an imagery URL string"
        )
    return data


def request_value(
    data: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Any:
    """Read a top-level field, falling back to the conventional parameters object."""
    parameters = data.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise RequestValidationError("parameters must be a JSON object")
    return data.get(name, parameters.get(name, default))


def require_http_url(value: Any, field: str = "imagery") -> str:
    if isinstance(value, Mapping):
        value = value.get("url")
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{field} must be a non-empty URL string")
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        raise RequestValidationError(f"{field} must use an http:// or https:// URL")
    return value
