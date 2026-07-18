"""Shared endpoint request validation, logging, and response helpers."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

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


def require_mapping(data: Any) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise RequestValidationError("Request body must be a JSON object")
    return dict(data)


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


def require_http_url(value: Any, field: str = "inputs") -> str:
    if isinstance(value, Mapping):
        value = value.get("url")
    if not isinstance(value, str) or not value.strip():
        raise RequestValidationError(f"{field} must be a non-empty URL string")
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        raise RequestValidationError(f"{field} must use an http:// or https:// URL")
    return value


def download_to_temp(
    url: str,
    *,
    suffix: str = "",
    connect_timeout: float = 15,
    read_timeout: float = 120,
    logger: logging.Logger | None = None,
) -> str:
    """Stream an HTTP resource to an isolated temporary file."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    started = time.perf_counter()
    try:
        with requests.get(
            url,
            stream=True,
            timeout=(connect_timeout, read_timeout),
        ) as response:
            response.raise_for_status()
            with open(path, "wb") as output:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        output.write(chunk)
        if logger:
            logger.info(
                "Downloaded %.1f MB in %.1fs",
                os.path.getsize(path) / (1 << 20),
                time.perf_counter() - started,
            )
        return path
    except Exception:
        if os.path.exists(path):
            os.unlink(path)
        raise
