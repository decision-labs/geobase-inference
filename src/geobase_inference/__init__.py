"""Reusable geospatial inference handlers."""

from typing import Any

__all__ = ["ChangeStarHandler", "ClayHandler"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in __all__:
        from geobase_inference import models

        return getattr(models, name)
    raise AttributeError(name)
