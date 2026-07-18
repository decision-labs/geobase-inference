"""Model-specific Hugging Face endpoint handlers with lazy imports."""

from typing import Any

__all__ = ["ChangeStarHandler", "ClayHandler"]


def __getattr__(name: str) -> Any:
    if name == "ChangeStarHandler":
        from geobase_inference.models.changestar import ChangeStarHandler

        return ChangeStarHandler
    if name == "ClayHandler":
        from geobase_inference.models.clay import ClayHandler

        return ClayHandler
    raise AttributeError(name)
