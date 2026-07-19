"""Typed imagery source definitions shared by model handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class BaseImageryInput:
    """Discriminator shared by all imagery sources."""

    type: str


@dataclass(frozen=True)
class GeoTiffInput(BaseImageryInput):
    url: str


@dataclass(frozen=True)
class EsriMapParams:
    size: tuple[int, int] | None = None
    interpolation: str | None = None
    time: str | None = None
    band_ids: tuple[int, ...] | None = None
    rendering_rule: Mapping[str, Any] | None = None
    mosaic_rule: Mapping[str, Any] | None = None
    no_data: str | int | float | tuple[int | float, ...] | None = None


@dataclass(frozen=True)
class EsriImageryInput(BaseImageryInput):
    # Examples: https://host/arcgis/rest/services/NAIP/ImageServer
    # or https://host/arcgis/rest/services/Sentinel2/ImageServer.
    url: str
    polygon: Mapping[str, Any]
    map_params: EsriMapParams


@dataclass(frozen=True)
class MapboxMapParams:
    tileset: str
    zoom: int
    tile_size: Literal[256, 512] = 256
    image_format: str = "png"


@dataclass(frozen=True)
class MapboxImageryInput(BaseImageryInput):
    polygon: Mapping[str, Any]
    map_params: MapboxMapParams


ImageryInput = GeoTiffInput | EsriImageryInput | MapboxImageryInput
