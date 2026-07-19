"""Parsing and validation for supported imagery inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shapely.geometry import mapping, shape

from geobase_inference.core import RequestValidationError, require_http_url
from geobase_inference.geo.input_types import (
    EsriImageryInput,
    EsriMapParams,
    GeoTiffInput,
    ImageryInput,
    MapboxImageryInput,
    MapboxMapParams,
)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    """Validate and copy a JSON object field."""
    if not isinstance(value, Mapping):
        raise RequestValidationError(f"{field} must be a JSON object")
    return dict(value)


def _integer(value: Any, field: str) -> int:
    """Convert a request field to an integer with a useful validation error."""
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise RequestValidationError(f"{field} must be an integer") from error


def _polygon(value: Any) -> dict[str, Any]:
    """Validate an EPSG:4326 GeoJSON Polygon or MultiPolygon."""
    geometry = _mapping(value, "imagery.polygon")
    if geometry.get("type") == "Feature":
        geometry = _mapping(geometry.get("geometry"), "imagery.polygon.geometry")
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise RequestValidationError(
            "imagery.polygon must be a GeoJSON Polygon or MultiPolygon"
        )
    try:
        parsed = shape(geometry)
    except Exception as error:
        raise RequestValidationError(
            "imagery.polygon contains invalid coordinates"
        ) from error
    if parsed.is_empty or not parsed.is_valid:
        raise RequestValidationError("imagery.polygon must be non-empty and valid")
    min_lon, min_lat, max_lon, max_lat = parsed.bounds
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise RequestValidationError(
            "imagery.polygon must use EPSG:4326 longitude/latitude"
        )
    return mapping(parsed)


def _esri_map_params(value: Any) -> EsriMapParams:
    """Parse and validate provider-specific ESRI export parameters."""
    data = _mapping(value or {}, "imagery.mapParams")
    size_raw = data.get("size")
    size = None
    if size_raw is not None:
        if (
            not isinstance(size_raw, (list, tuple))
            or len(size_raw) != 2
            or isinstance(size_raw, (str, bytes))
        ):
            raise RequestValidationError(
                "imagery.mapParams.size must be [width, height]"
            )
        size = (
            _integer(size_raw[0], "imagery.mapParams.size[0]"),
            _integer(size_raw[1], "imagery.mapParams.size[1]"),
        )
        if not all(1 <= dimension <= 8192 for dimension in size):
            raise RequestValidationError(
                "ESRI output dimensions must be between 1 and 8192"
            )
    band_ids_raw = data.get("bandIds")
    band_ids = None
    if band_ids_raw is not None:
        if not isinstance(band_ids_raw, (list, tuple)):
            raise RequestValidationError(
                "imagery.mapParams.bandIds must be an array of integers"
            )
        band_ids = tuple(
            _integer(value, "imagery.mapParams.bandIds") for value in band_ids_raw
        )
    rendering_rule = data.get("renderingRule")
    if rendering_rule is not None:
        rendering_rule = _mapping(rendering_rule, "imagery.mapParams.renderingRule")
    mosaic_rule = data.get("mosaicRule")
    if mosaic_rule is not None:
        mosaic_rule = _mapping(mosaic_rule, "imagery.mapParams.mosaicRule")
    no_data = data.get("noData")
    if isinstance(no_data, list):
        no_data = tuple(no_data)
    return EsriMapParams(
        size=size,
        interpolation=data.get("interpolation"),
        time=str(data["time"]) if data.get("time") is not None else None,
        band_ids=band_ids,
        rendering_rule=rendering_rule,
        mosaic_rule=mosaic_rule,
        no_data=no_data,
    )


def _mapbox_map_params(value: Any) -> MapboxMapParams:
    """Parse and validate provider-specific Mapbox raster tile parameters."""
    data = _mapping(value, "imagery.mapParams")
    tileset = data.get("tileset", "mapbox.satellite")
    if not isinstance(tileset, str) or not tileset.strip():
        raise RequestValidationError(
            "imagery.mapParams.tileset must be a non-empty string"
        )
    if "zoom" not in data:
        raise RequestValidationError(
            "imagery.mapParams.zoom is required for Mapbox imagery"
        )
    zoom = _integer(data["zoom"], "imagery.mapParams.zoom")
    if not 0 <= zoom <= 22:
        raise RequestValidationError("imagery.mapParams.zoom must be between 0 and 22")
    tile_size = _integer(data.get("tileSize", 256), "imagery.mapParams.tileSize")
    if tile_size not in {256, 512}:
        raise RequestValidationError("imagery.mapParams.tileSize must be 256 or 512")
    image_format = str(data.get("imageFormat", "png")).lower()
    if image_format not in {"png", "png32", "jpg70", "jpg80", "jpg90"}:
        raise RequestValidationError(
            "imagery.mapParams.imageFormat must be png, png32, jpg70, jpg80, or jpg90"
        )
    return MapboxMapParams(
        tileset=tileset.strip(),
        zoom=zoom,
        tile_size=tile_size,
        image_format=image_format,
    )


def parse_imagery_input(value: Any) -> ImageryInput:
    """Parse and validate a backward-compatible URL or typed provider object."""
    if isinstance(value, str):
        return GeoTiffInput(type="geotiff", url=require_http_url(value))
    data = _mapping(value, "imagery")
    source_type = str(data.get("type", "geotiff")).strip().lower()
    if source_type in {"geotiff", "tif", "tiff"}:
        return GeoTiffInput(type="geotiff", url=require_http_url(data, "imagery.url"))
    if source_type == "esri":
        return EsriImageryInput(
            type="esri",
            url=require_http_url(data.get("url"), "imagery.url"),
            polygon=_polygon(data.get("polygon")),
            map_params=_esri_map_params(data.get("mapParams")),
        )
    if source_type == "mapbox":
        return MapboxImageryInput(
            type="mapbox",
            polygon=_polygon(data.get("polygon")),
            map_params=_mapbox_map_params(data.get("mapParams")),
        )
    raise RequestValidationError(
        "imagery.type must be one of: geotiff, esri, mapbox"
    )
