"""Shared raster processing primitives."""

from geobase_inference.geo.input_types import (
    BaseImageryInput,
    EsriImageryInput,
    EsriMapParams,
    GeoTiffInput,
    ImageryInput,
    MapboxImageryInput,
    MapboxMapParams,
)
from geobase_inference.geo.input_validation import parse_imagery_input
from geobase_inference.geo.raster import (
    feather_weight,
    mask_tiff_bytes,
    mask_to_geojson,
    tile_to_rgb_uint8,
)
from geobase_inference.geo.sources import create_temporary_geotiff

__all__ = [
    "BaseImageryInput",
    "EsriImageryInput",
    "EsriMapParams",
    "GeoTiffInput",
    "ImageryInput",
    "MapboxImageryInput",
    "MapboxMapParams",
    "feather_weight",
    "create_temporary_geotiff",
    "mask_to_geojson",
    "mask_tiff_bytes",
    "parse_imagery_input",
    "tile_to_rgb_uint8",
]
