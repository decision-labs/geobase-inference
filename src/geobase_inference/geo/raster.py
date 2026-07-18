"""GeoTIFF conversion, stitching, vectorization, and serialization."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping
from shapely.geometry import shape as shapely_shape


def tile_to_rgb_uint8(tile: np.ndarray) -> np.ndarray:
    """Convert a channels-first raster tile to RGB uint8."""
    if tile.ndim != 3:
        raise ValueError(f"Expected a C,H,W tile, got shape {tile.shape}")
    tile = np.transpose(tile, (1, 2, 0))
    if tile.shape[2] >= 3:
        tile = tile[:, :, :3]
    elif tile.shape[2] == 1:
        tile = np.repeat(tile, 3, axis=2)
    else:
        raise ValueError("Two-band rasters are unsupported")

    if tile.dtype != np.uint8:
        finite = tile[np.isfinite(tile)]
        if finite.size == 0:
            return np.zeros(tile.shape, dtype=np.uint8)
        tmin = float(finite.min())
        tmax = float(finite.max())
        if tmax > tmin:
            tile = np.nan_to_num((tile - tmin) / (tmax - tmin) * 255)
            tile = np.clip(tile, 0, 255).astype(np.uint8)
        else:
            tile = np.zeros(tile.shape, dtype=np.uint8)
    return tile


def feather_weight(size: int, overlap: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    ramp = np.ones(size, dtype=np.float32)
    if overlap:
        edge = np.linspace(0, 1, overlap, dtype=np.float32)
        ramp[:overlap] = edge
        ramp[-overlap:] = edge[::-1]
    return np.outer(ramp, ramp)


def mask_to_geojson(
    mask: np.ndarray,
    transform: Any,
    crs: Any,
    *,
    class_name: str = "building",
    output_crs: Any = "EPSG:4326",
) -> dict[str, Any]:
    if crs is None:
        raise ValueError("Cannot create reprojected GeoJSON from a raster without a CRS")
    output_crs_name = rasterio.crs.CRS.from_user_input(output_crs).to_string()
    features = []
    for geometry, value in rio_shapes(
        mask.astype(np.uint8),
        mask=mask.astype(bool),
        transform=transform,
    ):
        if value == 1:
            geometry = transform_geom(crs, output_crs_name, geometry)
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(shapely_shape(geometry)),
                    "properties": {"class": class_name},
                }
            )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": output_crs_name}},
        "features": features,
    }


def mask_tiff_bytes(mask: np.ndarray, profile: dict[str, Any]) -> bytes:
    """Serialize a one-band mask without incompatible source RGB metadata."""
    output_profile = profile.copy()
    for key in (
        "photometric",
        "jpeg_quality",
        "jpegtables",
        "compress",
        "interleave",
        "nbits",
    ):
        output_profile.pop(key, None)
    output_profile.update(
        count=1,
        dtype="uint8",
        photometric="MINISBLACK",
        compress="deflate",
    )
    fd, path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    try:
        with rasterio.open(path, "w", **output_profile) as destination:
            destination.write(mask.astype(np.uint8), 1)
        with open(path, "rb") as source:
            return source.read()
    finally:
        if os.path.exists(path):
            os.unlink(path)
