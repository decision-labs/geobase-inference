"""Typed imagery inputs and provider-specific raster materialization."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from io import BytesIO
from typing import Any

import mercantile
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import shape

from geobase_inference.core import RequestValidationError
from geobase_inference.geo.input_types import (
    EsriImageryInput,
    GeoTiffInput,
    ImageryInput,
    MapboxImageryInput,
)

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"


def _temporary_path(suffix: str = ".tif") -> str:
    """Create and return an empty temporary file path."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


def _download_to_temp(
    url: str,
    *,
    suffix: str = "",
    connect_timeout: float = 15,
    read_timeout: float = 120,
    logger: logging.Logger | None = None,
) -> str:
    """Stream a remote imagery resource into a temporary local file."""
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


def _response_bytes(response: requests.Response) -> bytes:
    """Validate an HTTP response and collect its streamed body."""
    response.raise_for_status()
    return b"".join(response.iter_content(chunk_size=1 << 20))


def _write_clipped_geotiff(
    path: str,
    data: np.ndarray,
    *,
    crs: str,
    transform: Any,
    polygon: Mapping[str, Any],
) -> None:
    """Mask a raster to the requested polygon and write a georeferenced TIFF."""
    projected_polygon = transform_geom(WGS84, crs, polygon)
    inside = geometry_mask(
        [projected_polygon],
        out_shape=(data.shape[1], data.shape[2]),
        transform=transform,
        invert=True,
    )
    clipped = np.where(inside[np.newaxis, ...], data, 0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=clipped.shape[2],
        height=clipped.shape[1],
        count=clipped.shape[0],
        dtype=clipped.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
        compress="deflate",
    ) as destination:
        destination.write(clipped)


def _create_esri_geotiff(
    source: EsriImageryInput, logger: logging.Logger | None
) -> str:
    """Export, clip, and save ESRI ImageServer imagery as a temporary GeoTIFF."""
    bounds = shape(source.polygon).bounds
    projected_bounds = transform_bounds(
        WGS84,
        WEB_MERCATOR,
        *bounds,
        densify_pts=21,
    )
    crs = rasterio.crs.CRS.from_user_input(WEB_MERCATOR)
    spatial_reference: int | str = crs.to_epsg() or crs.to_wkt()
    params: dict[str, Any] = {
        "f": "image",
        "bbox": ",".join(str(value) for value in projected_bounds),
        "bboxSR": spatial_reference,
        "imageSR": spatial_reference,
        "format": "tiff",
    }
    optional = {
        "size": (
            f"{source.map_params.size[0]},{source.map_params.size[1]}"
            if source.map_params.size
            else None
        ),
        "interpolation": source.map_params.interpolation,
        "time": source.map_params.time,
        "bandIds": (
            ",".join(str(value) for value in source.map_params.band_ids)
            if source.map_params.band_ids
            else None
        ),
        "renderingRule": (
            json.dumps(source.map_params.rendering_rule)
            if source.map_params.rendering_rule
            else None
        ),
        "mosaicRule": (
            json.dumps(source.map_params.mosaic_rule)
            if source.map_params.mosaic_rule
            else None
        ),
        "noData": source.map_params.no_data,
        "token": os.environ.get("ESRI_TOKEN"),
    }
    params.update({key: value for key, value in optional.items() if value is not None})
    endpoint = f"{source.url.rstrip('/')}/exportImage"
    with requests.post(endpoint, data=params, stream=True, timeout=(15, 180)) as response:
        body = _response_bytes(response)
        if "json" in response.headers.get("Content-Type", "").lower():
            try:
                detail = json.loads(body).get("error", {}).get("message")
            except Exception:
                detail = None
            raise RequestValidationError(f"ESRI exportImage failed: {detail or 'unexpected JSON'}")
    raw_path = _temporary_path()
    output_path = _temporary_path()
    try:
        with open(raw_path, "wb") as output:
            output.write(body)
        with rasterio.open(raw_path) as source_raster:
            data = source_raster.read()
        transform = from_bounds(
            *projected_bounds,
            width=data.shape[2],
            height=data.shape[1],
        )
        _write_clipped_geotiff(
            output_path,
            data,
            crs=WEB_MERCATOR,
            transform=transform,
            polygon=source.polygon,
        )
        return output_path
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise
    finally:
        if os.path.exists(raw_path):
            os.unlink(raw_path)


def _create_mapbox_geotiff(
    source: MapboxImageryInput,
    logger: logging.Logger | None,
) -> str:
    """Download, mosaic, clip, and save Mapbox tiles as a temporary GeoTIFF."""
    token = os.environ.get("MAPBOX_ACCESS_TOKEN")
    if not token:
        raise RequestValidationError(
            "Mapbox imagery requires the MAPBOX_ACCESS_TOKEN endpoint secret"
        )
    bounds = shape(source.polygon).bounds
    tiles = list(mercantile.tiles(*bounds, zooms=source.map_params.zoom))
    if not tiles:
        raise RequestValidationError("Mapbox polygon does not intersect any tiles")
    min_x = min(tile.x for tile in tiles)
    max_x = max(tile.x for tile in tiles)
    min_y = min(tile.y for tile in tiles)
    max_y = max(tile.y for tile in tiles)
    tile_size = source.map_params.tile_size
    mosaic = np.zeros(
        (
            3,
            (max_y - min_y + 1) * tile_size,
            (max_x - min_x + 1) * tile_size,
        ),
        dtype=np.uint8,
    )
    suffix = "@2x" if tile_size == 512 else ""
    if logger:
        logger.info(
            "Fetching %d Mapbox tiles tileset=%s zoom=%d",
            len(tiles),
            source.map_params.tileset,
            source.map_params.zoom,
        )
    for tile in tiles:
        url = (
            f"https://api.mapbox.com/v4/{source.map_params.tileset}/"
            f"{tile.z}/{tile.x}/{tile.y}{suffix}.{source.map_params.image_format}"
        )
        with requests.get(
            url,
            params={"access_token": token},
            stream=True,
            timeout=(15, 60),
        ) as response:
            body = _response_bytes(response)
        try:
            image = Image.open(BytesIO(body)).convert("RGB")
        except Exception as error:
            raise RequestValidationError("Mapbox returned an invalid raster tile") from error
        if image.size != (tile_size, tile_size):
            image = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
        tile_data = np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))
        row = (tile.y - min_y) * tile_size
        column = (tile.x - min_x) * tile_size
        mosaic[:, row : row + tile_size, column : column + tile_size] = tile_data
    upper_left = mercantile.xy_bounds(mercantile.Tile(min_x, min_y, source.map_params.zoom))
    lower_right = mercantile.xy_bounds(
        mercantile.Tile(max_x, max_y, source.map_params.zoom)
    )
    transform = from_bounds(
        upper_left.left,
        lower_right.bottom,
        lower_right.right,
        upper_left.top,
        width=mosaic.shape[2],
        height=mosaic.shape[1],
    )
    output_path = _temporary_path()
    try:
        _write_clipped_geotiff(
            output_path,
            mosaic,
            crs=WEB_MERCATOR,
            transform=transform,
            polygon=source.polygon,
        )
        return output_path
    except Exception:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise


@contextmanager
def create_temporary_geotiff(
    source: ImageryInput,
    *,
    logger: logging.Logger | None = None,
) -> Iterator[str]:
    """Create a temporary GeoTIFF from any imagery source and clean it up."""
    if isinstance(source, GeoTiffInput):
        path = _download_to_temp(source.url, suffix=".tif", logger=logger)
    elif isinstance(source, EsriImageryInput):
        path = _create_esri_geotiff(source, logger)
    elif isinstance(source, MapboxImageryInput):
        path = _create_mapbox_geotiff(source, logger)
    else:
        raise TypeError(f"Unsupported imagery input: {type(source).__name__}")
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.unlink(path)
