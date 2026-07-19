import os
from io import BytesIO

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from geobase_inference.core import RequestValidationError
from geobase_inference.geo.input_types import (
    EsriImageryInput,
    GeoTiffInput,
    MapboxImageryInput,
)
from geobase_inference.geo.input_validation import parse_imagery_input
from geobase_inference.geo.sources import create_temporary_geotiff


@pytest.fixture
def polygon() -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [1.0, 1.0],
                [1.01, 1.0],
                [1.005, 1.01],
                [1.0, 1.0],
            ]
        ],
    }


class FakeResponse:
    def __init__(self, body: bytes, content_type: str) -> None:
        self.body = body
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body


def _tiff_bytes() -> bytes:
    data = np.full((3, 8, 8), 100, dtype=np.uint8)
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=8,
            height=8,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=from_bounds(0, 0, 1, 1, 8, 8),
        ) as destination:
            destination.write(data)
        return memory.read()


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (256, 256), color=(10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def test_parse_legacy_and_typed_geotiff() -> None:
    legacy = parse_imagery_input("https://example.com/image.tif")
    typed = parse_imagery_input(
        {"type": "geotiff", "url": "https://example.com/image.tif"}
    )
    assert legacy == typed == GeoTiffInput(
        type="geotiff",
        url="https://example.com/image.tif",
    )


def test_parse_esri_map_params(polygon) -> None:
    source = parse_imagery_input(
        {
            "type": "esri",
            "url": "https://example.com/ImageServer",
            "polygon": polygon,
            "mapParams": {
                "size": [512, 256],
                "bandIds": [0, 1, 2],
            },
        }
    )
    assert isinstance(source, EsriImageryInput)
    assert source.map_params.size == (512, 256)
    assert source.map_params.band_ids == (0, 1, 2)


def test_provider_size_defaults(polygon) -> None:
    esri = parse_imagery_input(
        {
            "type": "esri",
            "url": "https://example.com/ImageServer",
            "polygon": polygon,
        }
    )
    mapbox = parse_imagery_input(
        {
            "type": "mapbox",
            "polygon": polygon,
            "mapParams": {"zoom": 10},
        }
    )
    assert isinstance(esri, EsriImageryInput)
    assert isinstance(mapbox, MapboxImageryInput)
    assert esri.map_params.size is None
    assert mapbox.map_params.tile_size == 256
    assert mapbox.map_params.image_format == "png"


def test_parse_mapbox_requires_zoom(polygon) -> None:
    with pytest.raises(RequestValidationError, match="zoom is required"):
        parse_imagery_input(
            {
                "type": "mapbox",
                "polygon": polygon,
                "mapParams": {"tileset": "mapbox.satellite"},
            }
        )


def test_esri_materialization_clips_polygon(monkeypatch, polygon) -> None:
    captured = {}

    def fake_post(url, data, stream, timeout):
        captured.update(url=url, data=data, stream=stream, timeout=timeout)
        return FakeResponse(_tiff_bytes(), "image/tiff")

    monkeypatch.setattr("geobase_inference.geo.sources.requests.post", fake_post)
    source = parse_imagery_input(
        {
            "type": "esri",
            "url": "https://example.com/ImageServer",
            "polygon": polygon,
            "mapParams": {"size": [8, 8]},
        }
    )
    with create_temporary_geotiff(source) as path:
        assert os.path.exists(path)
        with rasterio.open(path) as raster:
            data = raster.read()
            assert raster.crs.to_string() == "EPSG:3857"
            assert data.max() == 100
            assert data.min() == 0
    assert not os.path.exists(path)
    assert captured["url"].endswith("/exportImage")
    assert captured["data"]["size"] == "8,8"


def test_mapbox_materialization_mosaics_and_clips(monkeypatch, polygon) -> None:
    monkeypatch.setenv("MAPBOX_ACCESS_TOKEN", "secret")
    requested = []

    def fake_get(url, params, stream, timeout):
        requested.append((url, params, stream, timeout))
        return FakeResponse(_png_bytes(), "image/png")

    monkeypatch.setattr("geobase_inference.geo.sources.requests.get", fake_get)
    source = parse_imagery_input(
        {
            "type": "mapbox",
            "polygon": polygon,
            "mapParams": {
                "tileset": "mapbox.satellite",
                "zoom": 10,
                "tileSize": 256,
            },
        }
    )
    assert isinstance(source, MapboxImageryInput)
    with create_temporary_geotiff(source) as path:
        with rasterio.open(path) as raster:
            data = raster.read()
            assert raster.crs.to_string() == "EPSG:3857"
            assert raster.count == 3
            assert data.max() == 30
            assert data.min() == 0
    assert not os.path.exists(path)
    assert requested
    assert requested[0][1] == {"access_token": "secret"}
