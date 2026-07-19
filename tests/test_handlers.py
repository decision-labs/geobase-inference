import sys
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from geobase_inference.core import RequestValidationError
from geobase_inference.geo import EsriImageryInput, GeoTiffInput, MapboxImageryInput
from geobase_inference.models.changestar import ChangeStarHandler
from geobase_inference.models.clay import ClayHandler


def test_changestar_request_defaults() -> None:
    request = ChangeStarHandler._parse_request({"imagery": "https://example.com/image.tif"})
    assert request.threshold == 0.5
    assert request.overlap == 64
    assert request.output_crs == "EPSG:4326"
    assert request.use_bucket is False
    assert isinstance(request.imagery, GeoTiffInput)


def test_changestar_accepts_custom_output_crs() -> None:
    request = ChangeStarHandler._parse_request(
        {
            "imagery": "https://example.com/image.tif",
            "parameters": {"output_crs": "EPSG:3857"},
        }
    )
    assert request.output_crs == "EPSG:3857"


def test_changestar_rejects_invalid_output_crs() -> None:
    with pytest.raises(RequestValidationError, match="Invalid output_crs"):
        ChangeStarHandler._parse_request(
            {
                "imagery": "https://example.com/image.tif",
                "output_crs": "not-a-crs",
            }
        )


def test_changestar_rejects_non_1024_tiles() -> None:
    with pytest.raises(RequestValidationError, match="tile_size must be 1024"):
        ChangeStarHandler._parse_request(
            {
                "imagery": "https://example.com/image.tif",
                "parameters": {"tile_size": 512},
            }
        )


def test_handlers_accept_shared_provider_inputs() -> None:
    polygon = {
        "type": "Polygon",
        "coordinates": [
            [
                [-117.60, 47.65],
                [-117.59, 47.65],
                [-117.59, 47.66],
                [-117.60, 47.65],
            ]
        ],
    }
    changestar = ChangeStarHandler._parse_request(
        {
            "imagery": {
                "type": "esri",
                "url": "https://example.com/ImageServer",
                "polygon": polygon,
                "mapParams": {"size": [512, 512]},
            }
        }
    )
    clay = ClayHandler._parse_request(
        {
            "imagery": {
                "type": "mapbox",
                "polygon": polygon,
                "mapParams": {"zoom": 16},
            },
            "chip_size": 256,
        }
    )
    assert isinstance(changestar.imagery, EsriImageryInput)
    assert isinstance(clay.imagery, MapboxImageryInput)


def test_clay_requires_chip_size() -> None:
    with pytest.raises(RequestValidationError, match="chip_size is required"):
        ClayHandler._parse_request({"imagery": "https://example.com/image.tif"})


def test_clay_request_defaults() -> None:
    request = ClayHandler._parse_request(
        {
            "imagery": "https://example.com/image.tif",
            "chip_size": 256,
        }
    )
    assert request.sensor == "naip"
    assert request.embedding_type == "global"
    assert request.clay_output_format == "json"
    assert request.use_bucket is False


def _stub_changestar_handler(monkeypatch) -> ChangeStarHandler:
    handler = object.__new__(ChangeStarHandler)
    handler.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    handler._run_inference = lambda path, request: (
        np.ones((2, 2), dtype=np.uint8),
        None,
        SimpleNamespace(to_string=lambda: "EPSG:3857"),
        {},
    )

    @contextmanager
    def temporary_geotiff(*args, **kwargs):
        yield "image.tif"

    monkeypatch.setattr(
        "geobase_inference.models.changestar.create_temporary_geotiff",
        temporary_geotiff,
    )
    monkeypatch.setattr(
        "geobase_inference.models.changestar.mask_to_geojson",
        lambda *args, **kwargs: {
            "type": "FeatureCollection",
            "features": [{"type": "Feature"}],
        },
    )
    return handler


def test_changestar_geojson_response_contract(monkeypatch) -> None:
    handler = _stub_changestar_handler(monkeypatch)
    response = handler(
        {
            "imagery": "https://example.com/image.tif",
            "use_bucket": False,
        }
    )

    assert response["polygon_count"] == 1
    assert response["building_pixels"] == 4
    assert response["geojson"]["type"] == "FeatureCollection"
    assert "storage" not in response


def test_changestar_persisted_response_contract(monkeypatch) -> None:
    handler = _stub_changestar_handler(monkeypatch)
    monkeypatch.setattr(
        "geobase_inference.models.changestar.hub_persistence_config",
        lambda **kwargs: {
            "bucket": "decision-labs/results",
            "token": "token",
            "prefix": "runs/",
        },
    )
    monkeypatch.setattr(
        "geobase_inference.models.changestar.mask_tiff_bytes",
        lambda *args: b"tiff",
    )
    monkeypatch.setattr(
        "geobase_inference.models.changestar.upload_artifacts_to_hub",
        lambda *args, **kwargs: [
            "runs/buildings.geojson",
            "runs/buildings_mask.tif",
        ],
    )

    response = handler(
        {
            "imagery": "https://example.com/image.tif",
            "use_bucket": True,
        }
    )

    assert response["storage"] == {
        "provider": "huggingface_hub",
        "bucket": "decision-labs/results",
        "keys": [
            "runs/buildings.geojson",
            "runs/buildings_mask.tif",
        ],
    }
    assert "geojson" not in response


def _stub_clay_handler() -> ClayHandler:
    handler = object.__new__(ClayHandler)
    handler.create_chips = lambda *args, **kwargs: (None, "chips")
    handler.load_chips = lambda path: [
        (
            np.ones((4, 16, 16), dtype=np.float32),
            "chip_0001.tif",
            [0.0, 0.0, 1.0, 1.0],
        )
    ]
    handler._prepare_pixels = lambda pixels: pixels
    handler._encode = lambda pixels, request, bounds: (
        np.array([[0.1, 0.2]], dtype=np.float32),
        np.array([[[0.3, 0.4]]], dtype=np.float32),
    )
    handler.bounds_to_geojson = lambda bounds: {
        "type": "Polygon",
        "coordinates": [],
    }
    handler.geojson_to_ewkb = lambda geometry: b"geometry"
    handler.results_to_arrow_ipc = lambda results: b"arrow-ipc"
    return handler


def test_clay_json_and_geoarrow_response_contracts() -> None:
    handler = _stub_clay_handler()
    request = ClayHandler._parse_request(
        {
            "imagery": "https://example.com/image.tif",
            "chip_size": 256,
            "use_bucket": False,
        }
    )

    json_response = handler._process_tiff(request, "image.tif")
    binary = handler._process_tiff(
        replace(request, clay_output_format="geoarrow"),
        "image.tif",
    )

    assert json_response == {
        "results": [
            {
                "id": "chip_0001",
                "geom": {"type": "Polygon", "coordinates": []},
                "embeddings": pytest.approx([0.1, 0.2]),
            }
        ]
    }
    assert binary == b"arrow-ipc"


def test_clay_persisted_response_contract(monkeypatch) -> None:
    handler = _stub_clay_handler()
    request = ClayHandler._parse_request(
        {
            "imagery": "https://example.com/image.tif",
            "chip_size": 256,
            "use_bucket": True,
            "output_prefix": "embeddings/result.parquet",
        }
    )
    monkeypatch.setattr(
        "geobase_inference.models.clay.hub_persistence_config",
        lambda **kwargs: {
            "bucket": "decision-labs/results",
            "token": "token",
            "prefix": "clay/",
        },
    )
    monkeypatch.setattr(
        "geobase_inference.models.clay.upload_file_to_hub",
        lambda *args, **kwargs: kwargs["key"],
    )

    response = handler._process_tiff(request, "image.tif")

    assert response == {
        "row_count": 1,
        "storage": {
            "provider": "huggingface_hub",
            "bucket": "decision-labs/results",
            "keys": ["embeddings/result.parquet"],
            "key": "embeddings/result.parquet",
            "format": "parquet",
        },
    }


def test_clay_handler_loads_model_assets_from_repository(tmp_path, monkeypatch) -> None:
    model_dir = tmp_path / "v1.5"
    config_dir = model_dir / "configs"
    config_dir.mkdir(parents=True)
    (model_dir / "clay-v1.5.ckpt").write_bytes(b"checkpoint")
    (config_dir / "metadata.yaml").write_text(
        "naip:\n"
        "  gsd: 1\n"
        "  band_order: [red]\n"
        "  bands:\n"
        "    mean: {red: 0.5}\n"
        "    std: {red: 0.2}\n"
        "    wavelength: {red: 0.65}\n"
    )

    class FakeEncoder:
        def __init__(self) -> None:
            self.weight = SimpleNamespace(device="cpu")

        def eval(self):
            return self

        def parameters(self):
            return iter([self.weight])

    class FakeClayModule:
        @staticmethod
        def load_from_checkpoint(path, map_location):
            assert path.endswith("clay-v1.5.ckpt")
            assert map_location == "cpu"
            return SimpleNamespace(model=SimpleNamespace(encoder=FakeEncoder()))

    claymodel = ModuleType("claymodel")
    claymodel_module = ModuleType("claymodel.module")
    claymodel_module.ClayMAEModule = FakeClayModule
    monkeypatch.setitem(sys.modules, "claymodel", claymodel)
    monkeypatch.setitem(sys.modules, "claymodel.module", claymodel_module)

    fake_image_utils = SimpleNamespace(
        bounds_to_geojson=lambda value: value,
        create_chips=lambda *args, **kwargs: None,
        download_imagery=lambda *args, **kwargs: None,
        load_chips=lambda *args, **kwargs: (),
        patch_bounds_from_chip=lambda *args, **kwargs: (),
    )
    fake_geoarrow_utils = SimpleNamespace(
        geojson_to_ewkb=lambda value: b"",
        results_to_arrow_ipc=lambda value: b"",
    )
    original_import = __import__("importlib").import_module

    def fake_import(name):
        if name == "image_utils":
            return fake_image_utils
        if name == "geoarrow_utils":
            return fake_geoarrow_utils
        return original_import(name)

    monkeypatch.setattr(
        "geobase_inference.models.clay.importlib.import_module",
        fake_import,
    )
    handler = ClayHandler(str(tmp_path))
    assert isinstance(handler.encoder, FakeEncoder)
    assert handler.metadata["naip"]["gsd"] == 1
