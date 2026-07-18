import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from geobase_inference.core import RequestValidationError
from geobase_inference.models.changestar import ChangeStarHandler
from geobase_inference.models.clay import ClayHandler


def test_changestar_request_defaults() -> None:
    request = ChangeStarHandler._parse_request({"inputs": "https://example.com/image.tif"})
    assert request.threshold == 0.5
    assert request.overlap == 64
    assert request.use_bucket is False


def test_changestar_rejects_non_1024_tiles() -> None:
    with pytest.raises(RequestValidationError, match="tile_size must be 1024"):
        ChangeStarHandler._parse_request(
            {
                "inputs": "https://example.com/image.tif",
                "parameters": {"tile_size": 512},
            }
        )


def test_clay_requires_chip_size() -> None:
    with pytest.raises(RequestValidationError, match="chip_size is required"):
        ClayHandler._parse_request({"inputs": "https://example.com/image.tif"})


def test_clay_request_defaults() -> None:
    request = ClayHandler._parse_request(
        {
            "inputs": "https://example.com/image.tif",
            "chip_size": 256,
        }
    )
    assert request.sensor == "naip"
    assert request.embedding_type == "global"
    assert request.format == "json"
    assert request.use_bucket is False


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

    class FakeEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

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
