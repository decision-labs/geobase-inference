import os
from unittest.mock import patch

import pytest

from geobase_inference.core import (
    RequestValidationError,
    request_value,
    require_http_url,
)
from geobase_inference.geo import feather_weight, tile_to_rgb_uint8


def test_request_value_supports_nested_parameters() -> None:
    data = {"parameters": {"threshold": 0.7}}
    assert request_value(data, "threshold", 0.5) == 0.7


def test_top_level_request_value_wins() -> None:
    data = {"threshold": 0.8, "parameters": {"threshold": 0.7}}
    assert request_value(data, "threshold", 0.5) == 0.8


def test_require_http_url() -> None:
    assert require_http_url({"url": "https://example.com/a.tif"}) == ("https://example.com/a.tif")
    with pytest.raises(RequestValidationError):
        require_http_url("/tmp/a.tif")


def test_raster_helpers() -> None:
    import numpy as np

    single_band = np.ones((1, 4, 4), dtype=np.uint16)
    rgb = tile_to_rgb_uint8(single_band)
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8
    assert feather_weight(8, 2).shape == (8, 8)


def test_no_environment_leak() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert "HF_TOKEN" not in os.environ
