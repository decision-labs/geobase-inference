import os
from unittest.mock import patch

import pytest

from geobase_inference.storage import (
    hub_persistence_config,
    validate_hf_bucket_id,
)


def test_bucket_storage_is_optional() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert hub_persistence_config() is None


def test_partial_bucket_config_is_disabled() -> None:
    with patch.dict(os.environ, {"HF_TOKEN": "token"}, clear=True):
        assert hub_persistence_config() is None


def test_complete_bucket_config() -> None:
    environment = {
        "HF_TOKEN": "token",
        "HF_BUCKET": "decision-labs/results",
        "HF_OUTPUT_PREFIX": "runs",
    }
    with patch.dict(os.environ, environment, clear=True):
        assert hub_persistence_config() == {
            "token": "token",
            "bucket": "decision-labs/results",
            "prefix": "runs/",
        }


def test_bucket_requires_namespace() -> None:
    with pytest.raises(ValueError):
        validate_hf_bucket_id("results")
