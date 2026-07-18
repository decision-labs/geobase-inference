import os
from unittest.mock import patch

import pytest

from geobase_inference.storage import (
    hub_persistence_config,
    upload_artifacts_to_hub,
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


def test_artifacts_are_uploaded_and_committed(monkeypatch) -> None:
    import hf_xet

    committed = {}

    class FakeInfo:
        hash = "xet-hash"

    class FakeConnection:
        endpoint = "https://xet.example"
        access_token = "xet-token"
        expiration_unix_epoch = 9999999999

    class FakeClient:
        def __init__(self, token):
            assert token == "hub-token"

        def create_bucket_if_needed(self, bucket):
            assert bucket == "decision-labs/results"

        def fetch_xet_write_token(self, bucket):
            assert bucket == "decision-labs/results"
            return FakeConnection()

        def batch_commit_add_files(self, bucket, payloads):
            committed["bucket"] = bucket
            committed["payloads"] = payloads

    def fake_upload_bytes(
        bodies,
        endpoint,
        token,
        refresh,
        progress,
        repo_type,
    ):
        assert bodies == [b"data"]
        assert endpoint == "https://xet.example"
        assert token == ("xet-token", 9999999999)
        assert refresh() == ("xet-token", 9999999999)
        assert progress is None
        assert repo_type == "bucket"
        return [FakeInfo()]

    monkeypatch.setattr(
        "geobase_inference.storage.hub.HubBucketClient",
        FakeClient,
    )
    monkeypatch.setattr(hf_xet, "upload_bytes", fake_upload_bytes)
    keys = upload_artifacts_to_hub(
        {"result.json": b"data"},
        bucket="decision-labs/results",
        token="hub-token",
        prefix="runs/123",
    )
    assert keys == ["runs/123/result.json"]
    assert committed["bucket"] == "decision-labs/results"
    assert committed["payloads"][0]["path"] == keys[0]
