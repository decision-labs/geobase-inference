"""Optional artifact uploads to Hugging Face Hub buckets."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from geobase_inference.storage.hub_bucket_client import (
    HubBucketClient,
    XetWriteConnection,
)


def validate_hf_bucket_id(raw: str) -> str:
    bucket = raw.strip()
    if "/" not in bucket:
        raise ValueError(f"HF_BUCKET must be namespace/bucket-name; got {bucket!r}")
    namespace, name = bucket.split("/", 1)
    if not namespace or not name:
        raise ValueError("HF_BUCKET must be namespace/bucket-name")
    return bucket


def hub_persistence_config(
    *,
    default_prefix: str = "geobase-inference/",
) -> dict[str, str] | None:
    """Return deployment bucket config, or None when storage is not configured."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    raw_bucket = os.environ.get("HF_BUCKET")
    if not token or not raw_bucket:
        return None
    prefix = os.environ.get("HF_OUTPUT_PREFIX", default_prefix)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return {
        "token": token,
        "bucket": validate_hf_bucket_id(raw_bucket),
        "prefix": prefix,
    }


def _clean_key(value: str) -> str:
    parts = [part for part in value.strip().split("/") if part not in ("", ".")]
    if ".." in parts:
        raise ValueError("Bucket object keys cannot contain '..'")
    return "/".join(parts)


def upload_artifacts_to_hub(
    artifacts: Mapping[str, bytes],
    *,
    bucket: str,
    token: str,
    prefix: str,
) -> list[str]:
    """Upload byte artifacts through Xet and commit them to a Hub bucket."""
    from hf_xet import upload_bytes

    if not artifacts:
        return []
    bucket = validate_hf_bucket_id(bucket)
    clean_prefix = _clean_key(prefix)
    if clean_prefix:
        clean_prefix += "/"
    keys = [f"{clean_prefix}{_clean_key(name)}" for name in artifacts]
    bodies = list(artifacts.values())

    client = HubBucketClient(token)
    client.create_bucket_if_needed(bucket)

    def refresh_token() -> tuple[str, int]:
        connection: XetWriteConnection = client.fetch_xet_write_token(bucket)
        return connection.access_token, connection.expiration_unix_epoch

    connection = client.fetch_xet_write_token(bucket)
    upload_infos = upload_bytes(
        bodies,
        connection.endpoint,
        (connection.access_token, connection.expiration_unix_epoch),
        refresh_token,
        None,
        "bucket",
    )
    if len(upload_infos) != len(keys):
        raise RuntimeError(f"Hub uploaded {len(upload_infos)} objects for {len(keys)} artifacts")

    mtime = int(time.time() * 1000)
    payloads: list[dict[str, Any]] = [
        {
            "type": "addFile",
            "path": key,
            "xetHash": info.hash,
            "mtime": mtime,
        }
        for key, info in zip(keys, upload_infos, strict=True)
    ]
    client.batch_commit_add_files(bucket, payloads)
    return keys


def upload_file_to_hub(
    file_path: str,
    *,
    bucket: str,
    token: str,
    key: str,
) -> str:
    """Upload one local file and return its committed bucket key."""
    with open(file_path, "rb") as source:
        body = source.read()
    prefix, _, name = _clean_key(key).rpartition("/")
    keys = upload_artifacts_to_hub(
        {name: body},
        bucket=bucket,
        token=token,
        prefix=prefix,
    )
    return keys[0]
