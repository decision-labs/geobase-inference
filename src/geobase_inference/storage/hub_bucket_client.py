"""Minimal HTTP client for Hugging Face Hub bucket operations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

_XET_ENDPOINT = "X-Xet-Cas-Url"
_XET_ACCESS_TOKEN = "X-Xet-Access-Token"
_XET_EXPIRATION = "X-Xet-Token-Expiration"


@dataclass(frozen=True)
class XetWriteConnection:
    endpoint: str
    access_token: str
    expiration_unix_epoch: int


class HubBucketClient:
    """Use bucket APIs without upgrading the toolkit-pinned Hub client."""

    def __init__(self, token: str, endpoint: str | None = None) -> None:
        self.token = token
        self.endpoint = (
            endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        ).rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "geobase-inference/hub-bucket-client",
        }

    def create_bucket_if_needed(self, bucket_id: str) -> None:
        namespace, name = bucket_id.split("/", 1)
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.endpoint}/api/buckets/{namespace}/{name}",
                json={"private": True},
                headers=self._headers(),
            )
        if response.status_code not in (200, 201, 409):
            response.raise_for_status()

    def fetch_xet_write_token(self, bucket_id: str) -> XetWriteConnection:
        with httpx.Client(timeout=120.0) as client:
            response = client.get(
                f"{self.endpoint}/api/buckets/{bucket_id}/xet-write-token",
                headers=self._headers(),
            )
        response.raise_for_status()
        return XetWriteConnection(
            endpoint=response.headers[_XET_ENDPOINT],
            access_token=response.headers[_XET_ACCESS_TOKEN],
            expiration_unix_epoch=int(response.headers[_XET_EXPIRATION]),
        )

    def batch_commit_add_files(
        self,
        bucket_id: str,
        add_payloads: list[dict[str, Any]],
    ) -> None:
        body = b"".join(json.dumps(payload).encode("utf-8") + b"\n" for payload in add_payloads)
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{self.endpoint}/api/buckets/{bucket_id}/batch",
                content=body,
                headers={
                    "Content-Type": "application/x-ndjson",
                    **self._headers(),
                },
            )
        response.raise_for_status()
