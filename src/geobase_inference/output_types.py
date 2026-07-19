"""Shared response types for geospatial model handlers."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeAlias, TypedDict, TypeVar

GeoJSON: TypeAlias = dict[str, Any]
GeoArrowBytes: TypeAlias = bytes

ResultT = TypeVar("ResultT")
StorageT = TypeVar("StorageT", bound="BucketStorage")


class SpatialEmbedding(TypedDict):
    """One embedding associated with a GeoJSON geometry."""

    id: str
    geom: GeoJSON
    embeddings: list[float]


class ResultCollection(TypedDict, Generic[ResultT]):
    """Model result records returned directly in JSON."""

    results: list[ResultT]


class GeoJSONOutput(TypedDict):
    """GeoJSON returned directly by a model."""

    geojson: GeoJSON


class BucketStorage(TypedDict):
    """Common Hugging Face Hub bucket result metadata."""

    provider: Literal["huggingface_hub"]
    bucket: str
    keys: list[str]


class ParquetBucketStorage(BucketStorage):
    """Bucket metadata for one Parquet result."""

    key: str
    format: Literal["parquet"]


class PersistedResults(TypedDict, Generic[StorageT]):
    """Metadata returned instead of inline model results."""

    row_count: int
    storage: StorageT
