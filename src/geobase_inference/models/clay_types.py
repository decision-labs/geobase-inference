"""Request and response types for the Clay v1.5 model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from geobase_inference.core import BaseModelRequest
from geobase_inference.output_types import (
    GeoArrowBytes,
    ParquetBucketStorage,
    PersistedResults,
    ResultCollection,
    SpatialEmbedding,
)

ClaySensor = Literal[
    "sentinel-2-l2a",
    "planetscope-sr",
    "landsat-c2l1",
    "landsat-c2l2-sr",
    "naip",
    "linz",
    "sentinel-1-rtc",
    "modis",
    "satellogic-MSI-L1D",
]
ClayEmbeddingType = Literal["global", "patch"]
ClayOutputFormat = Literal["json", "geoarrow"]

SUPPORTED_CLAY_SENSORS: tuple[ClaySensor, ...] = (
    "sentinel-2-l2a",
    "planetscope-sr",
    "landsat-c2l1",
    "landsat-c2l2-sr",
    "naip",
    "linz",
    "sentinel-1-rtc",
    "modis",
    "satellogic-MSI-L1D",
)


@dataclass(frozen=True)
class ClayRequest(BaseModelRequest):
    """Validated parameters accepted by the Clay inference handler."""

    sensor: ClaySensor
    date: str | None
    chip_size: int
    embedding_type: ClayEmbeddingType
    clay_output_format: ClayOutputFormat
    use_bucket: bool
    output_prefix: str | None


ClayEmbedding: TypeAlias = SpatialEmbedding
ClayResultsResponse: TypeAlias = ResultCollection[ClayEmbedding]
ClayPersistedResponse: TypeAlias = PersistedResults[ParquetBucketStorage]

# GeoArrow responses use the shared binary Arrow IPC transport type.
ClayResponse: TypeAlias = (
    ClayResultsResponse | ClayPersistedResponse | GeoArrowBytes
)
