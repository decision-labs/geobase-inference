"""Clay foundation-model endpoint integration."""

from __future__ import annotations

import gc
import importlib
import logging
import os
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from httpx import Client, Timeout
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions

from geobase_inference.core import (
    RequestValidationError,
    configure_logging,
    request_value,
    require_http_url,
    require_mapping,
)
from geobase_inference.storage import (
    hub_persistence_config,
    upload_file_to_hub,
)

PATCH_SIZE = 16
GEOARROW_CONTENT_TYPE = "application/vnd.apache.arrow.stream"
PARQUET_SCHEMA = pa.schema(
    [
        pa.field("id", pa.utf8()),
        pa.field("geom", pa.binary()),
        pa.field("embeddings", pa.list_(pa.float32())),
    ]
)


def _register_arrow_content_type(logger: logging.Logger) -> None:
    try:
        from huggingface_inference_toolkit.serialization.base import (
            content_type_mapping,
        )

        class ArrowSerializer:
            @staticmethod
            def deserialize(body: bytes) -> dict[str, bytes]:
                return {"inputs": body}

            @staticmethod
            def serialize(data: Any, accept: str | None = None) -> bytes:
                del accept
                if isinstance(data, (bytes, bytearray)):
                    return bytes(data)
                raise ValueError(f"Expected Arrow bytes, got {type(data).__name__}")

        content_type_mapping[GEOARROW_CONTENT_TYPE] = ArrowSerializer
    except ImportError:
        logger.info("Inference toolkit unavailable; Arrow type not registered")


@dataclass(frozen=True)
class ClayRequest:
    inputs: str
    sensor: str
    date: str | None
    chip_size: int
    embedding_type: str
    format: str
    geobase_url: str | None
    geobase_key: str | None
    geobase_bucket: str | None
    geobase_path: str | None
    use_bucket: bool
    output_prefix: str | None


class ClayHandler:
    """Hugging Face endpoint handler for Clay v1.5 embeddings."""

    chunk_rows = 4000

    def __init__(self, path: str = "") -> None:
        configure_logging()
        self.logger = logging.getLogger(type(self).__name__)
        self.path = os.path.abspath(path.rstrip("/") if path else ".")
        if self.path not in sys.path:
            sys.path.insert(0, self.path)

        checkpoint_path = os.path.join(
            self.path,
            "v1.5",
            "clay-v1.5.ckpt",
        )
        metadata_path = os.path.join(
            self.path,
            "v1.5",
            "configs",
            "metadata.yaml",
        )
        model_dir = os.path.join(self.path, "v1.5")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        with open(metadata_path) as source:
            self.metadata = yaml.safe_load(source)

        image_utils = importlib.import_module("image_utils")
        self.bounds_to_geojson = image_utils.bounds_to_geojson
        self.create_chips = image_utils.create_chips
        self.download_imagery = image_utils.download_imagery
        self.load_chips = image_utils.load_chips
        self.patch_bounds_from_chip = image_utils.patch_bounds_from_chip

        geoarrow_utils = importlib.import_module("geoarrow_utils")
        self.geojson_to_ewkb = geoarrow_utils.geojson_to_ewkb
        self.results_to_arrow_ipc = geoarrow_utils.results_to_arrow_ipc

        from claymodel.module import ClayMAEModule

        cwd = os.getcwd()
        try:
            os.chdir(model_dir)
            full_model = ClayMAEModule.load_from_checkpoint(
                checkpoint_path,
                map_location="cpu",
            )
        finally:
            os.chdir(cwd)
        self.encoder = full_model.model.encoder
        self.encoder.eval()
        del full_model
        gc.collect()

        if torch.cuda.is_available():
            self.encoder = self.encoder.cuda()
            torch.cuda.empty_cache()
        self.device = next(self.encoder.parameters()).device
        _register_arrow_content_type(self.logger)
        self.logger.info("Clay encoder ready on %s", self.device)

    @staticmethod
    def _parse_request(raw: Any) -> ClayRequest:
        data = require_mapping(raw)
        inputs = require_http_url(data.get("inputs"))
        chip_size_raw = request_value(data, "chip_size", None)
        if chip_size_raw is None:
            raise RequestValidationError("chip_size is required")
        chip_size = int(chip_size_raw)
        if chip_size <= 0 or chip_size % PATCH_SIZE:
            raise RequestValidationError(
                f"chip_size must be positive and divisible by {PATCH_SIZE}"
            )
        embedding_type = str(request_value(data, "embedding_type", "global")).lower()
        if embedding_type not in {"patch", "global"}:
            raise RequestValidationError("embedding_type must be 'patch' or 'global'")
        output_format = str(request_value(data, "format", "json")).lower()
        if output_format not in {"json", "geoarrow"}:
            raise RequestValidationError("format must be 'json' or 'geoarrow'")
        use_bucket = request_value(data, "use_bucket", None)
        if use_bucket is None:
            use_bucket = hub_persistence_config(default_prefix="clay-hub/") is not None
        if not isinstance(use_bucket, bool):
            raise RequestValidationError("use_bucket must be a boolean")
        output_prefix = request_value(data, "output_prefix", None)
        if output_prefix is not None and not isinstance(output_prefix, str):
            raise RequestValidationError("output_prefix must be a string")
        return ClayRequest(
            inputs=inputs,
            sensor=str(request_value(data, "sensor", "naip")),
            date=request_value(data, "date", None),
            chip_size=chip_size,
            embedding_type=embedding_type,
            format=output_format,
            geobase_url=request_value(data, "geobase_url", None),
            geobase_key=request_value(data, "geobase_key", None),
            geobase_bucket=request_value(data, "geobase_bucket", None),
            geobase_path=request_value(data, "geobase_path", None),
            use_bucket=use_bucket,
            output_prefix=output_prefix,
        )

    def _build_datacube(
        self,
        pixels: torch.Tensor,
        request: ClayRequest,
        bbox: list[float],
    ) -> dict[str, Any]:
        if request.sensor not in self.metadata:
            raise RequestValidationError(
                f"Unknown sensor {request.sensor!r}; supported: {list(self.metadata)}"
            )
        meta = self.metadata[request.sensor]
        band_order = meta["band_order"]
        means = torch.tensor(
            [meta["bands"]["mean"][band] for band in band_order],
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        stds = torch.tensor(
            [meta["bands"]["std"][band] for band in band_order],
            dtype=torch.float32,
        ).view(1, -1, 1, 1)
        pixels = (pixels - means) / stds

        if request.date:
            date = datetime.strptime(request.date, "%Y-%m-%d")
            week_angle = date.isocalendar()[1] * 2 * np.pi / 52
            hour_angle = np.pi
            time_encoding = [
                np.sin(week_angle),
                np.cos(week_angle),
                np.sin(hour_angle),
                np.cos(hour_angle),
            ]
        else:
            time_encoding = [0.0] * 4

        center_lat = (bbox[1] + bbox[3]) / 2
        center_lon = (bbox[0] + bbox[2]) / 2
        lat_angle = center_lat * np.pi / 180
        lon_angle = center_lon * np.pi / 180
        latlon_encoding = [
            np.sin(lat_angle),
            np.cos(lat_angle),
            np.sin(lon_angle),
            np.cos(lon_angle),
        ]
        batch_size = pixels.shape[0]
        waves = [meta["bands"]["wavelength"][band] for band in band_order]
        return {
            "pixels": pixels.to(self.device),
            "platform": [request.sensor] * batch_size,
            "date": [request.date or "2020-01-01"] * batch_size,
            "time": torch.tensor(
                [time_encoding] * batch_size,
                dtype=torch.float32,
                device=self.device,
            ),
            "bbox": torch.tensor(
                [bbox] * batch_size,
                dtype=torch.float32,
                device=self.device,
            ),
            "latlon": torch.tensor(
                [latlon_encoding] * batch_size,
                dtype=torch.float32,
                device=self.device,
            ),
            "gsd": torch.tensor(
                meta["gsd"],
                dtype=torch.float32,
                device=self.device,
            ),
            "waves": torch.tensor(
                waves,
                dtype=torch.float32,
                device=self.device,
            ),
        }

    def _prepare_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        _, _, height, width = pixels.shape
        height = height // PATCH_SIZE * PATCH_SIZE
        width = width // PATCH_SIZE * PATCH_SIZE
        if not height or not width:
            raise RequestValidationError(f"Image dimensions must be at least {PATCH_SIZE}")
        return pixels[:, :, :height, :width].to(self.device)

    def _encode(
        self,
        pixels: torch.Tensor,
        request: ClayRequest,
        bbox: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        datacube = self._build_datacube(pixels, request, bbox)
        with torch.no_grad():
            encoded = self.encoder(datacube)[0]
        return (
            encoded[:, 0, :].cpu().numpy(),
            encoded[:, 1:, :].cpu().numpy(),
        )

    @staticmethod
    def _write_parquet_row(
        writer: pq.ParquetWriter,
        row_id: str,
        geometry: bytes,
        embeddings: list[float],
    ) -> None:
        writer.write_batch(
            pa.RecordBatch.from_arrays(
                [
                    pa.array([row_id], type=pa.utf8()),
                    pa.array([geometry], type=pa.binary()),
                    pa.array(
                        [embeddings],
                        type=pa.list_(pa.float32()),
                    ),
                ],
                schema=PARQUET_SCHEMA,
            )
        )

    def _upload_to_supabase(
        self,
        file_path: str,
        request: ClayRequest,
    ) -> tuple[list[str], list[str]]:
        table = pq.read_table(file_path)
        base_path = (
            request.geobase_path
            or "clay-output/"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d')}/"
            f"{uuid.uuid4().hex}.parquet"
        )
        if base_path.endswith(".parquet"):
            base_path = base_path[:-8]
        http_client = Client(timeout=Timeout(600), http2=False)
        options = SyncClientOptions(httpx_client=http_client)
        client = create_client(
            request.geobase_url,
            request.geobase_key,
            options=options,
        )
        urls: list[str] = []
        paths: list[str] = []
        try:
            for index in range(0, table.num_rows, self.chunk_rows):
                chunk = table.slice(
                    index,
                    min(self.chunk_rows, table.num_rows - index),
                )
                chunk_path = (
                    f"{base_path}.parquet"
                    if table.num_rows <= self.chunk_rows
                    else f"{base_path}.part{index // self.chunk_rows:04d}.parquet"
                )
                sink = pa.BufferOutputStream()
                pq.write_table(chunk, sink)
                body = sink.getvalue().to_pybytes()
                bucket = client.storage.from_(request.geobase_bucket)
                last_error: Exception | None = None
                for attempt in range(3):
                    try:
                        bucket.upload(
                            chunk_path,
                            body,
                            file_options={"content-type": "application/octet-stream"},
                        )
                        urls.append(bucket.get_public_url(chunk_path))
                        paths.append(chunk_path)
                        last_error = None
                        break
                    except Exception as error:
                        last_error = error
                        if attempt < 2:
                            time.sleep(2**attempt)
                if last_error:
                    raise last_error
        finally:
            http_client.close()
        return urls, paths

    def _process(self, request: ClayRequest) -> dict[str, Any] | bytes:
        tiff_path = self.download_imagery(
            request.inputs,
            request.sensor,
            metadata=self.metadata,
        )
        _, output_folder = self.create_chips(
            tiff_path,
            chip_size=request.chip_size,
        )
        images_dir = os.path.join(output_folder, "images")
        supabase_enabled = bool(
            request.geobase_url and request.geobase_key and request.geobase_bucket
        )
        bucket_config = (
            hub_persistence_config(default_prefix="clay-hub/") if request.use_bucket else None
        )
        if request.use_bucket and bucket_config is None:
            raise RequestValidationError(
                "use_bucket=true requires HF_BUCKET and HF_TOKEN (or HUGGING_FACE_HUB_TOKEN)"
            )
        stream = supabase_enabled or bucket_config is not None
        results: list[dict[str, Any]] = []
        parquet_path: str | None = None
        writer: pq.ParquetWriter | None = None
        row_count = 0

        if stream:
            fd, parquet_path = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            writer = pq.ParquetWriter(parquet_path, PARQUET_SCHEMA)
        try:
            for chip_data, chip_path, bounds in self.load_chips(images_dir):
                pixels = self._prepare_pixels(torch.from_numpy(chip_data))
                global_embeddings, patch_embeddings = self._encode(
                    pixels,
                    request,
                    bounds,
                )
                chip_id = Path(chip_path).stem
                if request.embedding_type == "patch":
                    _, _, height, width = pixels.shape
                    patch_bounds = self.patch_bounds_from_chip(
                        bounds,
                        height // PATCH_SIZE,
                        width // PATCH_SIZE,
                    )
                    rows = (
                        (
                            f"{chip_id}_patch_{index:04d}",
                            geometry,
                            embedding.tolist(),
                        )
                        for index, (embedding, geometry) in enumerate(
                            zip(
                                patch_embeddings[0],
                                patch_bounds,
                                strict=True,
                            )
                        )
                    )
                else:
                    rows = iter(
                        [
                            (
                                chip_id,
                                bounds,
                                global_embeddings[0].tolist(),
                            )
                        ]
                    )
                for row_id, bounds_value, embedding in rows:
                    geometry = self.bounds_to_geojson(bounds_value)
                    if writer:
                        self._write_parquet_row(
                            writer,
                            row_id,
                            self.geojson_to_ewkb(geometry),
                            embedding,
                        )
                        row_count += 1
                    else:
                        results.append(
                            {
                                "id": row_id,
                                "geom": geometry,
                                "embeddings": embedding,
                            }
                        )

            if writer and parquet_path:
                writer.close()
                writer = None
                response: dict[str, Any] = {"row_count": row_count}
                if supabase_enabled:
                    urls, paths = self._upload_to_supabase(
                        parquet_path,
                        request,
                    )
                    response.update(
                        {
                            "urls": urls,
                            "paths": paths,
                            "url": urls[0] if urls else None,
                            "path": paths[0] if paths else None,
                        }
                    )
                if bucket_config:
                    date = datetime.now(timezone.utc).strftime("%Y%m%d")
                    key = request.output_prefix or (
                        f"{bucket_config['prefix']}{date}/{uuid.uuid4().hex}.parquet"
                    )
                    committed_key = upload_file_to_hub(
                        parquet_path,
                        bucket=bucket_config["bucket"],
                        token=bucket_config["token"],
                        key=key,
                    )
                    response["storage"] = {
                        "provider": "huggingface_hub",
                        "bucket": bucket_config["bucket"],
                        "keys": [committed_key],
                        "key": committed_key,
                        "format": "parquet",
                    }
                return response
        finally:
            if writer:
                writer.close()
            if parquet_path and os.path.exists(parquet_path):
                os.unlink(parquet_path)

        if request.format == "geoarrow":
            return self.results_to_arrow_ipc(results)
        return {"results": results}

    def __call__(self, data: dict[str, Any]) -> dict[str, Any] | bytes:
        request = self._parse_request(data)
        self.logger.info(
            "Clay request sensor=%s embedding_type=%s format=%s",
            request.sensor,
            request.embedding_type,
            request.format,
        )
        return self._process(request)
