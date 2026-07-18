"""ChangeStar ONNX building-segmentation endpoint integration."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import onnxruntime as ort
import rasterio
from rasterio.windows import Window

from geobase_inference.core import (
    RequestValidationError,
    configure_logging,
    download_to_temp,
    request_value,
    require_http_url,
    require_mapping,
)
from geobase_inference.geo import (
    feather_weight,
    mask_tiff_bytes,
    mask_to_geojson,
    tile_to_rgb_uint8,
)
from geobase_inference.storage import (
    hub_persistence_config,
    upload_artifacts_to_hub,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TILE_SIZE = 1024
DEFAULT_OVERLAP = 64
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class ChangeStarRequest:
    url: str
    overlap: int
    threshold: float
    use_bucket: bool
    output_prefix: str | None


class ChangeStarHandler:
    """Hugging Face handler for tiled ChangeStar building segmentation."""

    def __init__(self, path: str = "") -> None:
        configure_logging()
        self.logger = logging.getLogger(type(self).__name__)
        self.path = path.rstrip("/") if path else "."
        model_path = self._find_model()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        started = time.perf_counter()
        self.logger.info("Loading %s with providers %s", model_path, providers)
        self.session = ort.InferenceSession(model_path, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.logger.info(
            "Model ready in %.1fs (input=%s, shape=%s)",
            time.perf_counter() - started,
            self.input_name,
            model_input.shape,
        )

    def _find_model(self) -> str:
        candidates = (
            "onnx/model_quantized.onnx",
            "onnx/model.onnx",
            "model_quantized.onnx",
            "model.onnx",
        )
        for relative_path in candidates:
            candidate = os.path.join(self.path, relative_path)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No ONNX model found under {os.path.abspath(self.path)}; "
            f"checked {', '.join(candidates)}"
        )

    @staticmethod
    def _parse_request(raw: Any) -> ChangeStarRequest:
        data = require_mapping(raw)
        url = require_http_url(data.get("inputs"))
        tile_size = int(request_value(data, "tile_size", TILE_SIZE))
        if tile_size != TILE_SIZE:
            raise RequestValidationError(
                f"tile_size must be {TILE_SIZE}; the exported ViT positional "
                "embeddings do not support other dimensions"
            )
        overlap = int(request_value(data, "overlap", DEFAULT_OVERLAP))
        if overlap < 0 or overlap >= TILE_SIZE:
            raise RequestValidationError(f"overlap must satisfy 0 <= overlap < {TILE_SIZE}")
        threshold = float(request_value(data, "threshold", DEFAULT_THRESHOLD))
        if not 0 <= threshold <= 1:
            raise RequestValidationError("threshold must be between 0 and 1")

        use_bucket = request_value(data, "use_bucket", None)
        if use_bucket is None:
            use_bucket = hub_persistence_config(default_prefix="building-segmentation/") is not None
        if not isinstance(use_bucket, bool):
            raise RequestValidationError("use_bucket must be a boolean")
        output_prefix = request_value(data, "output_prefix", None)
        if output_prefix is not None and not isinstance(output_prefix, str):
            raise RequestValidationError("output_prefix must be a string")
        return ChangeStarRequest(
            url=url,
            overlap=overlap,
            threshold=threshold,
            use_bucket=use_bucket,
            output_prefix=output_prefix,
        )

    @staticmethod
    def _preprocess(rgb: np.ndarray) -> np.ndarray:
        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(tensor, (2, 0, 1))
        return tensor[np.newaxis, ...].astype(np.float32)

    def _run_inference(
        self,
        tif_path: str,
        request: ChangeStarRequest,
    ) -> tuple[np.ndarray, Any, Any, dict[str, Any]]:
        started = time.perf_counter()
        with rasterio.open(tif_path) as source:
            if source.count == 2:
                raise RequestValidationError(
                    "Two-band rasters are unsupported; provide one or at least three bands"
                )
            height, width = source.height, source.width
            profile = source.profile.copy()
            transform = source.transform
            crs = source.crs
            accum = np.zeros((height, width), dtype=np.float32)
            weight_sum = np.zeros((height, width), dtype=np.float32)
            weights = feather_weight(TILE_SIZE, request.overlap)
            stride = TILE_SIZE - request.overlap
            total = len(range(0, height, stride)) * len(range(0, width, stride))
            processed = 0
            skipped = 0
            indexes = list(range(1, min(source.count, 3) + 1))
            self.logger.info(
                "Raster %dx%d bands=%d crs=%s; %d tiles",
                width,
                height,
                source.count,
                crs,
                total,
            )

            for row_offset in range(0, height, stride):
                for column_offset in range(0, width, stride):
                    tile_height = min(TILE_SIZE, height - row_offset)
                    tile_width = min(TILE_SIZE, width - column_offset)
                    raw = source.read(
                        indexes=indexes,
                        window=Window(
                            column_offset,
                            row_offset,
                            tile_width,
                            tile_height,
                        ),
                    )
                    if raw.size == 0 or not np.any(raw):
                        skipped += 1
                        continue
                    pad_height = TILE_SIZE - tile_height
                    pad_width = TILE_SIZE - tile_width
                    if pad_height or pad_width:
                        mode = "reflect" if tile_height > 1 and tile_width > 1 else "edge"
                        raw = np.pad(
                            raw,
                            (
                                (0, 0),
                                (0, pad_height),
                                (0, pad_width),
                            ),
                            mode=mode,
                        )

                    probability = self.session.run(
                        None,
                        {self.input_name: self._preprocess(tile_to_rgb_uint8(raw))},
                    )[0]
                    probability = np.asarray(probability).squeeze()
                    if probability.ndim != 2:
                        raise RuntimeError(
                            "Expected a single-channel probability map, got "
                            f"shape {probability.shape}"
                        )
                    probability = probability[:tile_height, :tile_width]
                    weight = weights[:tile_height, :tile_width]
                    rows = slice(row_offset, row_offset + tile_height)
                    columns = slice(
                        column_offset,
                        column_offset + tile_width,
                    )
                    accum[rows, columns] += probability * weight
                    weight_sum[rows, columns] += weight
                    processed += 1
                    if processed == 1 or processed % 10 == 0:
                        self.logger.info(
                            "Tile progress: %d/%d processed (%d skipped)",
                            processed,
                            total,
                            skipped,
                        )

        weight_sum[weight_sum == 0] = 1
        mask = (accum / weight_sum > request.threshold).astype(np.uint8)
        self.logger.info(
            "Inference completed in %.1fs (%d processed, %d skipped)",
            time.perf_counter() - started,
            processed,
            skipped,
        )
        return mask, transform, crs, profile

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        request = self._parse_request(data)
        self.logger.info("Processing %s", request.url)
        tif_path = download_to_temp(
            request.url,
            suffix=".tif",
            logger=self.logger,
        )
        try:
            mask, transform, crs, profile = self._run_inference(
                tif_path,
                request,
            )
            geojson = mask_to_geojson(mask, transform, crs)
            building_pixels = int(mask.sum())
            result: dict[str, Any] = {
                "polygon_count": len(geojson["features"]),
                "building_pixels": building_pixels,
                "building_coverage": building_pixels / max(mask.size, 1),
                "width": int(mask.shape[1]),
                "height": int(mask.shape[0]),
                "crs": crs.to_string() if crs else None,
            }

            if request.use_bucket:
                config = hub_persistence_config(default_prefix="building-segmentation/")
                if config is None:
                    raise RequestValidationError(
                        "use_bucket=true requires HF_BUCKET and HF_TOKEN "
                        "(or HUGGING_FACE_HUB_TOKEN)"
                    )
                date = datetime.now(timezone.utc).strftime("%Y%m%d")
                prefix = request.output_prefix or (f"{config['prefix']}{date}/{uuid.uuid4().hex}/")
                keys = upload_artifacts_to_hub(
                    {
                        "buildings.geojson": json.dumps(geojson).encode(),
                        "buildings_mask.tif": mask_tiff_bytes(mask, profile),
                    },
                    bucket=config["bucket"],
                    token=config["token"],
                    prefix=prefix,
                )
                result["storage"] = {
                    "provider": "huggingface_hub",
                    "bucket": config["bucket"],
                    "keys": keys,
                }
            else:
                result["geojson"] = geojson

            result["duration_seconds"] = round(
                time.perf_counter() - started,
                3,
            )
            return result
        except Exception:
            self.logger.exception(
                "Request failed after %.1fs",
                time.perf_counter() - started,
            )
            raise
        finally:
            if os.path.exists(tif_path):
                os.unlink(tif_path)
