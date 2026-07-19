# ChangeStar

`ChangeStarHandler` performs ONNX building segmentation over tiled imagery.
It accepts every [shared imagery input](../README.md#imagery-inputs).

## Installation

```text
geobase-inference[changestar] @ git+https://github.com/decision-labs/geobase-inference.git@v0.1.1
```

The model repository exposes the handler through `handler.py`:

```python
from geobase_inference.models import ChangeStarHandler


class EndpointHandler(ChangeStarHandler):
    pass
```

## Output CRS

Building GeoJSON is returned in `EPSG:4326` by default. Set `output_crs`
at the top level or inside `parameters` to request another CRS:

```json
{
  "imagery": "https://example.com/image.tif",
  "parameters": {"output_crs": "EPSG:3857"}
}
```

The mask GeoTIFF remains in the source raster CRS.

## Response

Without bucket persistence, ChangeStar returns summary metrics and a GeoJSON
FeatureCollection:

```json
{
  "polygon_count": 3,
  "building_pixels": 1842,
  "building_coverage": 0.18,
  "width": 1024,
  "height": 1024,
  "crs": "EPSG:3857",
  "output_crs": "EPSG:4326",
  "geojson": {"type": "FeatureCollection", "features": []},
  "duration_seconds": 2.4
}
```

With [Hub bucket persistence](../README.md#optional-hub-bucket-output), the
response retains its summary metrics and replaces `geojson` with `storage`.
The uploaded artifacts are `buildings.geojson` and `buildings_mask.tif`.
