# Clay

`ClayHandler` generates global or patch embeddings from every
[shared imagery input](../README.md#imagery-inputs).

## Installation

```text
geobase-inference[clay] @ git+https://github.com/decision-labs/geobase-inference.git@v0.1.1
```

The model repository exposes the handler through `handler.py`:

```python
from geobase_inference.models import ClayHandler


class EndpointHandler(ClayHandler):
    pass
```

## Request

```json
{
  "inputs": {
    "imagery": "https://example.com/image.tif",
    "chip_size": 512,
    "sensor": "naip",
    "date": "2022-07-04",
    "embedding_type": "global",
    "format": "json"
  }
}
```

`chip_size` is required and divisible by 16. `embedding_type` is `global`
or `patch`; `format` is `json` or `geoarrow`. Sensor names must exist in
the Clay model repository's metadata.

## Response

JSON responses contain spatial embedding records:

```json
{
  "results": [
    {
      "id": "chip_0001",
      "geom": {"type": "Polygon", "coordinates": []},
      "embeddings": [0.12, -0.04]
    }
  ]
}
```

For GeoArrow output, Clay returns Arrow IPC bytes with content type
`application/vnd.apache.arrow.stream`.

With [Hub bucket persistence](../README.md#optional-hub-bucket-output), Clay
stores a Parquet file and returns `row_count` plus `storage`. The storage
object includes `provider`, `bucket`, `keys`, `key`, and `format`.
