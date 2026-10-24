# Data layout

- `real/` : prepared real image pairs written by `export_pair.py` (one folder per pair: aligned GeoTIFFs, `pair.json` metadata, `regions.json`, and `before.jpg` / `after.jpg` / `changes.jpg` previews).
- `catalog/sources.csv` : open imagery sources on AWS and elsewhere, with bucket, access and license.
- `catalog/datasets.csv` : survey of public change-detection benchmarks.

No synthetic or generated imagery is used anywhere in this repository.
