# Satellite change detection with agentic pairing and VLM interpretation

Pipeline that (1) discovers real multi-temporal satellite imagery from open AWS-hosted STAC catalogs, (2) builds and verifies before/after pairs of the same location, (3) detects and localizes changes, and (4) describes and verifies each change with a vision-language model (Claude). All imagery is real Sentinel-2 data from the AWS Registry of Open Data. No synthetic or generated imagery is used.

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/xoxo1833/satellite-detection-code.git
cd satellite-detection-code
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first. Python 3.10 or newer. Internet access is required; no AWS account or credentials are needed.

## Build the dataset (real imagery)

```powershell
python run.py dataset --config config.yaml
```

This runs four stages, which can also be run one at a time (`discover`, `pair`, `detect`, `export`):

1. `discover` queries the Element 84 Earth Search STAC API (`sentinel-2-l2a`, backed by the public `sentinel-cogs` bucket) for scenes over the area of interest and writes `runs/catalog.csv`.
2. `pair` selects one scene from the "before" window and one from the "after" window (lowest combined cloud cover, full coverage of the area), reads only the needed window from the cloud-hosted COGs, masks cloud and shadow with the scene classification layer, checks registration, histogram-matches the pair and writes aligned GeoTIFFs. Verdicts and rejection reasons are in `runs/pairs.csv`.
3. `detect` computes a change-probability map and vectorizes changed regions.
4. `export` copies each accepted pair to `data/real/<pair_id>/` with `before.jpg`, `after.jpg` and `changes.jpg` previews.

Example configuration: Navi Mumbai International Airport (about 1,160 ha at Ulwe, opened December 2025), before window January to March 2021, after window January to March 2026. Both windows are in the dry season to reduce seasonal differences. Change `aoi.bbox` and `pairing.windows` in `config.yaml` to use another site.

Commit the results so the dataset travels with the code:

```powershell
git add data/real
git commit -m "Add real Sentinel-2 pairs"
git push
```

Keep the area small (a few km across) so files stay under GitHub's size limits.

## VLM interpretation

```powershell
$env:ANTHROPIC_API_KEY = "your-key"
python run.py interpret --config config.yaml
```

For each region the VLM sees before, after and outlined crops and returns a structured description. Each claim is checked by a swap test (reversed order must invert the change), a null test (identical images must report no change), pixel-evidence contrast, a nuisance flag, and an independent judge call. Output: `runs/reports/<pair_id>/report.geojson`, with status `verified`, `uncertain` or `rejected`. Model names and prompts are in the config.

## Open imagery sources

`data/catalog/sources.csv` lists AWS-hosted and other open sources (Sentinel-2, Umbra SAR, Vantor open data, NAIP, Capella, SpaceNet) with bucket, access and license. `data/catalog/datasets.csv` lists labelled change-detection benchmarks. Respect each license: Vantor open data is CC BY-NC, and NAIP sits in a requester-pays bucket.

## Optional learned detector

```powershell
pip install -r requirements-learned.txt
python run.py train --config config.yaml
```

Expects a LEVIR-CD style folder (`train/A`, `train/B`, `train/label`, same for `val`) at `train.root`. Afterwards set `detect.method_by_modality.optical` to `learned`.

## Repository layout

| Path | Purpose |
|---|---|
| `run.py` | Command-line entry point for all stages |
| `discover.py` | STAC discovery and metadata normalization |
| `pairing.py` | Pair selection, temporal/spatial/radiometric verification, preparation |
| `detect.py`, `model.py`, `train.py` | Classical and learned change detection, training |
| `interpret.py` | VLM description and verification |
| `export_pair.py` | Copies accepted pairs into `data/real/` with JPEG previews |
| `config.yaml` | All parameters, prompts and sources |
| `data/` | Real pairs, source and dataset catalogs |
| `notebooks/` | Notebook version of the pipeline |

## Status and limitations

- The live stages were not run end to end by the author in the development environment, because it could not reach the AWS-hosted imagery. The pairing, verification, detection and interpretation logic was tested on locally generated test arrays only; those test files are not part of this repository. Report any failure you hit when running on real data.
- Classical detector thresholds are untuned, and real scenes will produce many false regions from vegetation, water and lighting. The VLM verification stage exists to filter these.
- Sentinel-2 processing baselines differ across years, so radiometric offsets between 2021 and 2026 are possible. The pipeline stretches each image and histogram-matches the pair, which reduces but does not remove this.
- Only openly licensed imagery is intended. Report observable change only.
