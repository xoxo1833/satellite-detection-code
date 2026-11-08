# Satellite change detection: real before-and-after image dataset and processing code

The deliverable of this repository is the **dataset**: real satellite and aerial image pairs, each stored in its own folder with the earlier image, the later image, the two side by side, and the dates of both. No detector output, red outlines or VLM text appears in the dataset. The processing code (detector and VLM interpretation) is kept separately and is not part of the dataset.

## Dataset layout

```
dataset/
  INDEX.csv                    folder, dataset, before_date, after_date (optional overview)
  DOWNLOAD_REPORT.md           which sources were downloaded, and the exact error for any that were not
  sentinel2/<site>/<before>_<after>/   before.jpg  after.jpg  side_by_side.jpg  meta.json
  levir_cd/<pair>/             same four files
  whu_cd/<pair>/
  oscd/<city>/
  xbd/<disaster>/<pair>/
  spacenet7/<aoi>/
```

`meta.json` holds both dates, the source, location, license and processing notes. Images are JPEG quality 95. Sentinel-2 images use a per-band percentile stretch for display and are not radiometrically calibrated.

## Sources

| Source | Imagery | Dates |
|---|---|---|
| Sentinel-2 L2A (AWS open data), three sites: Navi Mumbai airport, Western Sydney airport, Grand Ethiopian Renaissance Dam | 10 m, real scenes selected automatically | exact acquisition dates from the scene metadata |
| LEVIR-CD | 0.5 m aerial, building change | not published per pair (dataset range 2002 to 2018) |
| WHU-CD | 7.5 cm aerial, Christchurch NZ | 2012 and 2016 (years only) |
| OSCD | 10 m Sentinel-2, urban change | from each city's `dates.txt` |
| xBD | very high resolution, pre and post disaster | capture dates from label metadata |
| SpaceNet 7 | 4 m Planet monthly mosaics | year and month |

`docs/sources.csv` and `docs/datasets.csv` list further open sources and benchmarks.

xBD is non-commercial (CC BY-NC-SA) and Vantor open data is CC BY-NC. Both are used here for non-commercial academic work only. SpaceNet 7 is CC BY-SA 4.0. Check each license before reuse.

## Reproduce the dataset

```powershell
git clone https://github.com/xoxo1833/satellite-detection-code.git
cd satellite-detection-code
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python build_dataset.py --config config.yaml --dataset-config dataset.yaml
```

Internet access is required. No AWS account is needed. The script downloads automatically from public AWS buckets (Sentinel-2, SpaceNet 7) and from a Hugging Face mirror (LEVIR-CD+). For WHU-CD, OSCD and xBD, download the archives yourself, extract them under `external/` as described in `dataset.yaml` (`how_to_obtain` for each source), and run the same command. Add `--only sentinel2 oscd` to build a subset.

Every source that cannot be fetched is listed in `dataset/DOWNLOAD_REPORT.md` with the exact error and how to obtain it. Sites, date windows and counts are set in `dataset.yaml`.

## Processing code (separate from the dataset)

`run.py`, `pairing.py`, `detect.py`, `interpret.py` and the other modules implement automated pair discovery and verification, a classical change detector, and VLM-based description with verification. They write only to `runs/`, never to `dataset/`.

```powershell
python run.py dataset --config config.yaml
```

The VLM backend is configured under `vlm` in `config.yaml` (Gemini free tier, a local Ollama model, or Anthropic). Optional learned detector: `pip install -r requirements-learned.txt`, then `python run.py train --config config.yaml`.

## Files

| Path | Purpose |
|---|---|
| `build_dataset.py`, `public_sets.py`, `pair_io.py`, `dataset.yaml` | Build the image dataset |
| `discover.py`, `pairing.py` | Search AWS STAC catalogs, select and verify Sentinel-2 pairs |
| `run.py`, `detect.py`, `interpret.py`, `model.py`, `train.py`, `summarize.py` | Processing code |
| `config.yaml` | Parameters for discovery, pairing, detection and VLM |
| `docs/` | Source and benchmark catalogs |
| `notebooks/` | Notebook version |

## Limitations

- `dataset/` contents depend on what the build could download; read `DOWNLOAD_REPORT.md`.
- LEVIR-CD has no per-pair dates, and WHU-CD only years, because the datasets do not publish more.
- Sentinel-2 scene selection takes the lowest-cloud scenes inside each date window. Check each `side_by_side.jpg` before relying on a pair.
