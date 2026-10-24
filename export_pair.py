import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
import rasterio
from common import ensure, load_config, load_json
from interpret import to_display
from pairing import accepted_dirs


def save_jpeg(path, img, ec, quality):
    s = ec["preview_px"] / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    return s


def export(d, cfg):
    ec, vc = cfg["export"], cfg["vlm"]
    d = Path(d)
    meta = load_json(d / "pair.json")
    out = ensure(Path(ec["out_dir"]) / d.name)
    for name in ec["files"]:
        if (d / name).exists():
            shutil.copy2(d / name, out / name)
    with rasterio.open(d / "before.tif") as f:
        a = f.read()
    with rasterio.open(d / "after.tif") as f:
        b = f.read()
    ia, ib = to_display(a, meta["modality"], vc), to_display(b, meta["modality"], vc)
    save_jpeg(out / "before.jpg", ia, ec, vc["jpeg_quality"])
    s = save_jpeg(out / "after.jpg", ib, ec, vc["jpeg_quality"])
    over = ib.copy()
    for r in load_json(d / "regions.json")["regions"]:
        pts = np.array(r["polygon_px"]).astype("int32").reshape(-1, 1, 2)
        cv2.polylines(over, [pts], True, (255, 0, 0), 2)
    save_jpeg(out / "changes.jpg", over, ec, vc["jpeg_quality"])
    return out


def run(cfg, pair_dirs):
    return [export(d, cfg) for d in pair_dirs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    for out in run(cfg, accepted_dirs(cfg, args.pair)):
        print(out)


if __name__ == "__main__":
    main()
