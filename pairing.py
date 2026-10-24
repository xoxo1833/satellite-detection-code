import hashlib
import itertools
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape
from skimage.exposure import match_histograms
from common import dump_json, ensure, get_logger, utm_epsg

log = get_logger("pairing")


def both(a, b):
    return not (a is None or b is None or pd.isna(a) or pd.isna(b))


def build_candidates(cat, cfg):
    pc = cfg["pairing"]
    aoi = box(*cfg["aoi"]["bbox"])
    cat = cat.copy()
    cat["dt"] = pd.to_datetime(cat["datetime"], utc=True, errors="coerce")
    cat = cat.dropna(subset=["dt"]).sort_values("dt")
    rows = []
    for src, g in cat.groupby("source"):
        recs = g.to_dict("records")
        geoms = [shape(json.loads(r["geometry"])).intersection(aoi) for r in recs]
        found = []
        idx = range(len(recs))
        win = pc.get("windows")
        if win:
            wa, wb = (tuple(pd.to_datetime(x, utc=True) for x in win[k].split("/")) for k in ("before", "after"))
            combos = itertools.product([i for i in idx if wa[0] <= recs[i]["dt"] <= wa[1]],
                                       [i for i in idx if wb[0] <= recs[i]["dt"] <= wb[1]])
        else:
            combos = itertools.combinations(idx, 2)
        for i, j in combos:
            a, b = recs[i], recs[j]
            gap = (b["dt"] - a["dt"]).total_seconds() / 86400
            if not pc["min_gap_days"] <= gap <= pc["max_gap_days"]:
                continue
            cov = geoms[i].intersection(geoms[j]).area / aoi.area
            if cov < pc["min_aoi_coverage"]:
                continue
            fails = []
            ga, gb = a["gsd"], b["gsd"]
            if both(ga, gb):
                if max(ga, gb) / min(ga, gb) > pc["max_gsd_ratio"]:
                    fails.append("gsd_ratio")
            elif pc["strict_missing"]:
                fails.append("gsd_missing")
            if both(a["off_nadir"], b["off_nadir"]):
                if abs(a["off_nadir"] - b["off_nadir"]) > pc["max_off_nadir_diff"]:
                    fails.append("off_nadir")
            elif pc["strict_missing"]:
                fails.append("off_nadir_missing")
            if pc["same_platform"] and a["platform"] != b["platform"] and not (pd.isna(a["platform"]) and pd.isna(b["platform"])):
                fails.append("platform")
            if pc["same_orbit_state"] and a["modality"] == "sar" and a["orbit_state"] != b["orbit_state"]:
                fails.append("orbit_state")
            for r in (a, b):
                if r["modality"] == "optical" and not pd.isna(r["cloud"]) and r["cloud"] > pc["max_cloud"]:
                    fails.append("cloud")
                    break
            if not fails:
                cl = sum(0 if pd.isna(r["cloud"]) else r["cloud"] for r in (a, b))
                found.append((cov, -cl if win else -abs(gap - pc["target_gap_days"]), gap, a, b))
        found.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for cov, _, gap, a, b in found[: pc["max_pairs_per_source"]]:
            rows.append({"a": a, "b": b, "coverage": cov, "gap_days": gap})
    return rows


def grid_for(cfg, res):
    w, s, e, n = cfg["aoi"]["bbox"]
    epsg = utm_epsg((w + e) / 2, (s + n) / 2)
    minx, miny, maxx, maxy = transform_bounds("EPSG:4326", f"EPSG:{epsg}", w, s, e, n)
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    return f"EPSG:{epsg}", from_origin(minx, maxy, res, res), width, height


def read_band(href, crs, transform, width, height, resampling, nodata):
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                       resampling=resampling, src_nodata=src.nodata, nodata=nodata) as v:
            return v.read(1).astype("float32")


def load_scene(rec, crs, transform, w, h, pc):
    assets = json.loads(rec["assets"])
    keys = [k for k in assets if k != "_mask"]
    bands = np.stack([read_band(assets[k], crs, transform, w, h, Resampling.bilinear, pc["nodata_value"]) for k in keys])
    valid = (bands != pc["nodata_value"]).all(0)
    if "_mask" in assets:
        classes = rec.get("mask_classes") or []
        m = read_band(assets["_mask"], crs, transform, w, h, Resampling.nearest, pc["nodata_value"])
        valid &= ~np.isin(m, classes)
    return bands, valid, keys


def to_working(bands, valid, modality, pc):
    if modality == "sar":
        x = np.maximum(bands, 1e-6)
        k = 20.0 if pc["sar_input"] == "amplitude" else 10.0
        return k * np.log10(x)
    out = np.empty_like(bands)
    for i, b in enumerate(bands):
        lo, hi = np.percentile(b[valid], [pc["pct_low"], pc["pct_high"]])
        out[i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def gray(x, valid):
    g = x.mean(0).astype("float32")
    g[~valid] = g[valid].mean() if valid.any() else 0
    return g


def phase_shift(ga, gb):
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ga, gb, win)
    return dx, dy, resp


def shift_image(x, dx, dy):
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return np.stack([cv2.warpAffine(b, m, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0) for b in x])


def verify_and_prepare(cand, cfg):
    pc = cfg["pairing"]
    a, b = cand["a"], cand["b"]
    src = next(s for s in cfg["sources"] if s["name"] == a["source"])
    a["mask_classes"] = b["mask_classes"] = src.get("mask_classes")
    res = pc["target_res_m"] or max(a["gsd"], b["gsd"])
    crs, transform, w, h = grid_for(cfg, res)
    pid = hashlib.sha1(f"{a['id']}|{b['id']}".encode()).hexdigest()[:10]
    meta = {"pair_id": pid, "source": a["source"], "modality": a["modality"], "before_id": a["id"], "after_id": b["id"],
            "before_date": str(a["dt"]), "after_date": str(b["dt"]), "gap_days": cand["gap_days"],
            "aoi_coverage": cand["coverage"], "crs": crs, "res_m": res, "width": w, "height": h,
            "transform": list(transform)[:6], "license": [a.get("license"), b.get("license")], "steps": []}
    if w * h > pc["max_pixels"]:
        return {**meta, "verdict": "reject", "reason": "grid_too_large"}
    try:
        ba, va, keys = load_scene(a, crs, transform, w, h, pc)
        bb, vb, _ = load_scene(b, crs, transform, w, h, pc)
    except Exception as e:
        return {**meta, "verdict": "reject", "reason": f"read_failed:{e}"}
    valid = va & vb
    vf = float(valid.mean())
    meta.update(bands=keys, valid_fraction=vf)
    if vf < pc["min_valid_fraction"]:
        return {**meta, "verdict": "reject", "reason": "valid_fraction"}
    xa, xb = to_working(ba, va, a["modality"], pc), to_working(bb, vb, b["modality"], pc)
    dx, dy, resp = phase_shift(gray(xa, valid), gray(xb, valid))
    shift = float(np.hypot(dx, dy))
    meta.update(shift_px=shift, reg_response=float(resp))
    verdict = "accept"
    if shift > pc["reg_max_shift_px"]:
        if shift > pc["reg_repair_limit_px"] or resp < pc["reg_min_response"]:
            return {**meta, "verdict": "reject", "reason": "registration"}
        xb = shift_image(xb, dx, dy)
        valid &= shift_image(valid[None].astype("float32"), dx, dy)[0] > 0.5
        dx2, dy2, _ = phase_shift(gray(xa, valid), gray(xb, valid))
        meta["shift_after_px"] = float(np.hypot(dx2, dy2))
        if meta["shift_after_px"] > pc["reg_max_shift_px"]:
            return {**meta, "verdict": "reject", "reason": "registration_unrepaired"}
        verdict = "repair"
        meta["steps"].append("translation_correction")
    if a["modality"] == "optical" and pc["match_histograms"]:
        xb = match_histograms(xb, xa, channel_axis=0).astype("float32")
        meta["steps"].append("histogram_match")
    d = ensure(f"{cfg['paths']['prepared']}/{pid}")
    prof = {"driver": "GTiff", "height": h, "width": w, "crs": crs, "transform": transform, "compress": "deflate"}
    for name, arr in (("before", xa), ("after", xb)):
        with rasterio.open(d / f"{name}.tif", "w", count=arr.shape[0], dtype="float32", **prof) as f:
            f.write(arr.astype("float32"))
    with rasterio.open(d / "valid.tif", "w", count=1, dtype="uint8", **prof) as f:
        f.write(valid.astype("uint8")[None])
    lon = np.mean(cfg["aoi"]["bbox"][0::2])
    lat = np.mean(cfg["aoi"]["bbox"][1::2])
    cell = pc["dedupe_cell_deg"]
    meta["site_id"] = f"{round(lon / cell)}_{round(lat / cell)}"
    meta["verdict"] = verdict
    dump_json(meta, d / "pair.json")
    return meta


def run(cfg):
    cat = pd.read_csv(cfg["paths"]["catalog"])
    out = []
    for cand in build_candidates(cat, cfg):
        m = verify_and_prepare(cand, cfg)
        log.info("%s %s %s", m["pair_id"], m["verdict"], m.get("reason", ""))
        out.append({k: v for k, v in m.items() if k not in ("transform", "steps")})
    if not out:
        raise SystemExit("No candidate pairs. Check runs/catalog.csv: each window in pairing.windows needs scenes covering at least min_aoi_coverage of the area with cloud below max_cloud.")
    df = pd.DataFrame(out)
    df.to_csv(cfg["paths"]["pairs"], index=False)
    return df


def accepted_dirs(cfg, only=None):
    df = pd.read_csv(cfg["paths"]["pairs"])
    df = df[df["verdict"].isin(["accept", "repair"])]
    if only:
        df = df[df["pair_id"].isin(only)]
    return [Path(cfg["paths"]["prepared"]) / p for p in df["pair_id"]]
