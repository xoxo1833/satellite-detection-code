import cv2
import numpy as np
import rasterio
from rasterio.features import shapes
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from common import dump_json, get_logger, load_json
from pathlib import Path

log = get_logger("detect")


def load_pair(d):
    d = Path(d)
    with rasterio.open(d / "before.tif") as f:
        a, tf, crs = f.read(), f.transform, f.crs
    with rasterio.open(d / "after.tif") as f:
        b = f.read()
    with rasterio.open(d / "valid.tif") as f:
        v = f.read(1) > 0
    return a, b, v, tf, crs, load_json(d / "pair.json")


def robust_z(x, valid):
    vals = x[valid]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-9
    return (x - med) / mad


def classical(a, b, valid, modality, cfg):
    dc = cfg["detect"]
    idx = [0] if modality == "sar" else dc["bands"]
    diff = b[idx] - a[idx]
    mag = np.sqrt((diff ** 2).mean(0)).astype("float32")
    mag = cv2.GaussianBlur(mag, (0, 0), dc["blur_sigma"][modality])
    z = robust_z(mag, valid)
    th = dc["threshold"]
    if th["type"] == "otsu":
        u8 = np.clip(255 * mag / max(mag[valid].max(), 1e-9), 0, 255).astype("uint8")
        t, _ = cv2.threshold(u8[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        score = u8.astype("float32") / 255.0
        mask = u8 > t * th["value"]
    else:
        mask = z > th["value"]
        score = 1.0 / (1.0 + np.exp(-(z - th["value"])))
    return score.astype("float32"), mask & valid


def pad_to(n, t, step):
    return t - n if n < t else (-(n - t)) % step


def learned(a, b, valid, cfg):
    import torch
    from model import SiamUNet
    lc = cfg["detect"]["learned"]
    dev = torch.device(lc["device"])
    ck = torch.load(lc["checkpoint"], map_location=dev)
    net = SiamUNet(ck["model"]["in_channels"], ck["model"]["widths"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    idx = cfg["detect"]["bands"]
    xa, xb = a[idx], b[idx]
    t, ov = lc["tile"], lc["overlap"]
    step = t - ov
    c, h, w = xa.shape
    ph, pw = pad_to(h, t, step), pad_to(w, t, step)
    xa = np.pad(xa, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    xb = np.pad(xb, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    H, W = xa.shape[1:]
    acc, cnt = np.zeros((H, W), "float32"), np.zeros((H, W), "float32")
    with torch.no_grad():
        for y in range(0, H - t + 1, step):
            for x in range(0, W - t + 1, step):
                ta = torch.from_numpy(xa[None, :, y:y + t, x:x + t]).to(dev)
                tb = torch.from_numpy(xb[None, :, y:y + t, x:x + t]).to(dev)
                acc[y:y + t, x:x + t] += torch.sigmoid(net(ta, tb))[0, 0].cpu().numpy()
                cnt[y:y + t, x:x + t] += 1
    prob = (acc / np.maximum(cnt, 1))[:h, :w]
    return prob, (prob > lc["threshold"]) & valid


def clean(mask, cfg):
    dc = cfg["detect"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dc["morph_kernel"],) * 2)
    m = cv2.morphologyEx(mask.astype("uint8"), cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= dc["min_area_px"]
    return keep[lab].astype("uint8")


def regionize(mask, prob, tf, crs):
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    lab = lab.astype("int32")
    counts = np.bincount(lab.ravel(), minlength=n)
    sums = np.bincount(lab.ravel(), weights=prob.ravel(), minlength=n)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    inv = ~tf
    out = []
    for geom, val in shapes(lab, mask=lab > 0, transform=tf, connectivity=8):
        k = int(val)
        poly = shape(geom)
        ext = [list(inv * (x, y)) for x, y in poly.exterior.coords]
        px = np.array(ext)
        out.append({
            "id": k,
            "area_m2": poly.area,
            "area_px": int(counts[k]),
            "mean_prob": float(sums[k] / counts[k]),
            "bbox_px": [int(px[:, 0].min()), int(px[:, 1].min()), int(np.ceil(px[:, 0].max())), int(np.ceil(px[:, 1].max()))],
            "polygon_px": ext,
            "geometry_lonlat": mapping(shp_transform(to_ll, poly)),
        })
    return lab, out


def write_tif(path, arr, tf, crs, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=crs, transform=tf, compress="deflate") as f:
        f.write(arr.astype(dtype)[None])


def process(d, cfg):
    a, b, valid, tf, crs, meta = load_pair(d)
    method = cfg["detect"]["method_by_modality"][meta["modality"]]
    prob, mask = classical(a, b, valid, meta["modality"], cfg) if method == "classical" else learned(a, b, valid, cfg)
    mask = clean(mask, cfg)
    lab, regions = regionize(mask, prob, tf, crs)
    d = Path(d)
    write_tif(d / "prob.tif", prob, tf, crs, "float32")
    write_tif(d / "labels.tif", lab, tf, crs, "int32")
    dump_json({"method": method, "regions": regions}, d / "regions.json")
    log.info("%s: %d regions (%s)", meta["pair_id"], len(regions), method)
    return regions


def run(cfg, pair_dirs):
    return {str(d): process(d, cfg) for d in pair_dirs}
