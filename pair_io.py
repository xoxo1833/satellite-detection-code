from pathlib import Path
import cv2
import numpy as np
from common import dump_json, ensure


def to_u8(arr, pct):
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype("float32")
    lo, hi = np.percentile(a, pct)
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")


def fit(img, max_side):
    s = max_side / max(img.shape[:2])
    return img if s >= 1 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def with_banner(img, text, oc):
    b = np.full((oc["banner_px"], img.shape[1], 3), 255, "uint8")
    scale = oc["font_scale"]
    while scale > 0.3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, oc["font_thickness"])[0][0] > img.shape[1] - 16:
        scale -= 0.05
    cv2.putText(b, text, (8, int(oc["banner_px"] * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), oc["font_thickness"], cv2.LINE_AA)
    return np.vstack([b, img])


def side_by_side(a, b, label_a, label_b, oc):
    h = max(a.shape[0], b.shape[0])
    up = max(1.0, oc["min_panel_px"] / min(a.shape[1] * h / a.shape[0], b.shape[1] * h / b.shape[0]))
    h = round(h * up)
    rs = lambda x: cv2.resize(x, (round(x.shape[1] * h / x.shape[0]), h), interpolation=cv2.INTER_CUBIC)
    a, b = with_banner(rs(a), label_a, oc), with_banner(rs(b), label_b, oc)
    gap = np.full((a.shape[0], oc["gap_px"], 3), 255, "uint8")
    return np.hstack([a, gap, b])


def write_pair(root, rel, before, after, meta, oc):
    d = ensure(Path(root) / rel)
    a, b = fit(before, oc["max_side_px"]), fit(after, oc["max_side_px"])
    q = [cv2.IMWRITE_JPEG_QUALITY, oc["jpeg_quality"]]
    na = oc["missing_date_label"]
    sbs = side_by_side(a, b, f"BEFORE  {meta.get('before_date') or na}", f"AFTER  {meta.get('after_date') or na}", oc)
    for name, img in (("before.jpg", a), ("after.jpg", b), ("side_by_side.jpg", sbs)):
        cv2.imwrite(str(d / name), img[:, :, ::-1], q)
    dump_json(meta, d / "meta.json")
    return d
