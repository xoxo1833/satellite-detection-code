import base64
import json
import random
from pathlib import Path
import anthropic
import cv2
import numpy as np
import rasterio
from common import dump_json, get_logger, load_json

log = get_logger("interpret")


def parse_json(text):
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        return None
    try:
        return json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None


def call_vlm(client, model, system, content, max_tokens):
    r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                               messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in r.content if b.type == "text")


def to_display(arr, modality, vc):
    if modality == "sar":
        g = arr[0]
        lo, hi = np.percentile(g, vc["display_pct"])
        g = np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.stack([g, g, g])
    else:
        rgb = arr[vc["rgb_idx"]]
    return (np.clip(rgb, 0, 1) * 255).astype("uint8").transpose(1, 2, 0)


def window(bbox, shape, vc):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, vc["min_chip_px"]) * (1 + vc["context_pad"]) / 2
    h, w = shape
    return (max(int(cx - half), 0), max(int(cy - half), 0), min(int(cx + half), w), min(int(cy + half), h))


def chip(img, win, vc, poly=None):
    x0, y0, x1, y1 = win
    c = img[y0:y1, x0:x1].copy()
    scale = vc["chip_px"] / max(c.shape[:2])
    if poly is not None:
        pts = (np.array(poly) - [x0, y0]).astype("int32").reshape(-1, 1, 2)
        cv2.polylines(c, [pts], True, (255, 0, 0), 2)
    c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", c[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, vc["jpeg_quality"]])
    return base64.standard_b64encode(buf.tobytes()).decode()


def img_block(b64):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}


def build_content(prompt, imgs):
    out = []
    for label, b64 in imgs:
        out += [{"type": "text", "text": label}, img_block(b64)]
    return out + [{"type": "text", "text": prompt}]


def contrast(prob, lab, region, ring):
    x0, y0, x1, y1 = region["bbox_px"]
    h, w = lab.shape
    wx0, wy0, wx1, wy1 = max(x0 - ring - 1, 0), max(y0 - ring - 1, 0), min(x1 + ring + 1, w), min(y1 + ring + 1, h)
    L, P = lab[wy0:wy1, wx0:wx1], prob[wy0:wy1, wx0:wx1]
    inside = L == region["id"]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1,) * 2)
    ringm = (cv2.dilate(inside.astype("uint8"), k) > 0) & (L == 0)
    if not ringm.any():
        return float(P[inside].mean())
    return float(P[inside].mean() - P[ringm].mean())


def evaluate(desc, swap, null, judge, evidence, ic):
    f = []
    if desc is None:
        return ["vlm_parse_failed"], "uncertain"
    if not desc.get("change_detected"):
        f.append("vlm_no_change")
    if swap is not None:
        exp = ic["inverse_types"].get(desc.get("change_type"), desc.get("change_type"))
        if not swap.get("change_detected") or swap.get("change_type") != exp:
            f.append("swap_inconsistent")
    if null is not None and null.get("change_detected"):
        f.append("null_hallucination")
    if evidence["mean_prob"] < ic["min_prob"] or evidence["contrast"] < ic["min_contrast"]:
        f.append("weak_pixel_evidence")
    if desc.get("nuisance") not in (None, "none"):
        f.append("nuisance:" + str(desc.get("nuisance")))
    if judge is not None and not judge.get("supported"):
        f.append("judge_rejected")
    if not f:
        return f, "verified"
    return f, "rejected" if any(x in ic["reject_on"] for x in f) else "uncertain"


def interpret_pair(d, cfg, client):
    vc, ic, pr = cfg["vlm"], cfg["interpret"], cfg["prompts"]
    d = Path(d)
    meta = load_json(d / "pair.json")
    regs = load_json(d / "regions.json")["regions"]
    with rasterio.open(d / "before.tif") as f:
        a, tf, crs = f.read(), f.transform, f.crs
    with rasterio.open(d / "after.tif") as f:
        b = f.read()
    with rasterio.open(d / "prob.tif") as f:
        prob = f.read(1)
    with rasterio.open(d / "labels.tif") as f:
        lab = f.read(1)
    ia, ib = to_display(a, meta["modality"], vc), to_display(b, meta["modality"], vc)
    regs.sort(key=lambda r: r["area_m2"] * r["mean_prob"], reverse=True)
    fmt = dict(modality=meta["modality"], gsd=meta["res_m"], types=ic["change_types"], nuisance=ic["nuisance_types"])
    feats = []
    for n, r in enumerate(regs[: ic["max_regions"]]):
        rng = random.Random(ic["seed"] * 100003 + r["id"])
        win = window(r["bbox_px"], lab.shape, vc)
        cb, ca = chip(ia, win, vc), chip(ib, win, vc)
        ov = chip(ib, win, vc, r["polygon_px"])
        ovb = chip(ia, win, vc, r["polygon_px"])
        bd, ad = meta["before_date"][:10], meta["after_date"][:10]
        ask = lambda x, y, dx, dy, xo, p=pr["describe"]: parse_json(call_vlm(
            client, vc["model"], pr["system"],
            build_content(p.format(before_date=dx, after_date=dy, **fmt),
                          [("BEFORE", x), ("AFTER", y), ("AFTER with region outlined", xo)]), vc["max_tokens"]))
        desc = ask(cb, ca, bd, ad, ov)
        swap = ask(ca, cb, ad, bd, ovb) if rng.random() < ic["swap_rate"] else None
        null = ask(cb, cb, bd, bd, ovb) if rng.random() < ic["null_rate"] else None
        judge = None
        if ic["judge"] and desc:
            claim = json.dumps({k: desc.get(k) for k in ("change_type", "object", "before_state", "after_state", "description")})
            judge = parse_json(call_vlm(
                client, vc["judge_model"], pr["system"],
                build_content(pr["judge"].format(before_date=bd, after_date=ad, claim=claim),
                              [("BEFORE", cb), ("AFTER", ca), ("AFTER with region outlined", ov)]), vc["max_tokens"]))
        evidence = {"mean_prob": r["mean_prob"], "contrast": contrast(prob, lab, r, ic["ring_px"])}
        fails, status = evaluate(desc, swap, null, judge, evidence, ic)
        feats.append({
            "type": "Feature",
            "geometry": r["geometry_lonlat"],
            "properties": {"region_id": r["id"], "area_m2": r["area_m2"], **evidence, "status": status, "failures": fails,
                           "description": desc, "swap_check": swap, "null_check": null, "judge": judge},
        })
        log.info("%s region %s -> %s %s", meta["pair_id"], r["id"], status, fails)
    summary = None
    if ic["summarize"]:
        keep = [{"type": x["properties"]["description"].get("change_type"), "text": x["properties"]["description"].get("description"),
                 "status": x["properties"]["status"]} for x in feats if x["properties"]["status"] != "rejected" and x["properties"]["description"]]
        if keep:
            summary = call_vlm(client, vc["model"], pr["system"], [{"type": "text", "text": pr["summary"].format(
                before_date=meta["before_date"][:10], after_date=meta["after_date"][:10], findings=json.dumps(keep))}], vc["max_tokens"])
    rep = Path(cfg["paths"]["reports"]) / meta["pair_id"]
    rep.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(1 for x in feats if x["properties"]["status"] == s) for s in ("verified", "uncertain", "rejected")}
    dump_json({"type": "FeatureCollection", "features": feats,
               "pair": {k: meta[k] for k in ("pair_id", "source", "modality", "before_id", "after_id", "before_date", "after_date", "gap_days", "license")},
               "counts": counts, "summary": summary}, rep / "report.geojson")
    return counts


def run(cfg, pair_dirs):
    client = anthropic.Anthropic()
    return {str(d): interpret_pair(d, cfg, client) for d in pair_dirs}
