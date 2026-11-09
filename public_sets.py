import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from common import load_json
from pair_io import to_u8


def read_rgb(path, pct, tile=None, rng=None, origin=None):
    with rasterio.open(str(path)) as s:
        win = None
        if tile and (s.height > tile or s.width > tile):
            if origin is None:
                origin = (rng.randint(0, max(s.height - tile, 0)), rng.randint(0, max(s.width - tile, 0)))
            win = Window(origin[1], origin[0], min(tile, s.width - origin[1]), min(tile, s.height - origin[0]))
        arr = s.read(window=win)
    arr = arr[:3] if arr.shape[0] >= 3 else np.repeat(arr[:1], 3, axis=0)
    return to_u8(arr.transpose(1, 2, 0), pct), origin


def pick_even(items, count):
    step = max(len(items) // count, 1)
    return items[::step][:count]


def iso8(s):
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def levir_hf(c, pct):
    from datasets import load_dataset
    ds = load_dataset(c["hf_repo"], split=c["hf_split"], streaming=True)
    for i, row in enumerate(ds.take(c["count"])):
        yield {"rel": f"{c['hf_split']}_{i:04d}", "before": np.array(row["image1"].convert("RGB")),
               "after": np.array(row["image2"].convert("RGB")), "meta": {}}


def paired_folders(c, pct):
    bd, ad = Path(c["before_dir"]), Path(c["after_dir"])
    if not bd.is_dir() or not ad.is_dir():
        raise FileNotFoundError(f"folder not found: {bd if not bd.is_dir() else ad}")
    names = sorted(p.name for p in bd.glob(c["pattern"]) if (ad / p.name).exists())
    if not names:
        raise FileNotFoundError(f"no file name matching '{c['pattern']}' exists in both {bd} and {ad}")
    rng = random.Random(c["seed"])
    for name in pick_even(names, c["count"]):
        a, origin = read_rgb(bd / name, pct, c.get("tile_px"), rng)
        b, _ = read_rgb(ad / name, pct, c.get("tile_px"), None, origin)
        yield {"rel": Path(name).stem, "before": a, "after": b, "meta": {"source_file": name}}


def xbd_local(c, pct):
    root = Path(c["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"folder not found: {root}")
    pres = sorted(root.rglob("*_pre_disaster.png"))
    if not pres:
        raise FileNotFoundError(f"no *_pre_disaster.png files found under {root}")
    img_dir = pres[0].parent
    lab_dir = img_dir.parent / "labels"
    pres = sorted(img_dir.glob("*_pre_disaster.png"))
    groups = {}
    for p in pres:
        groups.setdefault(p.name.split("_")[0], []).append(p)
    for disaster in sorted(groups)[: c["max_disasters"]]:
        for pre in pick_even(groups[disaster], c["per_disaster"]):
            post = pre.with_name(pre.name.replace("_pre_disaster", "_post_disaster"))
            if not post.exists():
                continue
            dates = []
            for p in (pre, post):
                lj = lab_dir / (p.stem + ".json")
                cd = load_json(lj).get("metadata", {}).get("capture_date") if lj.exists() else None
                dates.append(str(cd)[:10] if cd else None)
            a, _ = read_rgb(pre, pct)
            b, _ = read_rgb(post, pct)
            yield {"rel": f"{disaster}/{pre.name.replace('_pre_disaster.png', '')}", "before": a, "after": b,
                   "meta": {"before_date": dates[0], "after_date": dates[1], "disaster": disaster}}


def oscd_local(c, pct):
    root = Path(c["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"folder not found: {root}")
    d1, d2 = c["dirs"]
    cities = sorted({p.parent for p in root.rglob(d1) if p.is_dir() and (p.parent / d2).is_dir()})
    if not cities:
        raise FileNotFoundError(f"no folder containing both {d1} and {d2} found under {root}")
    for city in pick_even(cities, c["count"]):
        imgs = []
        for sub in (d1, d2):
            bands = [rasterio.open(str(city / sub / f"{b}.tif")).read(1) for b in c["bands"]]
            imgs.append(to_u8(np.stack(bands, axis=-1), pct))
        dt = (city / c["dates_file"]).read_text() if (city / c["dates_file"]).exists() else ""
        found = re.findall(r"\d{8}", dt)
        yield {"rel": city.name, "before": imgs[0], "after": imgs[1],
               "meta": {"before_date": iso8(found[0]) if len(found) > 1 else None,
                        "after_date": iso8(found[1]) if len(found) > 1 else None, "city": city.name}}


def s3_list(base, prefix, delimiter=None):
    url = f"{base}/?list-type=2&prefix={prefix}" + (f"&delimiter={delimiter}" if delimiter else "")
    with urllib.request.urlopen(url, timeout=60) as r:
        root = ET.fromstring(r.read())
    ns = {"s": root.tag.split("}")[0].strip("{")}
    return ([e.text for e in root.findall("s:Contents/s:Key", ns)],
            [e.text for e in root.findall("s:CommonPrefixes/s:Prefix", ns)])


def spacenet7_s3(c, pct):
    _, prefixes = s3_list(c["s3_http"], c["train_prefix"], "/")
    aois = sorted(prefixes)[: c["aoi_count"]]
    if not aois:
        raise RuntimeError(f"no AOI folder listed under {c['train_prefix']}")
    for aoi in aois:
        name = aoi.rstrip("/").split("/")[-1]
        keys, _ = s3_list(c["s3_http"], f"{aoi}images/")
        dated = sorted((re.search(r"global_monthly_(\d{4})_(\d{2})", k).groups(), k) for k in keys
                       if k.endswith(".tif") and re.search(r"global_monthly_(\d{4})_(\d{2})", k))
        if len(dated) < 2:
            continue
        months = lambda ym: int(ym[0]) * 12 + int(ym[1])
        first = dated[0]
        later = [d for d in dated if months(d[0]) - months(first[0]) >= c["min_gap_months"]]
        if not later:
            continue
        last = later[-1]
        a, _ = read_rgb(f"{c['s3_http']}/{first[1]}", pct)
        b, _ = read_rgb(f"{c['s3_http']}/{last[1]}", pct)
        yield {"rel": name, "before": a, "after": b,
               "meta": {"before_date": "-".join(first[0]), "after_date": "-".join(last[0]), "aoi": name}}
