import json
import pandas as pd
import pystac
from pystac_client import Client
from shapely.geometry import box, shape
from common import ensure, get_logger

log = get_logger("discover")


def normalize(item, src):
    p = item.properties
    when = item.datetime or item.common_metadata.start_datetime
    assets = {k: item.assets[k].href for k in src.get("assets", []) if k in item.assets}
    if src.get("mask_asset") and src["mask_asset"] in item.assets:
        assets["_mask"] = item.assets[src["mask_asset"]].href
    return {
        "id": item.id,
        "source": src["name"],
        "modality": src["modality"],
        "collection": item.collection_id,
        "datetime": when.isoformat() if when else None,
        "platform": p.get("platform"),
        "gsd": p.get("gsd", src.get("gsd")),
        "cloud": p.get("eo:cloud_cover"),
        "off_nadir": p.get("view:off_nadir"),
        "sun_elev": p.get("view:sun_elevation"),
        "orbit_state": p.get("sat:orbit_state"),
        "license": p.get("license"),
        "geometry": json.dumps(item.geometry),
        "assets": json.dumps(assets),
        "needed": len(assets) >= len(src.get("assets", [])),
    }


def search_api(src, cfg, dt):
    client = Client.open(src["url"])
    res = client.search(
        collections=src["collections"],
        bbox=cfg["aoi"]["bbox"],
        datetime=dt,
        query=src.get("query"),
        max_items=src["max_items"],
    )
    return list(res.items())


def search_static(src, cfg, dt):
    cat = pystac.Catalog.from_file(src["url"])
    aoi = box(*cfg["aoi"]["bbox"])
    start, end = (pd.to_datetime(x, utc=True) for x in dt.split("/"))
    out, scanned = [], 0
    for item in cat.get_items(recursive=True):
        scanned += 1
        if scanned > src["max_scan"] or len(out) >= src["max_items"]:
            break
        if item.geometry is None or not shape(item.geometry).intersects(aoi):
            continue
        dt = item.datetime or (pd.to_datetime(item.properties.get("start_datetime"), utc=True).to_pydatetime())
        if dt is None or not (start <= pd.Timestamp(dt).tz_convert("UTC") <= end):
            continue
        out.append(item)
    return out


def search_ranges(cfg):
    win = cfg["pairing"].get("windows")
    return list(win.values()) if win else [cfg["aoi"]["datetime"]]


def run(cfg):
    rows = []
    for src in cfg["sources"]:
        fn = search_api if src["type"] == "api" else search_static
        for dt in search_ranges(cfg):
            try:
                items = fn(src, cfg, dt)
            except Exception as e:
                log.error("source %s %s failed: %s", src["name"], dt, e)
                continue
            log.info("%s %s: %d items", src["name"], dt, len(items))
            rows += [normalize(i, src) for i in items]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["needed"]].drop(columns="needed").drop_duplicates(["source", "id"])
    ensure(cfg["paths"]["work"])
    df.to_csv(cfg["paths"]["catalog"], index=False)
    log.info("catalog: %d items -> %s", len(df), cfg["paths"]["catalog"])
    return df
