import json
import logging
from pathlib import Path
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def ensure(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def get_logger(name):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)
