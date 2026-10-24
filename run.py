import argparse
from common import load_config
import detect
import discover
import export_pair
import interpret
import pairing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["discover", "pair", "detect", "export", "interpret", "dataset", "train", "all"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--pair", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s = args.stage
    if s in ("discover", "dataset", "all"):
        discover.run(cfg)
    if s in ("pair", "dataset", "all"):
        pairing.run(cfg)
    if s in ("detect", "dataset", "all"):
        detect.run(cfg, pairing.accepted_dirs(cfg, args.pair))
    if s in ("export", "dataset", "all"):
        export_pair.run(cfg, pairing.accepted_dirs(cfg, args.pair))
    if s in ("interpret", "all"):
        interpret.run(cfg, pairing.accepted_dirs(cfg, args.pair))
    if s == "train":
        import train
        train.run(cfg)


if __name__ == "__main__":
    main()
