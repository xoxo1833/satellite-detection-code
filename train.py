import random
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from common import ensure, get_logger
from model import SiamUNet

log = get_logger("train")


class PairFolder(Dataset):
    def __init__(self, root, split, dirs, patch, augment):
        base = Path(root) / split
        self.a = sorted((base / dirs["a"]).iterdir())
        self.b = base / dirs["b"]
        self.l = base / dirs["label"]
        self.patch = patch
        self.augment = augment

    def __len__(self):
        return len(self.a)

    def read(self, p, gray=False):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        return img if gray else img[:, :, ::-1]

    def __getitem__(self, i):
        pa = self.a[i]
        xa, xb, y = self.read(pa), self.read(self.b / pa.name), self.read(self.l / pa.name, True)
        h, w = y.shape
        py = random.randint(0, max(h - self.patch, 0))
        px = random.randint(0, max(w - self.patch, 0))
        sl = (slice(py, py + self.patch), slice(px, px + self.patch))
        xa, xb, y = xa[sl], xb[sl], y[sl]
        if self.augment:
            k = random.randint(0, 3)
            xa, xb, y = (np.rot90(t, k) for t in (xa, xb, y))
            if random.random() < 0.5:
                xa, xb, y = (t[:, ::-1] for t in (xa, xb, y))
        f = lambda t: torch.from_numpy(np.ascontiguousarray(t)).permute(2, 0, 1).float() / 255.0
        return f(xa), f(xb), torch.from_numpy((np.ascontiguousarray(y) > 127).astype("float32"))[None]


def dice_loss(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


def evaluate(net, loader, dev):
    tp = fp = fn = 0
    net.eval()
    with torch.no_grad():
        for a, b, y in loader:
            p = (torch.sigmoid(net(a.to(dev), b.to(dev))) > 0.5).cpu()
            tp += int((p & (y > 0.5)).sum())
            fp += int((p & (y <= 0.5)).sum())
            fn += int((~p & (y > 0.5)).sum())
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


def run(cfg):
    t = cfg["train"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    dev = torch.device(t["device"])
    tr = PairFolder(t["root"], t["splits"]["train"], t["dirs"], t["patch"], True)
    va = PairFolder(t["root"], t["splits"]["val"], t["dirs"], t["patch"], False)
    dl = DataLoader(tr, batch_size=t["batch"], shuffle=True, num_workers=t["workers"], drop_last=True)
    vl = DataLoader(va, batch_size=t["batch"], num_workers=t["workers"])
    net = SiamUNet(cfg["model"]["in_channels"], cfg["model"]["widths"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t["epochs"])
    pw = torch.tensor(t["pos_weight"], device=dev)
    best = -1
    ensure(Path(t["out"]).parent)
    for ep in range(t["epochs"]):
        net.train()
        tot = 0
        for a, b, y in dl:
            a, b, y = a.to(dev), b.to(dev), y.to(dev)
            out = net(a, b)
            loss = F.binary_cross_entropy_with_logits(out, y, pos_weight=pw) + t["dice_weight"] * dice_loss(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        f1, p, r = evaluate(net, vl, dev)
        log.info("epoch %d loss %.4f val f1 %.4f p %.4f r %.4f", ep, tot / len(dl), f1, p, r)
        if f1 > best:
            best = f1
            torch.save({"state": net.state_dict(), "model": cfg["model"], "f1": f1}, t["out"])
    log.info("best val f1 %.4f -> %s", best, t["out"])
