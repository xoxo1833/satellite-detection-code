import torch
import torch.nn as nn


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class SiamUNet(nn.Module):
    def __init__(self, in_ch, widths):
        super().__init__()
        chans = [in_ch] + list(widths)
        n = len(widths)
        self.enc = nn.ModuleList([block(chans[i], chans[i + 1]) for i in range(n)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2) for i in range(n - 1, 0, -1)])
        self.dec = nn.ModuleList([block(widths[i - 1] * 2, widths[i - 1]) for i in range(n - 1, 0, -1)])
        self.head = nn.Conv2d(widths[0], 1, 1)
        self.stride = 2 ** (n - 1)

    def encode(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats

    def forward(self, a, b):
        fa, fb = self.encode(a), self.encode(b)
        d = [(x - y).abs() for x, y in zip(fa, fb)]
        x = d[-1]
        for k, (u, dec) in enumerate(zip(self.up, self.dec)):
            x = dec(torch.cat([u(x), d[len(d) - 2 - k]], 1))
        return self.head(x)
