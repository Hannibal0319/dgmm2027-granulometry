"""Fig. 3: qualitative detection results on a BBBC039 test crop (seed 0 models)."""
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data import load_split  # noqa
from experiments.detect import H_TRAIN, UNet, detections  # noqa
from geodrec import hdome  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
IMG, Y0, X0, S = int(sys.argv[1]) if len(sys.argv) > 1 else 7, 150, 250, 220


def load(method):
    m = UNet().cuda()
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", "detect", f"{method}_s0.pt")))
    m.eval()
    r = json.load(open(os.path.join(ROOT, "results", "detect", f"{method}_s0.json")))
    return m, r


timg, tpts = load_split("test")
im, gt = timg[IMG], tpts[IMG]
x = torch.tensor(im)[None, None].cuda()
mh, rh = load("heatmap")
md, rd = load("hdome")
with torch.no_grad():
    uh, ud = mh(x), md(x)
    D = hdome(ud, H_TRAIN)
ph = detections(uh, rh["tuned"]["h"], rh["tuned"]["t"])
pd_ = detections(ud, H_TRAIN)


def crop(a):
    return a[Y0:Y0 + S, X0:X0 + S]


def pts_in(p):
    k = (p[:, 0] >= Y0) & (p[:, 0] < Y0 + S) & (p[:, 1] >= X0) & (p[:, 1] < X0 + S)
    return p[k] - [Y0, X0]


fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.0))
panels = [(crop(im), "image + ground truth", None, "gray"),
          (crop(uh[0, 0].cpu().numpy()), f"heatmap $u$, $h$-max. ($h$={rh['tuned']['h']})", ph, "magma"),
          (crop(ud[0, 0].cpu().numpy()), "ours: $u$ (unconstrained)", None, "magma"),
          (crop(D[0, 0].cpu().numpy()), r"ours: $D_1(u)$, 1-maxima", pd_, "magma")]
g = pts_in(gt)
for a, (img, title, pred, cm) in zip(ax, panels):
    a.imshow(img, cmap=cm)
    a.plot(g[:, 1], g[:, 0], "o", mfc="none", mec="lime", ms=5, mew=0.8)
    if pred is not None:
        q = pts_in(pred)
        a.plot(q[:, 1], q[:, 0], "x", color="cyan", ms=4, mew=1.0)
    a.set_title(title, fontsize=7)
    a.set_xticks([])
    a.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "detect.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "detect.png"), bbox_inches="tight", dpi=200)
print("heatmap dets", len(ph), "ours", len(pd_), "gt", len(gt))
