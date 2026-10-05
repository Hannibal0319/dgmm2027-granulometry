"""E3: nucleus detection on BBBC039 trained end-to-end through an h-dome layer.

Methods (same U-Net, same Gaussian target T, same budget):
  heatmap      : MSE(u, T); inference = h-maxima of u with (h, t) tuned on validation
  hdome        : MSE(D_1(u), T), D_1 = u - R_u(u - 1) with the exact provenance backward
  hdome_K{k}   : same loss, reconstruction truncated to k unrolled geodesic dilations
  hybrid       : MSE(u, T) + MSE(D_1(u), T)
Usage: python experiments/detect.py METHOD SEED
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.morphology import local_maxima

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data import load_split  # noqa
from geodrec import reconstruction_by_dilation, reconstruction_unrolled  # noqa

DEV = "cuda"
ROOT = os.path.join(os.path.dirname(__file__), "..")
SIGMA, CROP, BATCH, STEPS, LR, MATCH_R = 3.0, 256, 8, 4000, 2e-3, 10.0
H_TRAIN = 1.0


# ----------------------------------------------------------------------------- model
def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.d1, self.d2, self.d3, self.b = block(1, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.u3, self.u2, self.u1 = block(12 * c, 4 * c), block(6 * c, 2 * c), block(3 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        up = lambda a: F.interpolate(a, scale_factor=2, mode="bilinear", align_corners=False)
        x1 = self.d1(x)
        x2 = self.d2(F.max_pool2d(x1, 2))
        x3 = self.d3(F.max_pool2d(x2, 2))
        b = self.b(F.max_pool2d(x3, 2))
        y = self.u3(torch.cat([up(b), x3], 1))
        y = self.u2(torch.cat([up(y), x2], 1))
        y = self.u1(torch.cat([up(y), x1], 1))
        return self.out(y)


# ----------------------------------------------------------------------------- data
def gaussian_target(shape, pts, sigma=SIGMA):
    """max of unit-height Gaussians centred at the nucleus centroids."""
    T = np.zeros(shape, np.float32)
    r = int(3 * sigma)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    for cy, cx in pts:
        iy, ix = int(round(cy)), int(round(cx))
        k = np.exp(-((yy + iy - cy) ** 2 + (xx + ix - cx) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        y0, y1, x0, x1 = iy - r, iy + r + 1, ix - r, ix + r + 1
        ky0, kx0 = max(0, -y0), max(0, -x0)
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(shape[0], y1), min(shape[1], x1)
        if y1 > y0 and x1 > x0:
            T[y0:y1, x0:x1] = np.maximum(T[y0:y1, x0:x1], k[ky0:ky0 + y1 - y0, kx0:kx0 + x1 - x0])
    return T


class Sampler:
    def __init__(self, split, rng):
        self.imgs, self.pts = load_split(split)
        self.T = [gaussian_target(im.shape, p) for im, p in zip(self.imgs, self.pts)]
        self.rng = rng

    def batch(self):
        xs, ts = [], []
        for _ in range(BATCH):
            i = self.rng.integers(len(self.imgs))
            im, T = self.imgs[i], self.T[i]
            y = self.rng.integers(im.shape[0] - CROP + 1)
            x = self.rng.integers(im.shape[1] - CROP + 1)
            a, b = im[y:y + CROP, x:x + CROP], T[y:y + CROP, x:x + CROP]
            k = self.rng.integers(4)
            a, b = np.rot90(a, k), np.rot90(b, k)
            if self.rng.random() < 0.5:
                a, b = a[:, ::-1], b[:, ::-1]
            xs.append(a.copy())
            ts.append(b.copy())
        to = lambda z: torch.tensor(np.stack(z))[:, None].to(DEV)
        return to(xs), to(ts)


# ----------------------------------------------------------------------------- losses
def hdome_layer(u, h, method):
    if method.startswith("hdome_K"):
        K = int(method[len("hdome_K"):])
        return u - reconstruction_unrolled(u - h, u, n_iter=K)
    return u - reconstruction_by_dilation(u - h, u)


def loss_fn(method, u, T):
    if method == "heatmap":
        return F.mse_loss(u, T)
    if method == "hybrid":
        return F.mse_loss(u, T) + F.mse_loss(hdome_layer(u, H_TRAIN, "hdome"), T)
    return F.mse_loss(hdome_layer(u, H_TRAIN, method), T)


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def maxima(u, h):
    """h-maxima of u (8-connectivity): centroids and peak heights of the regional
    maxima of R_u(u - h).  Reconstruction on GPU, plateau-aware maxima on CPU."""
    r = reconstruction_by_dilation(u - h, u)[0, 0].cpu().numpy()
    reg = local_maxima(r, connectivity=2, allow_borders=True)
    lab, k = ndi.label(reg, structure=np.ones((3, 3)))
    if k == 0:
        return np.zeros((0, 2)), np.zeros(0)
    ids = np.arange(1, k + 1)
    pts = np.array(ndi.center_of_mass(reg, lab, ids)).reshape(-1, 2)
    peak = np.asarray(ndi.maximum(u[0, 0].cpu().numpy(), lab, ids))
    return pts, peak


def detections(u, h, t=None):
    pts, peak = maxima(u, h)
    return pts if t is None else pts[peak >= t]


def match(pred, gt, r=MATCH_R):
    if len(pred) == 0 or len(gt) == 0:
        return 0
    d = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    d[d > r] = 1e6
    i, j = linear_sum_assignment(d)
    return int((d[i, j] <= r).sum())


def scores(preds, gts):
    tp = sum(match(p, g) for p, g in zip(preds, gts))
    npred, ngt = sum(len(p) for p in preds), sum(len(g) for g in gts)
    P = tp / max(npred, 1)
    R = tp / max(ngt, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)
    mae = float(np.mean([abs(len(p) - len(g)) for p, g in zip(preds, gts)]))
    return dict(P=P, R=R, F1=F1, MAE=mae)


@torch.no_grad()
def predict(model, imgs):
    model.eval()
    out = [model(torch.tensor(im)[None, None].to(DEV)) for im in imgs]
    model.train()
    return out


def tune_and_test(method, model):
    vimg, vpts = load_split("validation")
    timg, tpts = load_split("test")
    uv, ut = predict(model, vimg), predict(model, timg)
    hs = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5) if method == "heatmap" else (0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3)
    ts = (None, 0.1, 0.2, 0.3, 0.4, 0.5) if method == "heatmap" else (None,)
    res = {}
    if method != "heatmap":
        # parameter-free rule: a detected dome must reach half the target height
        res["default"] = scores([detections(u, H_TRAIN / 2) for u in ut], tpts)
        res["default"]["h"] = H_TRAIN / 2
    best = None
    for h in hs:
        mv = [maxima(u, h) for u in uv]
        for t in ts:
            preds = [p if t is None else p[pk >= t] for p, pk in mv]
            s = scores(preds, vpts)
            if best is None or s["F1"] > best[0]["F1"]:
                best = (s, h, t)
    _, h, t = best
    res["tuned"] = scores([detections(u, h, t) for u in ut], tpts)
    res["tuned"].update(h=h, t=t, val_F1=best[0]["F1"])
    return res, ut


def main(method, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = Sampler("training", rng)
    model = UNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS, pct_start=0.1)
    torch.cuda.reset_peak_memory_stats()
    log, t0 = [], time.time()
    for step in range(STEPS):
        x, T = sampler.batch()
        loss = loss_fn(method, model(x), T)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            log.append((step, loss.item()))
            print(method, seed, step, f"{loss.item():.5f}", f"{time.time() - t0:.0f}s", flush=True)
    train_time = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    res, ut = tune_and_test(method, model)
    res.update(method=method, seed=seed, train_time_s=train_time, peak_MB=peak, log=log)
    os.makedirs(os.path.join(ROOT, "results", "detect"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "detect", f"{method}_s{seed}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    torch.save(model.state_dict(), os.path.join(ROOT, "results", "detect", f"{method}_s{seed}.pt"))
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


def reevaluate(method, seed):
    """Re-run tuning / testing from a saved checkpoint, keeping the training stats."""
    path = os.path.join(ROOT, "results", "detect", f"{method}_s{seed}")
    model = UNet().to(DEV)
    model.load_state_dict(torch.load(path + ".pt"))
    old = json.load(open(path + ".json"))
    res, _ = tune_and_test(method, model)
    old.update(res)
    json.dump(old, open(path + ".json", "w"), indent=1)
    print(json.dumps({k: v for k, v in old.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "eval":
        reevaluate(sys.argv[2], int(sys.argv[3]))
    else:
        main(sys.argv[1], int(sys.argv[2]))
