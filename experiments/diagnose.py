"""Error breakdown: FN split by crowding (another GT nucleus within 20 px), FP split by
whether they fall on a nucleus (over-segmentation) or on background."""
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from experiments.data import load_split  # noqa
from experiments.detect import MATCH_R, UNet, detections, predict  # noqa

ROOT = os.path.join(os.path.dirname(__file__), "..")


def breakdown(method, seed, split="test"):
    path = os.path.join(ROOT, "results", "detect", f"{method}_s{seed}")
    r = json.load(open(path + ".json"))
    m = UNet().cuda()
    m.load_state_dict(torch.load(path + ".pt"))
    imgs, gts = load_split(split)
    h, t = r["tuned"]["h"], r["tuned"]["t"]
    c = dict(fn_crowded=0, fn_isolated=0, fp_near=0, fp_far=0, n_crowded=0, n_isolated=0)
    for u, g in zip(predict(m, imgs), gts):
        p = detections(u, h, t)
        dg = np.linalg.norm(g[:, None] - g[None], axis=2) + np.eye(len(g)) * 1e9
        crowded = dg.min(1) < 20
        c["n_crowded"] += int(crowded.sum())
        c["n_isolated"] += int((~crowded).sum())
        d = np.linalg.norm(p[:, None] - g[None], axis=2) if len(p) else np.zeros((0, len(g)))
        dd = d.copy()
        dd[dd > MATCH_R] = 1e6
        i, j = linear_sum_assignment(dd) if len(p) else (np.array([], int), np.array([], int))
        ok = dd[i, j] <= MATCH_R
        mg = np.zeros(len(g), bool)
        mg[j[ok]] = True
        mp = np.zeros(len(p), bool)
        mp[i[ok]] = True
        c["fn_crowded"] += int((~mg & crowded).sum())
        c["fn_isolated"] += int((~mg & ~crowded).sum())
        near = d.min(1) < 15 if len(p) else np.zeros(0, bool)
        c["fp_near"] += int((~mp & near).sum())
        c["fp_far"] += int((~mp & ~near).sum())
    return c


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        meth, seed = spec.rsplit("_s", 1)
        print(meth, seed, breakdown(meth, int(seed)))
