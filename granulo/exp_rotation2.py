"""Rotation invariance and monotonicity of size distributions for all families, on a synthetic scene
(rotated squares, exact rasterisation) and on two real images (rotated, then thresholded).

S = mean_r std_theta F_theta(r),  F_theta(r) = |X_theta o D_r| / |X_theta|;
violations = #(theta, r) with F_theta(r+1) > F_theta(r) (impossible for a granulometry)."""
import json
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage import data, filters, transform
from skimage.morphology import disk

from all_families import load
from exp_rotation import scene
from families import open_cascade

ANGLES = list(range(0, 46, 5))
NAMES = ["gauss", "octagon", "adams", "periodic", "ours", "dss_milp"]


def real_scene(name, theta):
    if name == "mitosis":
        img = filters.gaussian(data.human_mitosis().astype(float), 1.0)
    elif name == "gravel":
        img = filters.gaussian(data.gravel().astype(float), 1.5)
    else:
        img = filters.gaussian(data.coins().astype(float), 1.0)
    thr = filters.threshold_otsu(img)
    rot = transform.rotate(img, theta, resize=True, order=1, mode="constant", cval=float(img.min()))
    X = rot > thr
    if name == "coins":
        X = ndi.binary_fill_holes(X)
    return X


def open_by(fam_name, fam, r, X):
    if fam_name == "gauss":
        return ndi.binary_opening(X, structure=disk(r).astype(bool))
    return open_cascade(X, fam[r])


def run(scene_fn, R, fams):
    out = {}
    Xs = [scene_fn(t) for t in ANGLES]
    for name in NAMES:
        fam = fams[name][0]
        F = np.zeros((len(ANGLES), R + 1))
        for i, X in enumerate(Xs):
            A0 = X.sum()
            F[i, 0] = 1
            for r in range(1, R + 1):
                F[i, r] = open_by(name, fam, r, X).sum() / A0
        out[name] = dict(S=float(F.std(0).mean()), violations=int((np.diff(F, axis=1) > 1e-12).sum()), F=F.tolist())
        print(f"  {name:9s} S={out[name]['S']:.4f}  violations={out[name]['violations']}", flush=True)
    return out


if __name__ == "__main__":
    fams = load()
    which = sys.argv[1:] or ["squares", "mitosis", "coins"]
    res = {}
    for w in which:
        print(w, flush=True)
        if w == "squares":
            res[w] = run(scene, 40, fams)
        elif w == "gravel":
            res[w] = run(lambda t: real_scene("gravel", t), 20, fams)
        elif w == "mitosis":
            res[w] = run(lambda t: real_scene("mitosis", t), 15, fams)
        else:
            res[w] = run(lambda t: real_scene("coins", t), 30, fams)
        json.dump(res, open(f"exp_rotation2_{'_'.join(which)}.json", "w"))
