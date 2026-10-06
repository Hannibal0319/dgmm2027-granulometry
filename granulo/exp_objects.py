"""Per-object sieve size on real images.

For each connected object O of the binary image, the sieve size of a family (D_r) is
    s(O) = max{ r : (X o D_r) meets O },
the largest element that fits inside O. The Euclidean reference is the inscribed-disk radius, the
maximum of the Euclidean distance transform inside O. For each object we report the error of s(O)
against the reference, and how much s(O) varies when the image is rotated (objects matched by centroid).
"""
import json
import sys

import numpy as np
from scipy import ndimage as ndi

from all_families import load
from exp_rotation2 import real_scene, open_by, NAMES

ANGLES = list(range(0, 46, 5))


def sizes(X, fam_name, fam, R):
    lab, n = ndi.label(X, structure=np.ones((3, 3)))
    s = np.zeros(n + 1)
    for r in range(1, R + 1):
        O = open_by(fam_name, fam, r, X)
        alive = np.unique(lab[O])
        alive = alive[alive > 0]
        if len(alive) == 0:
            break
        s[alive] = r
    edt = ndi.distance_transform_edt(X)
    ref = np.asarray(ndi.maximum(edt, lab, np.arange(1, n + 1)))
    cen = np.array(ndi.center_of_mass(X, lab, np.arange(1, n + 1)))
    area = np.bincount(lab.ravel())[1:]
    return s[1:], ref, cen, area


def rotate_points(P, theta, shape0, shape1):
    """Map centroids of the unrotated image into the rotated (resize=True) image."""
    t = np.deg2rad(theta)
    c0 = (np.array(shape0) - 1) / 2
    c1 = (np.array(shape1) - 1) / 2
    y, x = P[:, 0] - c0[0], P[:, 1] - c0[1]
    # skimage.transform.rotate rotates counter-clockwise in (row, col) display coordinates
    xr = x * np.cos(t) + y * np.sin(t)
    yr = -x * np.sin(t) + y * np.cos(t)
    return np.stack([yr + c1[0], xr + c1[1]], 1)


def run(scene, R, min_ref):
    fams = load()
    X0 = real_scene(scene, 0)
    out = {}
    for name in [n for n in NAMES if n != 'gauss_sup']:  # sup-closure: same per-object size as gauss
        fam = fams[name][0]
        S, REF = [], []
        base = None
        for th in ANGLES:
            X = real_scene(scene, th)
            s, ref, cen, area = sizes(X, name, fam, R)
            if base is None:
                base = (s, ref, cen, area)
                ids = np.nonzero((ref >= min_ref) & (area > 30) & (ref <= R - 1))[0]  # sieve range must cover the object
                S.append(s[ids]); REF.append(ref[ids])
                continue
            pred = rotate_points(base[2][ids], th, X0.shape, X.shape)
            d = np.linalg.norm(pred[:, None] - cen[None], axis=2)
            j = d.argmin(1)
            ok = d[np.arange(len(ids)), j] < 3
            ss = np.full(len(ids), np.nan); rr = np.full(len(ids), np.nan)
            ss[ok] = s[j[ok]]; rr[ok] = ref[j[ok]]
            S.append(ss); REF.append(rr)
        S = np.array(S); REF = np.array(REF)          # (angles, objects)
        rel_err = np.nanmean(np.abs(S - REF) / REF)
        bias = np.nanmean((S - REF) / REF)
        rot_std = np.nanmean(np.nanstd(S, axis=0) / np.nanmean(REF, axis=0))
        out[name] = dict(rel_err=float(rel_err), bias=float(bias), rot_std=float(rot_std), n=int(S.shape[1]))
        print(f"  {name:9s} objects={S.shape[1]:4d}  mean |s-ref|/ref={rel_err:.4f}  bias={bias:+.4f}  "
              f"rotation std/ref={rot_std:.4f}", flush=True)
    return out


if __name__ == "__main__":
    res = {}
    for scene, R, mr in (("coins", 40, 8), ("gravel", 25, 3), ("mitosis", 15, 3)):
        if len(sys.argv) > 1 and scene not in sys.argv[1:]:
            continue
        print(scene, flush=True)
        res[scene] = run(scene, R, mr)
    json.dump(res, open("exp_objects.json", "w"), indent=1)
