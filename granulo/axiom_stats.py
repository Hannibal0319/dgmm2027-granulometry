"""E1: how often do standard digital disks violate 'D_r is D_s-open' (s < r)?"""
import numpy as np, cv2, json
from scipy import ndimage as ndi
from skimage.morphology import disk

def is_open(B, A):
    pad = A.shape[0]; Bp = np.pad(B, pad).astype(bool)
    return np.array_equal(ndi.binary_opening(Bp, structure=A.astype(bool)), Bp)

def cv_disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)).astype(bool)

R = 40
res = {}
for name, f in (("skimage.disk", lambda r: disk(r).astype(bool)), ("OpenCV ellipse", cv_disk)):
    V = np.zeros((R + 1, R + 1), bool)
    for r in range(2, R + 1):
        Br = f(r)
        for s in range(1, r):
            V[r, s] = not is_open(Br, f(s))
    n = R * (R - 1) // 2 - 0
    tot = sum(1 for r in range(2, R + 1) for s in range(1, r))
    viol = int(V.sum())
    consec = int(sum(V[r, r - 1] for r in range(2, R + 1)))
    res[name] = dict(pairs=tot, violations=viol, frac=viol / tot, consecutive_violations=consec)
    np.save(f"violations_{name.split('.')[0].split()[0]}.npy", V)
    print(name, res[name], flush=True)
json.dump(res, open("axiom_stats.json", "w"), indent=1)
