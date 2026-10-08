"""3D: digital balls B_r = {p in Z^3 : |p| <= r} (= skimage.morphology.ball) and size distributions of random
sphere packings, computed with single ball openings and with their sup-closure.

Openings are computed exactly with Euclidean distance transforms: X - B_r = {EDT(X) > r} and
Y + B_r = {EDT(not Y) <= r}; this equals ndimage.binary_opening with ball(r) (checked for r = 2, 3, 5)."""
import numpy as np
from scipy import ndimage as ndi


def opening(X, r):
    P = np.pad(X, r + 1)
    E = ndi.distance_transform_edt(P) > r
    O = ndi.distance_transform_edt(~E) <= r
    return O[r + 1:-r - 1, r + 1:-r - 1, r + 1:-r - 1]


def ball(r):
    k = int(r)
    z, y, x = np.ogrid[-k:k + 1, -k:k + 1, -k:k + 1]
    return z * z + y * y + x * x <= r * r


# (1) absorption between digital balls
R = 15
bad = cons = 0
for r in range(2, R + 1):
    Br = ball(r)
    for s in range(1, r):
        if (opening(Br, s) != Br).any():
            bad += 1
            cons += s == r - 1
print(f"3D balls: {bad}/{R * (R - 1) // 2} pairs (s<r<={R}) violate absorption, {cons}/{R - 1} consecutive")

# (2) random packings of overlapping Euclidean balls
rng = np.random.default_rng(1)
n, Rmax, trials = 80, 12, 100
single = sup = 0
neg = []
for t in range(trials):
    X = np.zeros((n, n, n), bool)
    zz, yy, xx = np.ogrid[:n, :n, :n]
    for _ in range(rng.integers(20, 61)):
        c = rng.uniform(0, n, 3)
        rr = rng.uniform(3, 12)
        X |= (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= rr ** 2
    O = [opening(X, r) for r in range(1, Rmax + 1)]
    a = np.array([X.sum()] + [o.sum() for o in O], dtype=float)
    acc = np.zeros_like(X)
    b = [0.0] * Rmax
    for i in range(Rmax - 1, -1, -1):
        acc |= O[i]
        b[i] = float(acc.sum())
    b = np.array([X.sum()] + b)
    ps = -np.diff(a)
    if (ps < 0).any():
        single += 1
        neg.append(-ps[ps < 0].sum() / ps[ps > 0].sum())
    sup += bool((np.diff(b) > 0).any())
print(f"3D packings: non-monotone in {single}/{trials} (single openings), {sup}/{trials} (sup-closure)")
if neg:
    print(f"  negative / positive pattern-spectrum mass: median {np.median(neg):.4f}, max {max(neg):.4f}")
