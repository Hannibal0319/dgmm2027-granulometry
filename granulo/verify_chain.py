"""Verify for every element of a chain2 family: (1) hole-free: element == conv(element) cap Z^2;
(2) granulometric: element_r is element_{r-1}-open (true by construction, re-checked);
(3) report per-r roundness (free centre) next to octagons and Gauss disks on the same metric."""
import sys, numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
from skimage.morphology import disk
from families import element_mask
from dither_chain import cheb_disk, U as U2

def hull_points(M):
    ys, xs = np.nonzero(M); P = np.stack([xs, ys], 1).astype(float)
    h = ConvexHull(P); E = h.equations
    yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]
    Q = np.stack([xx.ravel(), yy.ravel()], 1)
    return np.all(Q @ E[:, :2].T + E[:, 2] <= 1e-9, axis=1).reshape(M.shape)

def is_open(B, A):
    pad = max(A.shape); Bp = np.pad(B, pad)
    return np.array_equal(ndi.binary_opening(Bp, structure=A), Bp)

def eps_of_mask(M):
    ys, xs = np.nonzero(M); P = np.stack([xs, ys], 1).astype(float)
    return cheb_disk((U2 @ P.T).max(1))

d = np.load(sys.argv[1], allow_pickle=True)
C, sets = d["C"], list(d["sets"])
holes = 0; notopen = 0; prev = None; rows = []
for r, c in enumerate(C, 1):
    summ = []
    for S, k in zip(sets, c): summ += [list(map(tuple, np.asarray(S, int)))] * int(k)
    M = element_mask(summ)
    holes += not np.array_equal(M, hull_points(M))
    if prev is not None and not is_open(M, prev): notopen += 1
    prev = M
    e, rho = eps_of_mask(M)
    eg, rg = eps_of_mask(disk(r).astype(bool))
    rows.append((r, rho, e, rg, eg))
print("elements:", len(C), " with holes:", holes, " consecutive pairs not open:", notopen)
rows = np.array(rows); np.save(sys.argv[1].replace(".npz", "_eval.npy"), rows)
for r in (4, 8, 12, 16, 20, 24, 32, 40, 48):
    if r <= len(rows):
        x = rows[r - 1]; print(f"r={r:2d}  ours rho={x[1]:6.2f} eps={x[2]:.3f}   gauss eps={x[4]:.3f}")
print("ours max eps:", rows[:, 2].max().round(3), " mean:", rows[:, 2].mean().round(3), " max step:", np.diff(rows[:, 1]).max().round(3))
