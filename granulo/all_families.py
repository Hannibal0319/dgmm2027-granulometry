"""All structuring-element families used in the experiments, r = 1..R (cached in families_R48.pkl).

- gauss    : Gauss digital disks {|p| <= r} (scikit-image disk) -- not granulometric
- octagon  : Minkowski chain of N4/N8-type steps (D4-orbits of the unit and diagonal segments)
- adams    : Minkowski chain of 8-connected Bresenham segments in the 8 directions of
             (1,0),(1,1),(2,1) and their D4 images (radial decomposition in the spirit of Adams [1])
- periodic : Minkowski chain of periodic lines {0, v} in the same 8 directions + the 2x2 square
             (cascades of periodic lines in the spirit of Jones & Soille)
- ours     : LP-weighted periodic-line chain, 13 directions (D4 images of (1,0),(1,1),(2,1),(3,1)) + square
- dss_milp : unit-step MILP chain of 8-connected segments (chain2_M6_R48.npz)
For the fixed libraries the multiplicities are round(t * y) with y the optimal LP weights (Corollary 2)
and t increasing, so every family is a monotone Minkowski chain; for each integer r the element whose
fitted radius is closest to r is used.
"""
import math
import os
import pickle

import numpy as np
from scipy.optimize import linprog
from skimage.morphology import disk

from dither_chain import cheb_disk, U as U2
from families import element_mask
from geom import Generator

R = 48
CACHE = os.path.join(os.path.dirname(__file__), f"families_R{R}.pkl")
TH = np.linspace(0, math.pi, 3600, endpoint=False)
UW = np.stack([np.cos(TH), np.sin(TH)], 1)


def width(S):
    z = UW @ np.asarray(S, float).T
    return z.max(1) - z.min(1)


def lp_weights(sets):
    Wm = np.stack([width(S) for S in sets], 1)
    n = Wm.shape[1]
    c = np.r_[np.zeros(n), 1.0]
    A = np.vstack([np.hstack([-Wm, np.zeros((len(TH), 1))]), np.hstack([Wm, -4 * np.ones((len(TH), 1))])])
    b = np.r_[-2 * np.ones(len(TH)), 2 * np.ones(len(TH))]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * (n + 1), method="highs")
    return res.x[:n], res.x[-1]


def eps_rho(points):
    P = np.asarray(points, float)
    return cheb_disk((U2 @ P.T).max(1))


def mask_points(M):
    ys, xs = np.nonzero(M)
    return np.stack([xs, ys], 1)


def library_chain(sets, base, t_step=0.05):
    """Monotone chain base (+) sum_g n_g sets_g following n = round(t y) (y: optimal LP weights).
    Whenever the target counts increase, the new summands are added one at a time (round-robin over
    generators), so consecutive elements differ by a single segment (small radius steps)."""
    y, kappa = lp_weights(sets)
    hs = np.stack([(U2 @ np.asarray(S, float).T).max(1) for S in sets], 1)
    h_base = sum((U2 @ np.asarray(S, float).T).max(1) for S in base) if base else np.zeros(len(U2))
    cur = np.zeros(len(sets), int)
    elems = []  # (rho, eps, counts)
    t = 0.0
    while True:
        target = np.round(t * y).astype(int)
        while (target > cur).any():
            # greedy: add the single summand that keeps the element roundest
            best = None
            for i in np.nonzero(target > cur)[0]:
                cur[i] += 1
                e, rho = cheb_disk(h_base + hs @ cur)
                cur[i] -= 1
                if best is None or e < best[0]:
                    best = (e, rho, i)
            cur[best[2]] += 1
            elems.append((best[1], best[0], cur.copy()))
        if elems and elems[-1][0] > R + 1:
            break
        t += t_step
    rhos = np.array([x[0] for x in elems])
    fam, info = {}, {}
    last = -1
    for r in range(1, R + 1):
        j = max(int(np.argmin(np.abs(rhos - r))), last)   # keep the chain order
        last = j
        summ = list(base)
        for S, k in zip(sets, elems[j][2]):
            summ += [S] * int(k)
        fam[r] = summ
        e, rho = eps_rho(mask_points(element_mask(summ)))
        info[r] = (rho, e)
    return fam, info, kappa


def build():
    sq = [[(0, 0), (1, 0)], [(0, 0), (0, 1)]]
    out = {}
    # Gauss disks
    fam, info = {}, {}
    for r in range(1, R + 1):
        D = disk(r).astype(bool)
        ys, xs = np.nonzero(D)
        fam[r] = [list(zip(xs - r, ys - r))]
        info[r] = eps_rho(mask_points(D))[::-1][::-1]
        e, rho = eps_rho(mask_points(D))
        info[r] = (rho, e)
    out["gauss"] = (fam, info, None)
    keys8 = [(1, 0), (1, 1), (2, 1)]
    dss = [list(map(tuple, S)) for k in keys8 for S in Generator(*k).sets]
    oct_sets = [list(map(tuple, S)) for k in [(1, 0), (1, 1)] for S in Generator(*k).sets]
    # octagons: best monotone (n_(1,0), n_(1,1)) per radius (stronger than LP rounding)
    from exp_rotation import octagon_chain_counts
    k8, C8 = octagon_chain_counts(R)
    fam, info = {}, {}
    for r in range(1, R + 1):
        summ = []
        for k, c in zip(k8, C8[r - 1]):
            for S in Generator(*k).sets:
                summ += [list(map(tuple, S))] * int(c)
        fam[r] = summ
        e, rho = eps_rho(mask_points(element_mask(summ)))
        info[r] = (rho, e)
    out["octagon"] = (fam, info, lp_weights(oct_sets)[1])
    out["adams"] = library_chain(dss, [])
    dirs = set()
    for a, b in keys8:
        for v in {(a, b), (b, a), (-a, b), (-b, a)}:
            dirs.add(v)
    per = [[(0, 0), v] for v in sorted(dirs)]
    out["periodic"] = library_chain(per, sq)
    dirs13 = set()
    for a, b in keys8 + [(3, 1)]:
        for v in {(a, b), (b, a), (-a, b), (-b, a)}:
            dirs13.add(v)
    out["ours"] = library_chain([[(0, 0), v] for v in sorted(dirs13)], sq)
    # MILP chain of digital segments
    d = np.load(os.path.join(os.path.dirname(__file__), "chain2_M6_R48.npz"), allow_pickle=True)
    C, sets = d["C"], list(d["sets"])
    fam, info = {}, {}
    for r in range(1, R + 1):
        summ = []
        for S, k in zip(sets, C[r - 1]):
            summ += [list(map(tuple, np.asarray(S, int)))] * int(k)
        fam[r] = summ
        e, rho = eps_rho(mask_points(element_mask(summ)))
        info[r] = (rho, e)
    out["dss_milp"] = (fam, info, None)
    pickle.dump(out, open(CACHE, "wb"))
    return out


def load():
    return pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else build()


if __name__ == "__main__":
    out = build()
    from scipy.spatial import ConvexHull
    def holes(M):
        ys, xs = np.nonzero(M); P = np.stack([xs, ys], 1).astype(float); E = ConvexHull(P).equations
        yy, xx = np.mgrid[0:M.shape[0], 0:M.shape[1]]; Q = np.stack([xx.ravel(), yy.ravel()], 1)
        return int(np.all(Q @ E[:, :2].T + E[:, 2] <= 1e-9, axis=1).sum() - M.sum())
    for name, (fam, info, kappa) in out.items():
        nh = sum(holes(element_mask(fam[r])) for r in range(2, R + 1)) if name != "gauss" else 0
        rho = np.array([info[r][0] for r in range(1, R + 1)])
        print(f"{name:9s} holes={nh}  max radius step {np.diff(rho).max():.2f}")
        eps = np.array([info[r][1] for r in range(1, R + 1)])
        rho = np.array([info[r][0] for r in range(1, R + 1)])
        print(f"{name:9s} kappa={kappa if kappa is None else round(kappa, 4)}  eps r<=24 max {eps[:24].max():.3f}  "
              f"r<=48 max {eps.max():.3f} mean {eps.mean():.3f}  max|rho-r| {np.abs(rho - np.arange(1, R + 1)).max():.2f}"
              f"  distinct elements {len({id(fam[r][-1]) if fam[r] else 0 for r in fam})}")
