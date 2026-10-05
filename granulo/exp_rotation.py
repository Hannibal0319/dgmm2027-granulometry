"""E5: rotation invariance and monotonicity of size distributions.

X_theta = union of squares of several sizes rotated by theta (exact rasterisation). For each family
of structuring elements indexed by radius r = 1..R, F_theta(r) = |X_theta o D_r| / |X_theta|.
Reports: rotation sensitivity  S = mean_r std_theta F_theta(r),  and the number of (theta, r) with
F(r+1) > F(r) (violations of monotonicity, impossible for a granulometry)."""
import json
import math
import sys

import numpy as np

from families import chain_summands, gauss_disk_points, open_cascade
from geom import Generator, polygon_from_counts, roundness

R = 40
ANGLES = np.arange(0, 46, 3)


def scene(theta, n=420):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    X = np.zeros((n, n), bool)
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    centres = [(70, 70), (70, 210), (70, 350), (210, 70), (210, 210), (210, 350), (350, 70), (350, 210), (350, 350)]
    sides = [17, 25, 33, 41, 49, 57, 65, 73, 81]
    for (cy, cx), side in zip(centres, sides):
        u = c * (xx - cx) + s * (yy - cy)
        v = -s * (xx - cx) + c * (yy - cy)
        X |= (np.abs(u) <= side / 2) & (np.abs(v) <= side / 2)
    return X


def family_gauss():
    return {r: gauss_disk_points(r) for r in range(1, R + 1)}


def family_from_chain(keys, C):
    """map integer radius r -> summands of the chain element whose fitted radius is closest to r."""
    gens = [Generator(*k) for k in keys]
    rad = []
    for c in C:
        counts = {k: int(v) for k, v in zip(keys, c) if v > 0}
        rad.append(roundness(polygon_from_counts(gens, counts))[1])
    rad = np.array(rad)
    fam = {}
    for r in range(1, R + 1):
        j = int(np.argmin(np.abs(rad - r)))
        fam[r] = chain_summands(keys, C[j])
    return fam, rad


def octagon_chain_counts(Rmax):
    gens = [Generator(1, 0), Generator(1, 1)]
    keys = [(1, 0), (1, 1)]
    C = []
    a, b = 1, 0
    for r in range(1, Rmax + 6):
        best = None
        for aa in range(a, a + 4):
            for bb in range(b, b + 4):
                cnt = {(1, 0): aa}
                if bb: cnt[(1, 1)] = bb
                e, rad = roundness(polygon_from_counts(gens, cnt))
                t = max(e, abs(rad - r))
                if best is None or t < best[0]: best = (t, aa, bb)
        _, a, b = best
        C.append([a, b])
    return keys, np.array(C)


def run(families):
    out = {}
    for name, fam in families.items():
        F = np.zeros((len(ANGLES), R + 1))
        for i, th in enumerate(ANGLES):
            X = scene(th)
            A0 = X.sum()
            F[i, 0] = 1
            for r in range(1, R + 1):
                F[i, r] = open_cascade(X, fam[r]).sum() / A0
        S = float(F.std(0).mean())
        viol = int((np.diff(F, axis=1) > 0).sum())
        out[name] = dict(S=S, violations=viol, F=F.tolist())
        print(f"{name:12s} rotation sensitivity S={S:.4f}   monotonicity violations={viol}", flush=True)
    return out


if __name__ == "__main__":
    fams = {"Gauss": family_gauss()}
    k8, C8 = octagon_chain_counts(R)
    fams["octagon"], _ = family_from_chain(k8, C8)
    d = np.load(sys.argv[1], allow_pickle=True)
    C, sets = d["C"], list(d["sets"])
    R = min(R, len(C))
    fams["ours"] = {}
    for r in range(1, R + 1):
        summ = []
        for S, k in zip(sets, C[r - 1]): summ += [list(map(tuple, np.asarray(S, int)))] * int(k)
        fams["ours"][r] = summ
    globals()["R"] = R
    res = run(fams)
    json.dump(res, open("exp_rotation.json", "w"))
