"""Unit-step granulometric chains with single-segment increments (MILP, HiGHS).

Generators: individual 8-connected digital straight segments (all D4 images of DSS(a,b), |v| <= M),
plus nothing else; element r = sum_g c_{r,g} DSS_g (+ the unit square, required for exactness),
c_{r,g} non-decreasing in r.  Element r must be within t_r of a disk of radius rho_r in
[r - delta, r + delta] with a free centre: |h_r(theta) - <c_r, u_theta> - rho_r| <= t_r.
Objective: max_r t_r + w * mean_r t_r.  Warm start: radius-by-radius greedy.
"""
import math
import sys
import time

import highspy
import numpy as np
import scipy.sparse as sp

from geom import Generator, primitive_orbits, roundness
from optchain import solve_highs

NT = 240
TH = np.linspace(0, 2 * math.pi, NT, endpoint=False)
U = np.stack([np.cos(TH), np.sin(TH)], 1)


def members(M):
    sets, keys = [], []
    for k in primitive_orbits(M):
        for S in Generator(*k).sets:
            sets.append(np.asarray(S, float))
            keys.append(k)
    return sets, keys


def build(Hm, radii, prev, sq_rows, delta, w_mean):
    """vars per element j: counts (ng), cx, cy, rho, t ; then T."""
    nt, ng = Hm.shape
    J = len(radii)
    per = ng + 4
    nv = J * per + 1
    rows = []
    Hs = sp.csr_matrix(Hm)
    cu = sp.csr_matrix(U)
    one = sp.csr_matrix(np.ones((nt, 1)))
    for j in range(J):
        L = sp.csr_matrix((nt, j * per))
        Rr = sp.csr_matrix((nt, (J - j - 1) * per + 1))
        # h - c.u - rho - t <= 0     and    -h + c.u + rho - t <= 0
        rows.append(sp.hstack([L, Hs, -cu, -one, -one, Rr]))
        rows.append(sp.hstack([L, -Hs, cu, one, -one, Rr]))
    I = sp.identity(ng)
    for j in range(1, J):
        rows.append(sp.hstack([sp.csr_matrix((ng, (j - 1) * per)), I, sp.csr_matrix((ng, 4)), -I,
                               sp.csr_matrix((ng, (J - j - 1) * per + 4 + 1))]))
    # t_j <= T
    tsel = sp.lil_matrix((J, nv))
    for j in range(J):
        tsel[j, j * per + ng + 3] = 1
        tsel[j, nv - 1] = -1
    rows.append(tsel.tocsr())
    A = sp.vstack(rows).tocsr()
    rhs = np.zeros(A.shape[0])
    cost = np.zeros(nv)
    cost[-1] = 1
    lb = np.zeros(nv)
    ub = np.full(nv, highspy.kHighsInf)
    integ = np.zeros(nv)
    for j, r in enumerate(radii):
        o = j * per
        cost[o + ng + 3] = w_mean / J
        integ[o:o + ng] = 1
        lb[o:o + ng] = prev
        for i in sq_rows:
            lb[o + i] = max(lb[o + i], 1)
        lb[o + ng:o + ng + 2] = -highspy.kHighsInf
        lb[o + ng + 2], ub[o + ng + 2] = r - delta, r + delta
    return A, rhs, cost, lb, ub, integ, per


def optimise(M, R, delta=0.5, w_mean=0.05, tl_step=20, tl_global=1800, window=3, verbose=True):
    sets, keys = members(M)
    Hm = np.stack([(U @ S.T).max(1) for S in sets], 1)
    ng = len(sets)
    sq_rows = [i for i, k in enumerate(keys) if k == (1, 0)]
    radii = list(range(1, R + 1))
    t0 = time.time()
    prev = np.zeros(ng)
    start = []
    for j, r in enumerate(radii):
        rs = radii[j:j + window]
        A, rhs, cost, lb, ub, integ, per = build(Hm, rs, prev, sq_rows, delta, w_mean)
        x, obj, _ = solve_highs(A, rhs, cost, lb, ub, integ, tl_step)
        prev = np.round(x[:ng])
        start.append(np.r_[prev, x[ng:ng + 4]])
        if verbose and r % 10 == 0:
            print(f"  warm r={r} t={x[ng + 3]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    W = np.array(start)
    T0 = W[:, ng + 3].max()
    print(f"[M={M} R={R}] warm start max t = {T0:.3f} ({time.time() - t0:.0f}s)", flush=True)
    A, rhs, cost, lb, ub, integ, per = build(Hm, radii, np.zeros(ng), sq_rows, delta, w_mean)
    x, obj, bound = solve_highs(A, rhs, cost, lb, ub, integ, tl_global, start=np.r_[W.ravel(), T0])
    X = np.round(x[:-1]).reshape(len(radii), per)
    print(f"[M={M} R={R}] global max t = {x[-1]:.3f}, dual bound {bound:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return sets, keys, W[:, :ng], X[:, :ng], X[:, ng:]


def exact_eval(sets, C):
    """Exact roundness of each element with free centre (Chebyshev fit on a fine grid)."""
    from dither_chain import cheb_disk, TH as TH2, U as U2
    out = []
    for c in C:
        h = np.zeros(len(TH2))
        for S, k in zip(sets, c):
            if k: h += k * (U2 @ S.T).max(1)
        out.append(cheb_disk(h))
    return np.array(out)  # (eps, rho)


if __name__ == "__main__":
    M, R = float(sys.argv[1]), int(sys.argv[2])
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 1800
    sets, keys, C0, C, aux = optimise(M, R, tl_global=tl)
    for name, CC in (("warm", C0), ("global", C)):
        ev = exact_eval(sets, CC)
        rad_err = np.abs(ev[:, 1] - np.arange(1, R + 1))
        print(f"  {name}: max eps {ev[:, 0].max():.3f}  mean eps {ev[:, 0].mean():.3f}  max |rho-r| {rad_err.max():.3f}  "
              f"max radius step {np.diff(ev[:, 1]).max():.3f}")
    np.savez(f"chain2_M{int(M)}_R{R}.npz", C=C, C0=C0, keys=keys, sets=np.array(sets, dtype=object), allow_pickle=True)
