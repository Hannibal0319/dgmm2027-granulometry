"""Near-optimal granulometric chains of digital disks by mixed-integer programming (HiGHS).

Element r (r = 1..R) is the digital Minkowski sum of c_{r,g} copies of each generator g
(D4-orbit of the 8-connected digital straight segment of a primitive vector, |v| <= M);
c_{r,g} is non-decreasing in r, so D_r = D_{r-1} (+) C_r and the family is a granulometry.
Objective: minimise max_r t_r (+ small weight on the mean), where t_r bounds the deviation
of the centred support function of conv(D_r) from a disk radius rho_r in [r - delta, r + delta].
Step 1 builds a warm start radius by radius; step 2 optimises globally.
"""
import math
import sys
import time

import highspy
import numpy as np
import scipy.sparse as sp

from geom import Generator, polygon_from_counts, primitive_orbits, roundness


def support_matrix(gens, ntheta):
    th = np.linspace(0, math.pi / 4, ntheta)
    U = np.stack([np.cos(th), np.sin(th)], 1)
    return np.stack([g.support(U) for g in gens], 1)


def build(H, radii, ng, sq, delta, w_mean, prev=None):
    """LP data for consecutive radii; if prev is given (single radius), counts >= prev."""
    nt = H.shape[0]
    J = len(radii)
    nc = J * ng
    nv = nc + 2 * J + 1
    Hs = sp.csr_matrix(H)
    rows = []
    for j in range(J):
        L = sp.csr_matrix((nt, j * ng))
        Rr = sp.csr_matrix((nt, (J - j - 1) * ng))
        e = sp.csr_matrix((np.ones(nt), (np.arange(nt), np.full(nt, j))), shape=(nt, J))
        z = sp.csr_matrix((nt, 1))
        rows += [sp.hstack([L, Hs, Rr, -e, -e, z]), sp.hstack([L, -Hs, Rr, e, -e, z])]
    I = sp.identity(ng)
    for j in range(1, J):
        rows.append(sp.hstack([sp.csr_matrix((ng, (j - 1) * ng)), I, -I,
                               sp.csr_matrix((ng, (J - j - 1) * ng + 2 * J + 1))]))
    rows.append(sp.hstack([sp.csr_matrix((J, nc + J)), sp.identity(J), -sp.csr_matrix(np.ones((J, 1)))]))
    A = sp.vstack(rows).tocsr()
    rhs = np.zeros(A.shape[0])
    cost = np.zeros(nv)
    cost[-1] = 1
    cost[nc + J:nc + 2 * J] = w_mean / J
    lb = np.r_[np.zeros(nc), np.array(radii) - delta, np.zeros(J + 1)]
    ub = np.r_[np.full(nc, highspy.kHighsInf), np.array(radii) + delta, np.full(J + 1, highspy.kHighsInf)]
    for j in range(J):
        lb[j * ng + sq] = 1
        if prev is not None:
            lb[j * ng:(j + 1) * ng] = np.maximum(lb[j * ng:(j + 1) * ng], prev)
    integ = np.r_[np.ones(nc), np.zeros(2 * J + 1)]
    return A, rhs, cost, lb, ub, integ


def solve_highs(A, rhs, cost, lb, ub, integ, time_limit, start=None, threads=12, gap=1e-3, verbose=False):
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", gap)
    h.setOptionValue("threads", threads)
    lp = highspy.HighsLp()
    nv = len(cost)
    lp.num_col_ = nv
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = np.full(A.shape[0], -highspy.kHighsInf)
    lp.row_upper_ = rhs
    Ac = A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = Ac.indptr
    lp.a_matrix_.index_ = Ac.indices
    lp.a_matrix_.value_ = Ac.data
    lp.integrality_ = [highspy.HighsVarType.kInteger if v else highspy.HighsVarType.kContinuous for v in integ]
    h.passModel(lp)
    if start is not None:
        sol = highspy.HighsSolution()
        sol.col_value = list(start)
        sol.value_valid = True
        h.setSolution(sol)
    h.run()
    x = np.array(h.getSolution().col_value)
    info = h.getInfo()
    return x, info.objective_function_value, info.mip_dual_bound


def evaluate(gens, X, radii):
    keys = [g.key for g in gens]
    out = []
    for j, r in enumerate(radii):
        counts = {k: int(v) for k, v in zip(keys, X[j]) if v > 0}
        e, rad = roundness(polygon_from_counts(gens, counts))
        out.append((r, rad, e, max(e + abs(rad - r), 0)))
    return np.array(out)


def optimise(M, R, delta=0.5, ntheta=120, w_mean=0.05, tl_step=10, tl_global=1200, window=None):
    gens = [Generator(*k) for k in primitive_orbits(M)]
    keys = [g.key for g in gens]
    ng = len(gens)
    sq = keys.index((1, 0))
    H = support_matrix(gens, ntheta)
    radii = list(range(1, R + 1))
    # step 1: warm start, radius by radius (optionally a short look-ahead window)
    t0 = time.time()
    prev = np.zeros(ng)
    X0 = []
    rho0, t_0 = [], []
    W = window or 1
    for j, r in enumerate(radii):
        rs = radii[j:j + W]
        A, rhs, cost, lb, ub, integ = build(H, rs, ng, sq, delta, w_mean, prev=prev)
        x, obj, _ = solve_highs(A, rhs, cost, lb, ub, integ, tl_step)
        c = np.round(x[:ng])
        X0.append(c)
        rho0.append(x[len(rs) * ng])
        t_0.append(x[len(rs) * ng + len(rs)])
        prev = c
    X0 = np.array(X0)
    T0 = max(t_0)
    print(f"[M={M} R={R}] warm start: max t = {T0:.3f} ({time.time() - t0:.0f}s)", flush=True)
    # step 2: global
    A, rhs, cost, lb, ub, integ = build(H, radii, ng, sq, delta, w_mean)
    start = np.r_[X0.ravel(), rho0, t_0, T0]
    x, obj, bound = solve_highs(A, rhs, cost, lb, ub, integ, tl_global, start=start)
    X = np.round(x[:len(radii) * ng]).reshape(len(radii), ng)
    T = x[-1]
    print(f"[M={M} R={R}] global: max t = {T:.3f}, objective {obj:.3f}, dual bound {bound:.3f} ({time.time() - t0:.0f}s)",
          flush=True)
    return gens, X0, X, radii


if __name__ == "__main__":
    M, R = float(sys.argv[1]), int(sys.argv[2])
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 1200
    gens, X0, X, radii = optimise(M, R, tl_global=tl)
    for name, XX in (("warm", X0), ("global", X)):
        ev = evaluate(gens, XX, radii)
        print(f"  {name}: max Hausdorff-to-B_r {ev[:, 3].max():.3f}, max roundness {ev[:, 2].max():.3f}, "
              f"mean roundness {ev[:, 2].mean():.3f}")
    np.savez(f"chain_M{int(M)}_R{R}.npz", X=X, X0=X0, keys=[g.key for g in gens], radii=radii)
