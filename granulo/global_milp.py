"""Globally optimal monotone chain for integer radii r = 1..R: counts c_r (integer, c_r >= c_{r-1}),
centred support H c_r within [r - t, r + t] on a theta grid; minimise t."""
import math, sys, time, numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from geom import Generator, primitive_orbits, polygon_from_counts, roundness

def solve(M, R, r0=1, ntheta=120, tl=600, gap=1e-3, delta=0.5, w_mean=0.05):
    gens = [Generator(*k) for k in primitive_orbits(M)]
    keys = [g.key for g in gens]; ng = len(gens)
    th = np.linspace(0, math.pi / 4, ntheta); U = np.stack([np.cos(th), np.sin(th)], 1)
    H = sp.csr_matrix(np.stack([g.support(U) for g in gens], 1))
    radii = list(range(r0, R + 1)); J = len(radii)
    # vars: counts (J*ng), rho_j (J), t_j (J), T
    nc = J * ng; nv = nc + 2 * J + 1
    rows, rhs = [], []
    for j, r in enumerate(radii):
        L = sp.csr_matrix((ntheta, j * ng)); Rr = sp.csr_matrix((ntheta, (J - j - 1) * ng))
        e = sp.csr_matrix((np.ones(ntheta), (np.arange(ntheta), np.full(ntheta, j))), shape=(ntheta, J))
        Z1 = sp.csr_matrix((ntheta, 1))
        rows += [sp.hstack([L, H, Rr, -e, -e, Z1]), sp.hstack([L, -H, Rr, e, -e, Z1])]
        rhs += [np.zeros(ntheta), np.zeros(ntheta)]
    I = sp.identity(ng)
    for j in range(1, J):
        rows.append(sp.hstack([sp.csr_matrix((ng, (j - 1) * ng)), I, -I, sp.csr_matrix((ng, (J - j - 1) * ng + 2 * J + 1))]))
        rhs.append(np.zeros(ng))
    # t_j <= T
    rows.append(sp.hstack([sp.csr_matrix((J, nc + J)), sp.identity(J), -sp.csr_matrix(np.ones((J, 1)))]))
    rhs.append(np.zeros(J))
    A = sp.vstack(rows).tocsr(); b = np.concatenate(rhs)
    c = np.zeros(nv); c[-1] = 1; c[nc + J:nc + 2 * J] = w_mean / J
    lb = np.r_[np.zeros(nc), np.array(radii) - delta, np.zeros(J + 1)]
    ub = np.r_[np.full(nc, np.inf), np.array(radii) + delta, np.full(J + 1, np.inf)]
    for j in range(J): lb[j * ng + keys.index((1, 0))] = 1  # unit square in every element (exactness)
    t0 = time.time()
    res = milp(c, constraints=[LinearConstraint(A, -np.inf, b)], integrality=np.r_[np.ones(nc), np.zeros(2 * J + 1)],
               bounds=Bounds(lb, ub), options={"time_limit": tl, "mip_rel_gap": gap, "disp": False})
    X = np.round(res.x[:nc]).reshape(J, ng)
    errs = []
    for j, r in enumerate(radii):
        counts = {k: int(v) for k, v in zip(keys, X[j]) if v > 0}
        e, rad = roundness(polygon_from_counts(gens, counts))
        errs.append((r, rad, e, abs(rad - r)))
    print(f"M={M} R={R} status={res.status} T*={res.x[-1]:.3f} time={time.time()-t0:.0f}s")
    return keys, X, np.array(errs)

if __name__ == "__main__":
    M = float(sys.argv[1]); R = int(sys.argv[2]); tl = float(sys.argv[3]) if len(sys.argv) > 3 else 600
    keys, X, errs = solve(M, R, tl=tl)
    np.savez(f"global_M{int(M)}_R{R}.npz", X=X, keys=keys, errs=errs)
    print("max roundness (exact polygon):", errs[:, 2].max().round(3), " mean:", errs[:, 2].mean().round(3), " max |rho-r|:", errs[:, 3].max().round(2))
    print("per r roundness:", [(int(r), round(float(e), 2)) for r, _, e, _ in errs][:: max(1, R // 16)])
