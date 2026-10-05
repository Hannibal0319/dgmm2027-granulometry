"""Rolling-horizon MILP for monotone integer chains of 8-conn DSS orbits.
At each stage optimise counts for the next W target radii jointly (monotone, radius pinned
within +-0.5 of the target), minimise max error over the horizon; commit the first stage."""
import math, sys, time, numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
from geom import Generator, primitive_orbits, polygon_from_counts, roundness

def run(M, targets, W=4, ntheta=200, tl=30, maxinc=None):
    gens = [Generator(*k) for k in primitive_orbits(M)]
    keys = [g.key for g in gens]
    th = np.linspace(0, math.pi / 4, ntheta); U = np.stack([np.cos(th), np.sin(th)], 1)
    H = sp.csr_matrix(np.stack([g.support(U) for g in gens], 1)); ng = len(gens)
    prev = np.zeros(ng); prev[keys.index((1, 0))] = 1
    out = []; t0 = time.time()
    for i, r0 in enumerate(targets):
        rs = targets[i:i + W]; J = len(rs)
        nv = J * ng + J + 1          # counts per stage, rho per stage, t
        rows, lo_r, hi_r = [], [], []
        one = sp.csr_matrix(np.ones((ntheta, 1)))
        for j in range(J):
            left = sp.csr_matrix((ntheta, j * ng)); right = sp.csr_matrix((ntheta, (J - j - 1) * ng))
            e_rho = sp.csr_matrix(([1.0] * ntheta, (range(ntheta), [j] * ntheta)), shape=(ntheta, J))
            rows.append(sp.hstack([left, H, right, -e_rho, -one])); rows.append(sp.hstack([left, -H, right, e_rho, -one]))
        I = sp.identity(ng)
        for j in range(1, J):
            rows.append(sp.hstack([sp.csr_matrix((ng, (j - 1) * ng)), I, -I, sp.csr_matrix((ng, (J - j - 1) * ng + J + 1))]))
        A = sp.vstack(rows).tocsr()
        c = np.zeros(nv); c[-1] = 1
        lb = np.r_[np.tile(prev, J), np.array(rs) - 0.5, 0]
        ubc = np.tile(prev + (maxinc if maxinc else 10**6), J)
        ub = np.r_[ubc, np.array(rs) + 0.5, np.inf]
        res = milp(c, constraints=[LinearConstraint(A, -np.inf, 0)], integrality=np.r_[np.ones(J * ng), np.zeros(J + 1)],
                   bounds=Bounds(lb, ub), options={"time_limit": tl, "mip_rel_gap": 1e-2})
        if res.x is None: print("fail at", r0); break
        prev = np.round(res.x[:ng])
        counts = {k: int(v) for k, v in zip(keys, prev) if v > 0}
        e, rad = roundness(polygon_from_counts(gens, counts))
        out.append((r0, rad, e, len(counts)))
        if len(out) % 10 == 0:
            o = np.array(out); print(f"r={r0:7.1f} rad={rad:8.2f} err={e:.3f} max={o[:,2].max():.3f} gens={len(counts)} {time.time()-t0:.0f}s", flush=True)
    return np.array(out), counts

if __name__ == "__main__":
    M = float(sys.argv[1]); R = float(sys.argv[2]); W = int(sys.argv[3])
    targets = list(np.arange(2, 30, 1.0)) + list(np.arange(30, R + 1, 4.0))
    out, counts = run(M, targets, W=W)
    np.save(f"rolling_M{int(M)}_R{int(R)}_W{W}.npy", out)
    o = out[out[:, 1] >= 10]
    for lo_, hi_ in ((10, 50), (50, 100), (100, 200), (200, 400), (400, 1e9)):
        k = (o[:, 1] >= lo_) & (o[:, 1] < hi_)
        if k.any(): print(f"  rad in [{lo_},{hi_}): max {o[k,2].max():.3f} mean {o[k,2].mean():.3f}")
