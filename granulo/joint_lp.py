"""Joint LP: real-valued monotone trajectory x(r_j) >= x(r_{j-1}) >= 0 minimising
max_j sup_theta |H x(r_j) - r_j|  (absolute error), generators = 8-conn DSS orbits."""
import math, sys, time, numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from geom import Generator, primitive_orbits

def solve(M, radii, ntheta=240, verbose=True):
    gens = [Generator(*k) for k in primitive_orbits(M)]
    th = np.linspace(0, math.pi / 4, ntheta); U = np.stack([np.cos(th), np.sin(th)], 1)
    H = np.stack([g.support(U) for g in gens], 1)            # (nt, ng)
    nt, ng = H.shape; J = len(radii)
    nv = J * ng + 1                                           # x_{j,g}, t
    Hs = sp.csr_matrix(H)
    rows, rhs = [], []
    for j, r in enumerate(radii):
        Z = [None] * J
        blk = sp.lil_matrix((nt, nv))
        A = sp.hstack([sp.csr_matrix((nt, j * ng)), Hs, sp.csr_matrix((nt, (J - j - 1) * ng)), -sp.csr_matrix(np.ones((nt, 1)))])
        B = sp.hstack([sp.csr_matrix((nt, j * ng)), -Hs, sp.csr_matrix((nt, (J - j - 1) * ng)), -sp.csr_matrix(np.ones((nt, 1)))])
        rows += [A, B]; rhs += [np.full(nt, r), np.full(nt, -r)]
    # monotone: x_{j-1} - x_j <= 0
    I = sp.identity(ng)
    for j in range(1, J):
        rows.append(sp.hstack([sp.csr_matrix((ng, (j - 1) * ng)), I, -I, sp.csr_matrix((ng, (J - j - 1) * ng + 1))]))
        rhs.append(np.zeros(ng))
    A = sp.vstack(rows).tocsr(); b = np.concatenate(rhs)
    c = np.zeros(nv); c[-1] = 1
    t0 = time.time()
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * nv, method="highs")
    X = res.x[:-1].reshape(J, ng)
    if verbose: print(f"M={M} gens={ng} radii={J} LP status={res.status} max abs err={res.x[-1]:.4f} ({time.time()-t0:.0f}s)")
    return gens, X, res.x[-1]

if __name__ == "__main__":
    M = float(sys.argv[1]); R = float(sys.argv[2])
    radii = list(np.unique(np.round(np.geomspace(3, R, int(sys.argv[3]) if len(sys.argv) > 3 else 30), 2)))
    gens, X, t = solve(M, radii)
    np.savez(f"joint_M{int(M)}_R{int(R)}.npz", X=X, radii=radii, keys=[g.key for g in gens])
