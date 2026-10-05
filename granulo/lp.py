import math, numpy as np
from scipy.optimize import linprog
from geom import Generator, primitive_orbits

def grid(n=4000):
    t = np.linspace(0, math.pi / 4, n)
    return np.stack([np.cos(t), np.sin(t)], 1)

def best_weights(gens, U=None):
    """min t s.t. |sum_g x_g h_g(u) - 1| <= t for u in U (theta in [0, pi/4]), x >= 0."""
    U = grid() if U is None else U
    H = np.stack([g.support(U) for g in gens], 1)  # (nu, ng)
    ng = H.shape[1]
    c = np.zeros(ng + 1); c[-1] = 1
    A = np.vstack([np.hstack([H, -np.ones((len(U), 1))]), np.hstack([-H, -np.ones((len(U), 1))])])
    b = np.concatenate([np.ones(len(U)), -np.ones(len(U))])
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * ng + [(0, None)], method="highs")
    return res.x[:ng], res.x[-1]

if __name__ == "__main__":
    for m in [1, 1.5, 2.3, 3.2, 4.3, 5.4, 6.5, 8, 10, 13]:
        keys = primitive_orbits(m)
        gens = [Generator(*k) for k in keys]
        x, t = best_weights(gens)
        print(f"m={m:5.1f} orbits={len(gens):3d}  rel.err={t:.2e}  -> err at r=100: {100*t:.3f}, r=1000: {1000*t:.2f}   min weight {x[x>1e-9].min() if (x>1e-9).any() else 0:.4f}, used {int((x>1e-9).sum())}")
