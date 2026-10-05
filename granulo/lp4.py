import math, sys, numpy as np
from scipy.optimize import linprog
from gens4 import orbits

def grid(n=3000):
    t = np.linspace(0, math.pi / 4, n); return np.stack([np.cos(t), np.sin(t)], 1)

def best_weights(gens, U=None, fixed=None):
    U = grid() if U is None else U
    H = np.stack([g.support(U) for g in gens], 1)
    ng = H.shape[1]
    c = np.zeros(ng + 1); c[-1] = 1
    A = np.vstack([np.hstack([H, -np.ones((len(U), 1))]), np.hstack([-H, -np.ones((len(U), 1))])])
    b = np.concatenate([np.ones(len(U)), -np.ones(len(U))])
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * (ng + 1), method="highs")
    return res.x[:ng], res.x[-1]

if __name__ == "__main__":
    rows = []
    for m in [2, 3, 5, 8, 12, 17, 24, 34, 48]:
        gens = orbits(m)
        x, t = best_weights(gens)
        used = [(g.key, round(float(v), 4)) for g, v in zip(gens, x) if v > 1e-7]
        print(f"m={m:3d} orbits={len(gens):4d} rel.err={t:.3e}  m^2*err={m*m*t:.3f}  used={len(used)}", flush=True)
        rows.append((m, t))
    np.save("lp4_table.npy", np.array(rows))
