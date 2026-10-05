import math, sys, time, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from gens4 import orbits, polygon
from geom import roundness

M = float(sys.argv[1]); R = int(sys.argv[2])
gens = orbits(M)
th = np.linspace(0, math.pi / 4, 400)
U = np.stack([np.cos(th), np.sin(th)], 1)
H = np.stack([g.support(U) for g in gens], 1)
ng = len(gens)
prev = np.zeros(ng)
hist = []
t0 = time.time()
for r in range(2, R + 1):
    # vars: counts (ng, integer), rho, t ; minimise t
    c = np.zeros(ng + 2); c[-1] = 1
    A1 = np.hstack([H, -np.ones((len(U), 1)), -np.ones((len(U), 1))])   # H x - rho - t <= 0
    A2 = np.hstack([-H, np.ones((len(U), 1)), -np.ones((len(U), 1))])   # -H x + rho - t <= 0
    cons = [LinearConstraint(np.vstack([A1, A2]), -np.inf, 0)]
    lb = np.concatenate([prev, [r, 0]]); ub = np.concatenate([prev + 50, [r + 1, np.inf]])
    integ = np.concatenate([np.ones(ng), [0, 0]])
    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub), options={"time_limit": 20})
    x = np.round(res.x[:ng])
    prev = x
    counts = {g.key: int(v) for g, v in zip(gens, x) if v > 0}
    e, rad = roundness(polygon(gens, counts))
    hist.append((r, rad, e))
    if r % 10 == 0 or r == R:
        print(f"r={r:4d} rad={rad:7.2f} err={e:.3f} maxerr_so_far={max(h[2] for h in hist):.3f} used={len(counts)} {time.time()-t0:.0f}s", flush=True)
np.save(f"online_M{int(M)}_R{R}.npy", np.array(hist))
print('final counts', counts)
