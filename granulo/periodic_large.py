"""Error of LP-weighted periodic-line chains (+2x2 square) up to large radii, from support functions."""
import math, json, numpy as np
from all_families import lp_weights
from dither_chain import cheb_disk, U as U2
from geom import primitive_orbits

def lib(m):
    dirs = set()
    for a, b in primitive_orbits(m):
        for v in {(a, b), (b, a), (-a, b), (-b, a)}: dirs.add(v)
    return [[(0, 0), v] for v in sorted(dirs)]

def run(m, Rmax, dt=0.25):
    """Greedy single-segment steps towards round(t y) (same scheme as all_families.library_chain)."""
    per = lib(m); y, kappa = lp_weights(per)
    hs = np.stack([(U2 @ np.asarray(S, float).T).max(1) for S in per], 1)
    h0 = (U2 @ np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float).T).max(1)
    out = []; t = 0.0; cur = np.zeros(len(per))
    while True:
        target = np.round(t * y)
        while (target > cur).any():
            best = None
            for i in np.nonzero(target > cur)[0]:
                cur[i] += 1; e, rho = cheb_disk(h0 + hs @ cur); cur[i] -= 1
                if best is None or e < best[0]: best = (e, rho, i)
            cur[best[2]] += 1; out.append((best[1], best[0]))
        if out and out[-1][0] > Rmax: break
        t += dt
    return np.array(out), kappa

res = {}
for name, m in (("sqrt5", math.sqrt(5)), ("sqrt10", math.sqrt(10)), ("5", 5.0), ("sqrt50", math.sqrt(50))):
    o, k = run(m, 1000)
    print(name, "max radius step", np.diff(o[:, 0]).max().round(2), flush=True)
    res[name] = dict(kappa=k, rho=o[:, 0].tolist(), eps=o[:, 1].tolist())
    for R in (48, 100, 300, 1000):
        sel = o[o[:, 0] <= R]
        print(f"periodic {name:6s} kappa={k:.4f}  max eps rho<={R}: {sel[:, 1].max():.3f}", flush=True)
json.dump(res, open("periodic_large.json", "w"))
