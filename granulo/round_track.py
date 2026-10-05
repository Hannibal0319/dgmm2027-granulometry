"""Integer monotone chain tracking the joint-LP plan: at radius r each count is
max(prev, floor x(r)) or max(prev, ceil x(r)); a MILP picks the roundest choice."""
import math, sys, time, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from geom import Generator, polygon_from_counts, roundness

def track(plan_file, steps, ntheta=400, time_limit=10):
    d = np.load(plan_file); X, radii = d["X"], d["radii"]; keys = [tuple(k) for k in d["keys"]]
    gens = [Generator(*k) for k in keys]
    th = np.linspace(0, math.pi / 4, ntheta); U = np.stack([np.cos(th), np.sin(th)], 1)
    H = np.stack([g.support(U) for g in gens], 1)
    ng = len(gens)
    prev = np.zeros(ng); prev[keys.index((1, 0))] = 1  # unit square first (exactness)
    out = []
    t0 = time.time()
    for r in steps:
        j = np.searchsorted(radii, r)
        if j == 0: x = X[0] * r / radii[0]
        elif j >= len(radii): x = X[-1] * r / radii[-1]
        else:
            w = (r - radii[j-1]) / (radii[j] - radii[j-1]); x = (1 - w) * X[j-1] + w * X[j]
        lo = np.maximum(prev, np.floor(x + 1e-9)); hi = np.maximum(prev, np.ceil(x - 1e-9))
        free = hi > lo
        base = H @ lo
        Hf = H[:, free]; nf = Hf.shape[1]
        # vars: binaries (nf), rho in [r - 0.5, r + 0.5], t
        c = np.zeros(nf + 2); c[-1] = 1
        A = np.vstack([np.hstack([Hf, -np.ones((ntheta, 1)), -np.ones((ntheta, 1))]),
                       np.hstack([-Hf, np.ones((ntheta, 1)), -np.ones((ntheta, 1))])])
        ub = np.concatenate([-base, base])
        res = milp(c, constraints=[LinearConstraint(A, -np.inf, ub)], integrality=np.r_[np.ones(nf), 0, 0],
                   bounds=Bounds(np.r_[np.zeros(nf), r - 0.5, 0], np.r_[np.ones(nf), r + 0.5, np.inf]),
                   options={"time_limit": time_limit})
        cnew = lo.copy(); cnew[free] += np.round(res.x[:nf])
        prev = cnew
        counts = {k: int(v) for k, v in zip(keys, cnew) if v > 0}
        e, rad = roundness(polygon_from_counts(gens, counts))
        out.append((r, rad, e, int((cnew > 0).sum()), int(cnew.sum())))
        if len(out) % 20 == 0:
            o = np.array(out); print(f"r={r:7.1f} rad={rad:8.2f} err={e:.3f} max so far={o[:,2].max():.3f} (rad>20: {o[o[:,1]>20,2].max() if (o[:,1]>20).any() else 0:.3f}) gens used={out[-1][3]} {time.time()-t0:.0f}s", flush=True)
    return np.array(out), counts

if __name__ == "__main__":
    plan = sys.argv[1]; R = float(sys.argv[2])
    steps = list(np.arange(3, 50, 0.5)) + list(np.arange(50, R + 1, 2.0))
    out, counts = track(plan, steps)
    np.save(plan.replace(".npz", "_track.npy"), out)
    print("FINAL max err (all):", out[:, 2].max(), " (rad>=10):", out[out[:, 1] >= 10, 2].max())
