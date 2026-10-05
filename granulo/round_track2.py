"""Integer monotone tracking with a wider window: c_g in [max(prev, floor x - 1), max(prev, ceil x + 1)],
radius pinned to r +- 0.5, objective = sup error + small penalty on deviation from the plan."""
import math, sys, time, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from geom import Generator, polygon_from_counts, roundness

def plan_at(X, radii, r):
    j = np.searchsorted(radii, r)
    if j == 0: return X[0] * r / radii[0]
    if j >= len(radii): return X[-1] * r / radii[-1]
    w = (r - radii[j-1]) / (radii[j] - radii[j-1]); return (1 - w) * X[j-1] + w * X[j]

def _solve(H, x, lo, hi, r, ng, ntheta, lam, time_limit):
    nv = 2 * ng + 2
    cobj = np.zeros(nv); cobj[-1] = 1; cobj[ng:2*ng] = lam
    one = np.ones((ntheta, 1)); Zt = np.zeros((ntheta, ng))
    A1 = np.hstack([H, Zt, -one, -one]); A2 = np.hstack([-H, Zt, one, -one])
    A3 = np.hstack([np.eye(ng), -np.eye(ng), np.zeros((ng, 2))])
    A4 = np.hstack([-np.eye(ng), -np.eye(ng), np.zeros((ng, 2))])
    cons = [LinearConstraint(np.vstack([A1, A2]), -np.inf, 0),
            LinearConstraint(np.vstack([A3, A4]), -np.inf, np.r_[x, -x])]
    return milp(cobj, constraints=cons, integrality=np.r_[np.ones(ng), np.zeros(ng + 2)],
                bounds=Bounds(np.r_[lo, np.zeros(ng), r - 0.5, 0], np.r_[hi, np.full(ng, np.inf), r + 0.5, np.inf]),
                options={"time_limit": time_limit, "mip_rel_gap": 1e-3})


def track(plan_file, steps, ntheta=360, time_limit=20, lam=1e-3, slack=1):
    d = np.load(plan_file); X, radii = d["X"], d["radii"]; keys = [tuple(k) for k in d["keys"]]
    gens = [Generator(*k) for k in keys]
    th = np.linspace(0, math.pi / 4, ntheta); U = np.stack([np.cos(th), np.sin(th)], 1)
    H = np.stack([g.support(U) for g in gens], 1)
    ng = len(gens)
    prev = np.zeros(ng); prev[keys.index((1, 0))] = 1
    out = []; allc = []; t0 = time.time()
    for r in steps:
        x = plan_at(X, radii, r)
        for sl in (slack, 2 * slack + 1, 4 * slack + 3, 50):
          lo = np.maximum(prev, np.floor(x) - sl); hi = np.maximum(prev, np.ceil(x) + sl)
          res = _solve(H, x, lo, hi, r, ng, ntheta, lam, time_limit)
          if res.x is not None: break
        if res.x is not None:
            prev = np.round(res.x[:ng])  # otherwise: radius already beyond the window, keep element
        counts = {k: int(v) for k, v in zip(keys, prev) if v > 0}
        allc.append(prev.copy())
        e, rad = roundness(polygon_from_counts(gens, counts))
        out.append((r, rad, e, int((prev > 0).sum()), int(prev.sum())))
        if len(out) % 25 == 0:
            o = np.array(out); print(f"r={r:7.1f} rad={rad:8.2f} err={e:.3f} max so far={o[:,2].max():.3f} gens used={out[-1][3]} {time.time()-t0:.0f}s", flush=True)
    np.savez(plan_file.replace('.npz', '_chain.npz'), C=np.array(allc), keys=keys, steps=np.array([o[0] for o in out]))
    return np.array(out), counts

if __name__ == "__main__":
    plan = sys.argv[1]; R = float(sys.argv[2]); tag = sys.argv[3] if len(sys.argv) > 3 else ""
    step = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    steps = list(np.arange(3, 50, 0.5)) + list(np.arange(50, R + 1, step))
    out, counts = track(plan, steps)
    np.save(plan.replace(".npz", f"_track2{tag}.npy"), out)
    o = out[out[:, 1] >= 10]
    print("FINAL max err (rad>=10):", o[:, 2].max(), " mean:", o[:, 2].mean())
    for lo_, hi_ in ((10, 100), (100, 300), (300, 600), (600, 1e9)):
        k = (o[:, 1] >= lo_) & (o[:, 1] < hi_)
        if k.any(): print(f"  rad in [{lo_},{hi_}): max {o[k,2].max():.3f} mean {o[k,2].mean():.3f}")
