import math, sys, numpy as np
from gens4 import orbits, polygon
from geom import roundness
from lp4 import best_weights

MS = [2, 3, 5, 8, 12, 17, 24, 34, 48]
def build(beta, R, mode="floor"):
    allg = orbits(MS[-1]); idx = {g.key: i for i, g in enumerate(allg)}
    Y = []
    for m in MS:
        gs = orbits(m); x, t = best_weights(gs)
        y = np.zeros(len(allg))
        for g, v in zip(gs, x): y[idx[g.key]] = v
        Y.append(y)
    rk = [beta * m * m for m in MS]
    c = np.zeros(len(allg)); out = []
    for r in range(2, R + 1):
        k = max([i for i in range(len(MS)) if rk[i] <= r] + [0])
        if k + 1 < len(MS) and r >= rk[0]:
            lam = math.log(r / rk[k]) / math.log(rk[k + 1] / rk[k])
            y = (1 - lam) * Y[k] + lam * Y[k + 1]
        else:
            y = Y[k]
        x = r * y
        t = np.floor(x) if mode == "floor" else np.round(x)
        c = np.maximum(c, t)
        counts = {g.key: int(v) for g, v in zip(allg, c) if v > 0}
        if not counts: continue
        e, rad = roundness(polygon(allg, counts))
        out.append((r, rad, e))
    return np.array(out)

if __name__ == "__main__":
    R = int(sys.argv[1])
    for beta in (0.5, 1, 2, 4):
        for mode in ("floor", "round"):
            o = build(beta, R, mode)
            seg = lambda lo, hi: o[(o[:,1] >= lo) & (o[:,1] < hi), 2].max()
            print(f"beta={beta} {mode:5s}  max err rad<50: {seg(0,50):.2f}  50-200: {seg(50,200):.2f}  200-{R}: {seg(200,1e9):.2f}", flush=True)
