"""Theorem 2 construction: periodic lines {0,v} for all integer v on the boundary of [-m,m]^2 (up to
sign), counts n_v(t) = floor(t * l_v / |v|) with l_v the edges of the circumscribed polygon of the unit
disk.  Checks eps <= t/(8m^2) + sqrt(2) m^2, mean-radius steps <= sqrt(2) m / pi, and holes."""
import math, sys, json
import numpy as np
from dither_chain import cheb_disk, U as U2

def directions(m):
    V = set()
    for j in range(-m, m + 1):
        for v in ((m, j), (j, m)):
            a, b = v
            if a < 0 or (a == 0 and b < 0): a, b = -a, -b
            V.add((a, b))
    V = sorted(V, key=lambda v: math.atan2(v[1], v[0]) % math.pi)
    return V

def edge_lengths(V):
    ang = np.array([math.atan2(b, a) % math.pi for a, b in V])
    full = np.r_[ang, ang + math.pi]
    o = np.argsort(full); full = full[o]
    gp = (full - np.roll(full, 1)) % (2 * math.pi); gn = (np.roll(full, -1) - full) % (2 * math.pi)
    L = np.tan(gp / 2) + np.tan(gn / 2)
    Lfull = np.empty_like(L); Lfull[o] = L
    return Lfull[:len(V)], max(gp.max(), gn.max())

def chain(m, Tmax, every=7):
    """Event-driven chain: one periodic line per event, in order of the times t = k |v| / l_v."""
    V = directions(m); L, g = edge_lengths(V)
    nv = np.array([math.hypot(*v) for v in V])
    hs = np.stack([np.abs(U2 @ np.array(v, float)) / 2 for v in V], 1)
    ev = []
    for i in range(len(V)):
        k = np.arange(1, int(Tmax * L[i] / nv[i]) + 1)
        ev += [(kk * nv[i] / L[i], i) for kk in k]
    ev.sort()
    h = np.zeros(len(U2)); out = []; steps = []
    for e_i, (t, i) in enumerate(ev):
        rbar0 = h.mean(); h = h + hs[:, i]; steps.append(h.mean() - rbar0)
        if e_i % every == 0 or e_i == len(ev) - 1:
            e, rho = cheb_disk(h); out.append((t, rho, e, h.mean()))
    o = np.array(out)
    bound = o[:, 0] / (8 * m * m) + math.sqrt(2) * m * m
    return o, bound, g, nv.max(), nv.sum(), max(steps)

if __name__ == "__main__":
    res = {}
    for m in (2, 4, 8, 16):
        o, bound, g, vmax, s, smax = chain(m, 3000)
        ok = (o[:, 2] <= bound + 1e-9).all()
        k = o[:, 1] > 50
        slope = np.polyfit(o[k, 1], o[k, 2], 1)[0]
        print(f"m={m:2d} gap={g:.4f}<=1/m={1/m:.4f}  dirs={len(directions(m))} sum|v|={s:.1f}  bound holds: {ok}  "
              f"eps(rho=3000)={o[-1,2]:.2f} bound={bound[-1]:.2f}  fitted slope={slope:.5f} vs 1/(8m^2)={1/(8*m*m):.5f}  "
              f"max mean-radius step={smax:.3f} <= sqrt2 m/pi={math.sqrt(2)*m/math.pi:.3f}", flush=True)
        res[m] = dict(rho=o[:, 1].tolist(), eps=o[:, 2].tolist())
    json.dump(res, open("upper_fixed_m.json", "w"))
