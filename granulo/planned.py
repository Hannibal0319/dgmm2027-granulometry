import math, sys, numpy as np
from chain import Shape, generators

def ideal_counts(r, gens_sorted):
    """Circumscribed polygon of radius r with edge directions = all orbit directions:
    edge with direction angle a gets length r*(tan(d-/2)+tan(d+/2)); count = length/|v|."""
    dirs = []
    for key, es in gens_sorted:
        a, b = key
        for v in {(a,b),(b,a),(-a,b),(-b,a)}:
            for s in (1, -1):
                dirs.append((math.atan2(s*v[1], s*v[0]) % (2*math.pi), key, math.hypot(*v)))
    dirs = sorted(set(dirs))
    angs = np.array([d[0] for d in dirs])
    gap_prev = (angs - np.roll(angs, 1)) % (2*math.pi)
    gap_next = (np.roll(angs, -1) - angs) % (2*math.pi)
    L = r * (np.tan(gap_prev/2) + np.tan(gap_next/2))
    x = {}
    for (ang, key, nv), l in zip(dirs, L):
        x.setdefault(key, []).append(l / nv)
    # an orbit adds each of its directions once (both signs): count per orbit = mean over its 2*|orbit| edges
    return {k: float(np.mean(v)) for k, v in x.items()}

def shape_from_counts(counts, gens):
    E = [np.array(es) * c for k, es in gens if (c := counts.get(k, 0)) > 0]
    return Shape.polygon(None, np.vstack(E))

def run(R, kappa, mode, M=60):
    gens = generators(M)
    norm = {k: math.hypot(*k) for k, _ in gens}
    counts = {}
    out = []
    for r in range(1, R + 1):
        m = max(1.0, kappa * r ** (1/3))
        active = [(k, es) for k, es in gens if norm[k] <= m + 1e-9]
        x = ideal_counts(r, active)
        for k, xv in x.items():
            # inscribed-ish correction: DSS hulls add ~ width; ignore here
            t = math.floor(xv) if mode == 'floor' else int(round(xv))
            counts[k] = max(counts.get(k, 0), t)
        V = shape_from_counts(counts, gens)
        e, rr, c = Shape.error(V)
        out.append((r, rr, e, len(active)))
    return out

if __name__ == '__main__':
    R = int(sys.argv[1]); 
    for kappa in (0.5, 0.8, 1.2):
        for mode in ('floor', 'round'):
            o = run(R, kappa, mode)
            errs = np.array([e for r, rr, e, n in o])
            print(f"kappa={kappa} {mode:5s} max err r<=100: {errs[:100].max():.3f}  r in (R/2,R]: {errs[R//2:].max():.3f}  final dirs {o[-1][3]}")
