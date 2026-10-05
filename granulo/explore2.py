import numpy as np, itertools, math
from scipy.spatial import ConvexHull

def dsum(A, B):
    return {(a[0]+b[0], a[1]+b[1]) for a in A for b in B}

def dss(v, conn):
    """digital straight segment from 0 to v (primitive): points of the naive (8-conn) or
    standard (4-conn) digital line through 0 with direction v, between 0 and v."""
    a, b = v
    pts = set()
    if conn == 8:
        n = max(abs(a), abs(b))
        for t in range(n + 1):
            pts.add((round(a * t / n + 1e-9), round(b * t / n + 1e-9)) if True else None)
    else:
        x = y = 0; pts.add((0,0)); sx = 1 if a>=0 else -1; sy = 1 if b>=0 else -1
        A, B = abs(a), abs(b); ex = ey = 0
        while (x, y) != (a, b):
            # step in x or y keeping closest to the line
            if A*(abs(y)+0.5) - B*(abs(x)+0.5) >= 0 and abs(x) < A or abs(y) == B:
                x += sx
            else:
                y += sy
            pts.add((x, y))
    return pts

def lattice_points_of_sum(sets):
    hulls = []
    V = np.zeros((1,2))
    for S in sets:
        P = np.array(sorted(S), float)
        V = (V[:,None,:] + P[None,:,:]).reshape(-1,2)
        if len(V) > 3:
            try:
                h = ConvexHull(V); V = V[h.vertices]
            except Exception: pass
    h = ConvexHull(V)
    lo, hi = V.min(0).astype(int), V.max(0).astype(int)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    P = np.stack([xs.ravel(), ys.ravel()], 1)
    ok = np.all(P @ h.equations[:, :2].T + h.equations[:, 2] <= 1e-9, axis=1)
    return {tuple(p) for p in P[ok]}

rng = np.random.default_rng(0)
prims = [(a,b) for a in range(-4,5) for b in range(-4,5) if (a,b)!=(0,0) and math.gcd(a,b)==1]
for conn in (8, 4):
    bad = tot = 0
    for trial in range(300):
        k = rng.integers(2, 7)
        gens = [(1,0),(0,1)] + [prims[i] for i in rng.integers(len(prims), size=k)]
        sets = [dss(g, conn) for g in gens]
        D = {(0,0)}
        for S in sets: D = dsum(D, S)
        Z = lattice_points_of_sum(sets)
        tot += 1
        if D != Z:
            bad += 1
            if bad <= 2: print(conn, 'mismatch', gens, len(D), len(Z))
    print('conn', conn, 'mismatches', bad, 'of', tot)
