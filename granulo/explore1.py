import numpy as np, itertools, math
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def dsum(A, B):
    return {(a[0]+b[0], a[1]+b[1]) for a in A for b in B}

def zonotope_points(gens):
    # lattice points of the real zonotope sum [0,g]
    verts = np.array([sum((np.array(g)*e for g,e in zip(gens,eps)), np.zeros(2)) for eps in itertools.product((0,1), repeat=len(gens))])
    hull = ConvexHull(verts)
    lo, hi = verts.min(0).astype(int), verts.max(0).astype(int)
    pts = set()
    eqs = hull.equations
    for x in range(lo[0], hi[0]+1):
        for y in range(lo[1], hi[1]+1):
            if np.all(eqs[:, :2] @ [x, y] + eqs[:, 2] <= 1e-9):
                pts.add((x, y))
    return pts

rng = np.random.default_rng(0)
prims = [(a,b) for a in range(-3,4) for b in range(-3,4) if (a,b)!=(0,0) and math.gcd(a,b)==1]
bad = 0; tot = 0
for trial in range(300):
    k = rng.integers(2, 7)
    gens = [prims[i] for i in rng.integers(len(prims), size=k)]
    if len({tuple(np.sign(g)*np.array(g)) if g[0]!=0 else (0,abs(g[1])) for g in gens}|({(1,0),(0,1)} if trial%2==0 else set()))<2: continue
    with_sq = trial % 2 == 0
    if with_sq: gens = [(1,0),(0,1)] + gens
    D = {(0,0)}
    for g in gens: D = dsum(D, {(0,0), g})
    Z = zonotope_points(gens)
    tot += 1
    if D != Z:
        bad += 1
        if bad <= 3: print('mismatch', with_sq, gens, len(D), len(Z))
print('mismatches', bad, 'of', tot)
