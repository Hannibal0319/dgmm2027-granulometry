import numpy as np, math
from explore2 import lattice_points_of_sum, dsum, dss
rng = np.random.default_rng(2)
prims = [(a,b) for a in range(-7,8) for b in range(-7,8) if (a,b)!=(0,0) and math.gcd(a,b)==1]
def para(v, w): return {(0,0), v, w, (v[0]+w[0], v[1]+w[1])}
uni = [(v,w) for v in prims for w in prims if v[0]*w[1]-v[1]*w[0]==1]
SQ = {(0,0),(1,0),(0,1),(1,1)}
def test(name, make, with_sq, n=400):
    bad = 0
    for t in range(n):
        k = rng.integers(1, 6)
        sets = ([SQ] if with_sq else []) + [make() for _ in range(k)]
        D = {(0,0)}
        for S in sets: D = dsum(D, S)
        if len({p for S in sets for p in S}) < 3: continue
        try: Z = lattice_points_of_sum(sets)
        except Exception: continue
        bad += D != Z
    print(f"{name:12s} square={with_sq}: mismatches {bad}/{n}")
for sq in (False, True):
    test('dss8', lambda: dss(prims[rng.integers(len(prims))], 8), sq)
    test('dss4', lambda: dss(prims[rng.integers(len(prims))], 4), sq)
    test('parallelogram', lambda: para(*uni[rng.integers(len(uni))]), sq)
