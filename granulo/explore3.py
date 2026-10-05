import numpy as np, math, itertools
from explore2 import lattice_points_of_sum, dsum
rng = np.random.default_rng(1)
prims = [(a,b) for a in range(-6,7) for b in range(-6,7) if (a,b)!=(0,0) and math.gcd(a,b)==1]
def para(v, w): return {(0,0), v, w, (v[0]+w[0], v[1]+w[1])}
uni = [(v,w) for v in prims for w in prims if v[0]*w[1]-v[1]*w[0]==1]
print(len(uni), 'unimodular pairs')
for trial_kind in ('para', 'para+seg'):
    bad = 0
    for t in range(400):
        k = rng.integers(1, 6)
        sets = [para(*uni[i]) for i in rng.integers(len(uni), size=k)]
        if trial_kind == 'para+seg':
            v = prims[rng.integers(len(prims))]; sets.append({(0,0), v})
        D = {(0,0)}
        for S in sets: D = dsum(D, S)
        Z = lattice_points_of_sum(sets)
        if D != Z:
            bad += 1
            if bad < 3: print(trial_kind, 'mismatch', [sorted(S) for S in sets], len(D), len(Z))
    print(trial_kind, 'mismatches', bad, '/ 400')
