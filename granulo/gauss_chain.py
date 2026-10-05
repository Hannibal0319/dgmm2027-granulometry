"""Chains of origin-centred digital disks D_n = {p in Z^2 : |p|^2 <= n} that are granulometric
(D_m is D_n-open for consecutive elements, hence for all pairs by transitivity)."""
import math, sys, numpy as np
from scipy import ndimage as ndi

def norms(N):
    s = set()
    for x in range(0, int(math.isqrt(N)) + 1):
        for y in range(0, x + 1):
            if x * x + y * y <= N: s.add(x * x + y * y)
    return sorted(s)

def D(n):
    r = math.isqrt(n); y, x = np.mgrid[-r:r + 1, -r:r + 1]; return (x * x + y * y) <= n

def is_open(B, A):
    pad = A.shape[0]; Bp = np.pad(B, pad)
    return np.array_equal(ndi.binary_opening(Bp, structure=A), Bp)

Rmax = float(sys.argv[1])
ns = norms(int(Rmax ** 2))
print(len(ns), "distinct digital disks up to radius", Rmax)
# for each n, which later m (within radius gap <= 3) are D_n-open
best = {ns[0]: (0.0, None)}  # minimal possible max radius gap along a chain ending at n
order = ns
gapmax = {}
prev = {}
import bisect
for i, m in enumerate(ns[1:], 1):
    rm = math.sqrt(m); Dm = D(m)
    cand = None
    for j in range(i - 1, -1, -1):
        n = ns[j]; rn = math.sqrt(n)
        if rm - rn > 3.0: break
        if n not in best: continue
        g = max(best[n][0], rm - rn)
        if cand is not None and g >= cand[0]: continue
        if is_open(Dm, D(n)):
            cand = (g, n)
    if cand is not None: best[m] = cand
# reconstruct best chain to the largest reachable n
reach = [n for n in best if math.sqrt(n) >= Rmax - 3]
end = min(reach, key=lambda n: best[n][0])
chain = [end]
while best[chain[-1]][1] is not None: chain.append(best[chain[-1]][1])
chain = chain[::-1]
rad = [math.sqrt(n) for n in chain]
print("max radius gap along best chain:", round(best[end][0], 3), " chain length", len(chain))
print("radii:", [round(r, 2) for r in rad])
