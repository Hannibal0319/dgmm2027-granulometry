"""Granulometric chains among origin-centred digital disks D_n = {|p|^2 <= n}: for each allowed
radius gap G, the largest radius reachable by a chain D_{n0} < D_{n1} < ... with consecutive
elements open (hence a granulometry) and radius gaps <= G."""
import math, json, numpy as np
from scipy import ndimage as ndi

def norms(N):
    return sorted({x * x + y * y for x in range(int(math.isqrt(N)) + 1) for y in range(x + 1) if x * x + y * y <= N})

def D(n):
    r = math.isqrt(n); y, x = np.mgrid[-r:r + 1, -r:r + 1]; return (x * x + y * y) <= n

def is_open(m, n):
    B, A = D(m), D(n); pad = A.shape[0]; Bp = np.pad(B, pad)
    return np.array_equal(ndi.binary_opening(Bp, structure=A), Bp)

RMAX = 30
ns = norms(RMAX * RMAX)
op = {}
res = {}
for G in (1, 2, 3, 5, 8):
    reach = {0}  # the single point D_0
    for i, m in enumerate(ns[1:], 1):
        rm = math.sqrt(m)
        for j in range(i - 1, -1, -1):
            n = ns[j]
            if rm - math.sqrt(n) > G: break
            if n in reach:
                if (m, n) not in op: op[(m, n)] = is_open(m, n)
                if op[(m, n)]: reach.add(m); break
    res[G] = math.sqrt(max(reach))
    print(f"gap <= {G}: largest reachable radius {res[G]:.2f}", flush=True)
json.dump(res, open("gauss_chain.json", "w"))
