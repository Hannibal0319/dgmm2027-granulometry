"""Multiscale chain of Theorem 2: stage j uses m_j = 2^j and adds radius T_j = c 16^j (c = 8 sqrt 2),
on top of the frozen previous stages. Claim: eps = O(sqrt(rho)), mean-radius steps <= sqrt2 m_j / pi."""
import math, json, numpy as np
from upper import directions, edge_lengths
from dither_chain import cheb_disk, U as U2

c = 8 * math.sqrt(2)
h = np.zeros(len(U2)); out = []; base = 0.0
for j in range(4):
    m = 2 ** j; T = c * 16 ** j
    V = directions(m); L, g = edge_lengths(V); nv = np.array([math.hypot(*v) for v in V])
    hs = np.stack([np.abs(U2 @ np.array(v, float)) / 2 for v in V], 1)
    ev = sorted((k * nv[i] / L[i], i) for i in range(len(V)) for k in range(1, int(T * L[i] / nv[i]) + 1))
    every = max(1, len(ev) // 400)
    for e_i, (t, i) in enumerate(ev):
        h = h + hs[:, i]
        if e_i % every == 0:
            e, rho = cheb_disk(h); out.append((j, rho, e))
    print(f"stage {j}: m={m} T={T:.0f} events={len(ev)}", flush=True)
o = np.array(out)
r = o[:, 2] / np.sqrt(o[:, 1])
print("max eps/sqrt(rho) for rho>20:", r[o[:, 1] > 20].max().round(3), " eps at end:", o[-1, 2].round(2), " rho at end:", o[-1, 1].round(0))
for lo, hi in ((20, 200), (200, 3000), (3000, 50000)):
    k = (o[:, 1] >= lo) & (o[:, 1] < hi)
    print(f"  rho in [{lo},{hi}): max eps {o[k,2].max():.2f}, max eps/sqrt(rho) {r[k].max():.3f}")
json.dump(dict(rho=o[:, 1].tolist(), eps=o[:, 2].tolist()), open("upper_multi.json", "w"))
