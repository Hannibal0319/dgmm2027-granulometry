"""Is D_{n+1} = Z_{n+1} cap Z^2 open w.r.t. D_n = Z_n cap Z^2 when Z_{n+1} = Z_n + [0, v]?"""
import math, numpy as np
from scipy import ndimage as ndi

def zono_mask(gens_counts, pad=2):
    # gens_counts: list of (v, c); zonotope sum c [0, v]
    E = []
    for v, c in gens_counts:
        E += [np.array(v, float) * c]
    # vertices via edge merge (central symmetric): edges +e and -e
    E2 = E + [-e for e in E]
    E2 = sorted(E2, key=lambda e: math.atan2(e[1], e[0]))
    V = np.cumsum(E2, 0); V -= V.min(0)
    W = int(np.ceil(V[:, 0].max())) + 1; Hh = int(np.ceil(V[:, 1].max())) + 1
    ys, xs = np.mgrid[0:Hh, 0:W]
    P = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    inside = np.ones(len(P), bool)
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        d = b - a
        if np.hypot(*d) < 1e-12: continue
        inside &= (d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) >= -1e-9
    M = inside.reshape(Hh, W)
    return np.pad(M, pad)

def is_open(B, A):
    pad = max(A.shape)
    Bp = np.pad(B, pad)
    return np.array_equal(ndi.binary_opening(Bp, structure=A), Bp)

rng = np.random.default_rng(0)
prims = [(a, b) for a in range(0, 8) for b in range(-7, 8) if math.gcd(a, b) == 1 and (a > 0 or b > 0)]
fails = tot = 0
for trial in range(300):
    seq = [(1, 0), (0, 1)] + [prims[i] for i in rng.integers(len(prims), size=rng.integers(3, 12))]
    counts = {}
    prevM = None
    for v in seq:
        counts[v] = counts.get(v, 0) + 1
        M = zono_mask(list(counts.items()), pad=0)
        M = M[np.ix_(M.any(1), M.any(0))]
        if prevM is not None and len(counts) >= 2:
            tot += 1
            if not is_open(M, prevM):
                fails += 1
                if fails <= 3: print("not open:", dict(counts))
        prevM = M
print(f"consecutive pairs not open: {fails}/{tot}")
