import sys, numpy as np
from skimage.morphology import disk
from geom import Generator, polygon_from_counts, roundness, hull_vertices
d = np.load(sys.argv[1]); X = d["X"]; keys = [tuple(k) for k in d["keys"]]; radii = d["radii"]
gens = [Generator(*k) for k in keys]
def metric(V, r, delta=0.5):
    R = np.linalg.norm(V, axis=1).max(); W = np.roll(V, -1, 0); dd = W - V; nn = np.hypot(*dd.T); k = nn > 1e-12
    rin = (np.abs(V[k,0]*dd[k,1] - V[k,1]*dd[k,0]) / nn[k]).min()
    # min over rho in [r-delta, r+delta] of max(R - rho, rho - rin)
    rho = np.clip((R + rin) / 2, r - delta, r + delta)
    return max(R - rho, rho - rin), (R - rin) / 2
print(" r   chain(t, round)   gauss(t, round)")
for j, r in enumerate(radii):
    counts = {k: int(v) for k, v in zip(keys, X[j]) if v > 0}
    tc, rc = metric(polygon_from_counts(gens, counts), r)
    ys, xs = np.nonzero(disk(int(r))); V = hull_vertices(np.stack([xs, ys], 1).astype(float)); V -= (V.max(0)+V.min(0))/2
    tg, rg = metric(V, r)
    print(f"{int(r):3d}   {tc:5.2f} {rc:5.2f}        {tg:5.2f} {rg:5.2f}   {dict((k,v) for k,v in counts.items())}")
