import math, numpy as np
from skimage.morphology import disk
from geom import hull_vertices, roundness, Generator, primitive_orbits
from lp4 import grid
from scipy.optimize import linprog

def digital_roundness(mask):
    ys, xs = np.nonzero(mask)
    P = np.stack([xs, ys], 1).astype(float)
    V = hull_vertices(P); V = V - (V.max(0) + V.min(0)) / 2
    return roundness(V)

print("Gauss disks (skimage.disk):")
for r in (5, 10, 20, 31, 50, 100):
    e, rad = digital_roundness(disk(r)); print(f"  r={r}: roundness {e:.3f} radius {rad:.2f}")

def lp_value(keys):
    gens = [Generator(*k) for k in keys]
    U = grid(3000); Hm = np.stack([g.support(U) for g in gens], 1); ng = len(gens)
    c = np.zeros(ng + 1); c[-1] = 1
    A = np.vstack([np.hstack([Hm, -np.ones((len(U), 1))]), np.hstack([-Hm, -np.ones((len(U), 1))])])
    res = linprog(c, A_ub=A, b_ub=np.r_[np.ones(len(U)), -np.ones(len(U))], bounds=[(0, None)] * (ng + 1), method="highs")
    return res.x[-1]

libs = {"square only {(1,0)}": [(1, 0)], "octagon {(1,0),(1,1)}": [(1, 0), (1, 1)],
        "+(2,1)": [(1, 0), (1, 1), (2, 1)], "+(3,1),(3,2)": [(1, 0), (1, 1), (2, 1), (3, 1), (3, 2)],
        "|v|<=5": primitive_orbits(5), "|v|<=10": primitive_orbits(10), "|v|<=20": primitive_orbits(20)}
print("Fixed generator libraries: asymptotic roundness slope E*(G) (error ~ E* r):")
for name, keys in libs.items():
    E = lp_value(keys); print(f"  {name:28s} E*={E:.4f}  -> r=50: {50*E:.2f}, r=100: {100*E:.2f}")
