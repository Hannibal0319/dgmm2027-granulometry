"""kappa(L) of Corollary 1: min (1/4) A(sum lambda_C w_C) s.t. min_theta sum lambda_C w_C >= 2 (LP)."""
import math, json, numpy as np
from scipy.optimize import linprog
from geom import Generator, primitive_orbits, D4

TH = np.linspace(0, math.pi, 7200, endpoint=False)
U = np.stack([np.cos(TH), np.sin(TH)], 1)

def width(S):
    P = np.asarray(S, float); z = U @ P.T
    return z.max(1) - z.min(1)

def kappa(sets):
    Wm = np.stack([width(S) for S in sets], 1); n = Wm.shape[1]
    # vars lambda (n), t ; minimise t ; W lambda >= 2 ; W lambda <= 2 + 4 t
    c = np.r_[np.zeros(n), 1.0]
    A = np.vstack([np.hstack([-Wm, np.zeros((len(TH), 1))]), np.hstack([Wm, -4 * np.ones((len(TH), 1))])])
    b = np.r_[-2 * np.ones(len(TH)), 2 * np.ones(len(TH))]
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None)] * (n + 1), method="highs")
    return res.x[-1]

def orbit_members(keys):
    out = []
    for k in keys: out += [S for S in Generator(*k).sets]
    return out

def segment_members(keys):
    out = []
    for a, b in keys:
        for v in {(a, b), (b, a), (-a, b), (-b, a)}: out.append([(0, 0), v])
    return out

N4 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
N8 = [(x, y) for x in (-1, 0, 1) for y in (-1, 0, 1)]
rows = {}
rows["N4 (diamonds)"] = kappa([N4])
rows["N8 (squares)"] = kappa([N8])
rows["N4+N8 (octagons, neighbourhood seq.)"] = kappa([N4, N8])
for m in (1.0, math.sqrt(2), math.sqrt(5), math.sqrt(10), 5, 10, 20):
    keys = primitive_orbits(m)
    rows[f"8-conn DSS, |v|<={m:.2f} ({len(keys)} orbits)"] = kappa(orbit_members(keys))
    rows[f"segments/periodic lines, |v|<={m:.2f}"] = kappa(segment_members(keys))
for k, v in rows.items(): print(f"{k:45s} kappa = {v:.4f}   eps at rho=100: {100*v:.2f}")
json.dump(rows, open("kappa.json", "w"), indent=1)
