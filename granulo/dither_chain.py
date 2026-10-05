"""Chain of single-DSS increments obtained by phase-dithered monotone rounding of the joint-LP
plan; every addition yields a new element. Error = Chebyshev fit of the support function by a
disk with free centre (exact for polygons via dense theta grid)."""
import math, sys, numpy as np
from scipy.optimize import linprog
from geom import Generator

TH = np.linspace(0, 2 * math.pi, 2048, endpoint=False)
U = np.stack([np.cos(TH), np.sin(TH)], 1)

def cheb_disk(h):
    """min_{rho,c} max |h - rho - c.u|  -> (eps, rho)"""
    A = np.stack([np.ones_like(TH), U[:, 0], U[:, 1]], 1)
    c = np.r_[0, 0, 0, 1.0]
    Aub = np.vstack([np.hstack([A, -np.ones((len(TH), 1))]), np.hstack([-A, -np.ones((len(TH), 1))])])
    bub = np.r_[h, -h]
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * 3 + [(0, None)], method="highs")
    return res.x[3], res.x[0]

def member_support(S):
    return (U @ S.T).max(1)

def run(plan, dr=0.05, out_every=1):
    d = np.load(plan); X, radii = d["X"], d["radii"]; keys = [tuple(k) for k in d["keys"]]
    gens = [Generator(*k) for k in keys]
    members = []  # (gen index, support vector)
    for gi, g in enumerate(gens):
        for S in g.sets: members.append((gi, member_support(S)))
    nm = len(members)
    orbit_size = np.array([len(gens[gi].sets) for gi, _ in members])
    phase = np.zeros(nm)
    for gi in range(len(gens)):
        idx = [i for i, (g, _) in enumerate(members) if g == gi]
        for t, i in enumerate(idx): phase[i] = t / len(idx)
    cnt = np.zeros(nm, int)
    h = np.zeros(len(TH))
    # unit square first
    for i, (gi, s) in enumerate(members):
        if keys[gi] == (1, 0): cnt[i] = 1; h += s
    elems = []
    r = radii[0] * 0.5
    while r <= radii[-1]:
        j = np.searchsorted(radii, r)
        if j == 0: x = X[0] * r / radii[0]
        else:
            jj = min(j, len(radii) - 1); w = (r - radii[jj-1]) / (radii[jj] - radii[jj-1]); x = (1 - w) * X[jj-1] + w * X[jj]
        target = np.floor(np.array([x[gi] for gi, _ in members]) + phase)
        need = np.where(target > cnt)[0]
        # add one DSS at a time (largest deficit first), recording each element
        while len(need):
            i = need[np.argmax(target[need] - cnt[need])]
            cnt[i] += 1; h += members[i][1]
            elems.append(h.copy())
            need = np.where(target > cnt)[0]
        r += dr
    return elems

if __name__ == "__main__":
    elems = run(sys.argv[1])
    sel = list(range(0, len(elems), max(1, len(elems) // 400))) + [len(elems) - 1]
    res = []
    for k in sel:
        e, rho = cheb_disk(elems[k]); res.append((k, rho, e))
    res = np.array(res)
    # radius step between consecutive elements (bounded by the largest single increment)
    rhos_all = [cheb_disk(elems[k])[1] for k in range(0, len(elems), max(1, len(elems)//2000))]
    np.save(sys.argv[1].replace('.npz', '_dither.npy'), res)
    print("elements:", len(elems))
    for lo, hi in ((5, 50), (50, 100), (100, 200), (200, 400), (400, 700), (700, 1e9)):
        k = (res[:, 1] >= lo) & (res[:, 1] < hi)
        if k.any(): print(f"  rho in [{lo},{hi}): max eps {res[k,2].max():.3f}  mean {res[k,2].mean():.3f}")
    print("max radius step (sampled):", np.max(np.diff(rhos_all)).round(3))
