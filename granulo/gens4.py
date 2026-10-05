"""Generators: D4-orbits of standard (4-connected) digital straight segments."""
import math
import numpy as np
from geom import D4, canon, hull_vertices, roundness
from explore2 import dss as dss_any

def dss4(a, b):
    return np.array(sorted(dss_any((a, b), 4)))

class Gen:
    def __init__(self, a, b):
        self.key = (a, b)
        base = dss4(a, b)
        seen = {}
        for g in D4:
            S = base @ g.T
            seen.setdefault(canon(S), S)
        self.sets = list(seen.values())
        self.hulls = [hull_vertices(S) for S in self.sets]
    def support(self, U):
        h = np.zeros(len(U))
        for V in self.hulls:
            h += 0.5 * ((U @ V.T).max(1) + (-U @ V.T).max(1))
        return h
    def edges(self):
        E = []
        for V in self.hulls:
            if len(V) == 2:
                d = V[1] - V[0]; E += [d, -d]
            else:
                E += [V[(i + 1) % len(V)] - V[i] for i in range(len(V))]
        return E

def orbits(M):
    out = []
    for a in range(1, int(M) + 1):
        for b in range(0, a + 1):
            if math.gcd(a, b) == 1 and math.hypot(a, b) <= M + 1e-12:
                out.append(Gen(a, b))
    return out

def polygon(gens, counts):
    E = []
    for g in gens:
        c = counts.get(g.key, 0)
        if c: E.extend(np.array(e, float) * c for e in g.edges())
    E = np.array(E)
    E = E[np.argsort(np.arctan2(E[:, 1], E[:, 0]), kind="stable")]
    V = np.cumsum(E, 0)
    return V - (V.max(0) + V.min(0)) / 2
