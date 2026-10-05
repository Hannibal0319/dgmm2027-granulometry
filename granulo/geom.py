"""Geometry of granulometric chains of digital segments.

A generator is the D4-orbit of the digital straight segment (8-connected, naive) from 0 to a
primitive vector (a, b), a >= b >= 0.  Every element of a chain is the Minkowski sum of
generators; its digital set is exactly the lattice-point set of the real Minkowski sum of
the convex hulls (checked in explore2.py), so all geometry is computed on support functions.
"""
import math

import numpy as np
from scipy.spatial import ConvexHull

D4 = [np.array(m) for m in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]],
                            [[1, 0], [0, -1]], [[0, 1], [1, 0]], [[-1, 0], [0, 1]], [[0, -1], [-1, 0]])]


def dss(a, b):
    """Naive (8-connected) digital straight segment from (0,0) to (a,b), a >= b >= 0, gcd = 1.
    Points (x, floor(b x / a + 1/2)) with the tie convention making it symmetric."""
    pts = []
    for x in range(a + 1):
        # symmetric rounding: round half towards the segment midpoint
        y2 = 2 * b * x + a  # 2*(b x / a + 1/2) * a
        y = y2 // (2 * a)
        if y2 % (2 * a) == 0 and 2 * x > a:
            y -= 1
        pts.append((x, y))
    return np.array(pts)


def canon(points):
    P = np.array(points)
    P = P - P.min(0)
    return tuple(sorted(map(tuple, P)))


def orbit_sets(a, b):
    """Distinct (up to translation) images of DSS(a,b) under D4."""
    base = dss(a, b)
    seen = {}
    for g in D4:
        S = base @ g.T
        seen.setdefault(canon(S), S)
    return list(seen.values())


def primitive_orbits(m):
    """Canonical representatives (a, b), a >= b >= 0, gcd = 1, |(a,b)| <= m."""
    out = []
    for a in range(1, int(m) + 1):
        for b in range(0, a + 1):
            if math.gcd(a, b) == 1 and math.hypot(a, b) <= m + 1e-12:
                out.append((a, b))
    return sorted(out, key=lambda v: (math.hypot(*v), v))


def hull_vertices(P):
    P = np.asarray(P, float)
    if len(P) < 3 or np.linalg.matrix_rank(P - P[0]) < 2:
        o = np.lexsort((P[:, 1], P[:, 0]))
        return P[[o[0], o[-1]]]
    return P[ConvexHull(P).vertices]


class Generator:
    def __init__(self, a, b):
        self.key = (a, b)
        self.sets = orbit_sets(a, b)
        self.hulls = [hull_vertices(S) for S in self.sets]
        self.size = sum(len(S) for S in self.sets)

    def support(self, U):
        """Sum over the orbit of the support functions, centred: (n_dirs,) array."""
        h = np.zeros(len(U))
        for V in self.hulls:
            hv = (U @ V.T).max(1)
            hm = (-U @ V.T).max(1)
            h += 0.5 * (hv + hm)  # half-width: support of the centred (symmetrised) body
        return h

    def edges(self):
        E = []
        for V in self.hulls:
            if len(V) == 2:
                d = V[1] - V[0]
                E += [d, -d]
            else:
                E += [V[(i + 1) % len(V)] - V[i] for i in range(len(V))]
        return E


def polygon_from_counts(gens, counts):
    """Vertices of the Minkowski sum (edge merging), centred at its centre of symmetry."""
    E = []
    for g in gens:
        c = counts.get(g.key, 0)
        if c > 0:
            E.extend(np.array(e) * c for e in g.edges())
    E = np.array(E)
    E = E[np.argsort(np.arctan2(E[:, 1], E[:, 0]), kind="stable")]
    V = np.cumsum(E, 0)
    return V - (V.max(0) + V.min(0)) / 2  # D4-symmetric, hence centrally symmetric


def roundness(V):
    """Exact Hausdorff distance between the centrally symmetric polygon V (centred) and the
    best concentric disk: (R_out - r_in)/2, together with that disk's radius."""
    R = np.linalg.norm(V, axis=1).max()
    W = np.roll(V, -1, 0)
    d = W - V
    nn = np.hypot(d[:, 0], d[:, 1])
    k = nn > 1e-12
    r = (np.abs(V[k, 0] * d[k, 1] - V[k, 1] * d[k, 0]) / nn[k]).min()
    return (R - r) / 2, (R + r) / 2
