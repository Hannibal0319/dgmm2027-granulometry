"""Granulometric chains of digital segments: polygon geometry and greedy construction."""
import math
import numpy as np
from scipy.spatial import ConvexHull
from explore2 import dss

def hull_edges(points):
    P = np.array(sorted(points), float)
    if len(P) == 2 or np.linalg.matrix_rank(P - P[0]) < 2:
        d = P.max(0) - P.min(0) if True else None
        a = P[np.lexsort((P[:,1], P[:,0]))]
        d = a[-1] - a[0]
        return [d, -d]
    h = ConvexHull(P)
    V = P[h.vertices]  # ccw
    return [V[(i+1) % len(V)] - V[i] for i in range(len(V))]

def orbit(v):
    a, b = v
    dirs = {(a, b), (b, a), (-a, b), (-b, a)}
    canon = set()
    for x, y in dirs:
        if (x, y) == (0, 0): continue
        if x < 0 or (x == 0 and y < 0): x, y = -x, -y
        canon.add((x, y))
    return sorted(canon)

class Shape:
    """Convex lattice polygon as multiset of edge vectors (ccw), plus a count per generator."""
    def __init__(self):
        self.edges = np.zeros((0, 2))
        self.counts = {}
    def add(self, edge_list, key):
        self.edges = np.vstack([self.edges, np.array(edge_list)])
        self.counts[key] = self.counts.get(key, 0) + 1
    def polygon(self, edges=None):
        E = self.edges if edges is None else edges
        ang = np.arctan2(E[:,1], E[:,0])
        E = E[np.argsort(ang, kind='stable')]
        V = np.cumsum(E, 0)
        return V
    @staticmethod
    def error(V):
        """Hausdorff distance to the best concentric disk (center = centroid of vertices
        refined by a few Weiszfeld-like minimax steps) -> (err, radius, center)."""
        c = V.mean(0)
        best = None
        for _ in range(3):
            R = np.linalg.norm(V - c, axis=1).max()
            W = np.roll(V, -1, 0)
            d = W - V
            n = np.stack([d[:,1], -d[:,0]], 1)
            nn = np.linalg.norm(n, axis=1)
            keep = nn > 0
            dist = np.abs(((V - c) * n).sum(1))[keep] / nn[keep]
            r = dist.min()
            e = (R - r) / 2
            if best is None or e < best[0]: best = (e, (R + r) / 2, c.copy())
            break
        return best

def generators(M, conn=8):
    gens = []
    for a in range(0, M + 1):
        for b in range(0, a + 1):
            if math.gcd(a, b) != 1: continue
            if (a, b) == (0, 0): continue
            es = []
            for v in orbit((a, b)):
                es += hull_edges(dss(v, conn))
            gens.append(((a, b), es))
    return gens
