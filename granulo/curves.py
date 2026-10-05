"""Lower-bound curve of Theorem 1 and the octagon (neighbourhood-sequence) chain error."""
import math, numpy as np
from scipy.optimize import brentq
from geom import Generator, polygon_from_counts, roundness

def lower_bound(rho, Delta, rho0=0.0):
    """smallest eps >= 0 with 8 eps >= 2 (rho - rho0 - 2 eps)(1 - cos(alpha/2)), alpha = atan(1/(2Delta+4eps))"""
    f = lambda e: 8 * e - 2 * (rho - rho0 - 2 * e) * (1 - math.cos(math.atan(1 / (2 * Delta + 4 * e)) / 2))
    if f(0) >= 0: return 0.0
    return brentq(f, 0, rho)

def octagon_chain(R):
    gens = [Generator(1, 0), Generator(1, 1)]
    out = []; c = np.array([1, 0])
    best_prev = c
    for r in range(1, R + 1):
        # best monotone (c10, c11) with radius closest to r and small roundness: brute force near prev
        cand = []
        for a in range(best_prev[0], best_prev[0] + 4):
            for b in range(best_prev[1], best_prev[1] + 4):
                V = polygon_from_counts(gens, {(1, 0): a, (1, 1): b}) if b > 0 else polygon_from_counts(gens, {(1, 0): a})
                e, rad = roundness(V)
                cand.append((max(e, abs(rad - r) - 0.5), e, rad, a, b))
        t, e, rad, a, b = min(cand)
        best_prev = np.array([a, b]); out.append((r, rad, e))
    return np.array(out)

if __name__ == "__main__":
    for rho in (10, 100, 1000, 10000):
        print(rho, [round(lower_bound(rho, D), 3) for D in (0.5, 1, 2)])
    oc = octagon_chain(1000)
    np.save("octagon_chain.npy", oc)
    print("octagon chain errors:", [(int(r), round(e, 2)) for r, rad, e in oc[[9, 49, 99, 499, 999]]])
