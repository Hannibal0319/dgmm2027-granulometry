"""Structuring-element families and exact binary morphology on Z^2 (no origin ambiguity).

X (+) S = {x + s},  X (-) S = {x : x + s in X for all s};  opening = dilate(erode(X, S), S).
A Minkowski-chain element is applied as a cascade over its summands (fast, exact).
"""
import numpy as np
from skimage.morphology import disk

from geom import Generator


def _shift(X, dy, dx, fill):
    Y = np.full_like(X, fill)
    H, W = X.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    Y[yd, xd] = X[ys, xs]
    return Y


def dilate(X, S):
    out = np.zeros_like(X)
    for x, y in S:
        out |= _shift(X, int(y), int(x), False)
    return out


def erode(X, S):
    out = np.ones_like(X)
    for x, y in S:
        out &= _shift(X, -int(y), -int(x), False)
    return out


def open_cascade(X, summands):
    Y = X
    for S in summands:
        Y = erode(Y, S)
    for S in reversed(summands):
        Y = dilate(Y, S)
    return Y


def gauss_disk_points(r):
    D = disk(r)
    ys, xs = np.nonzero(D)
    return [list(zip(xs - r, ys - r))]


def chain_summands(keys, counts):
    """Summands (lists of points) of the Minkowski-chain element with given orbit counts."""
    out = []
    for k, c in zip(keys, counts):
        if c <= 0:
            continue
        g = Generator(*k)
        for S in g.sets:
            out += [list(map(tuple, S))] * int(c)
    return out


def element_mask(summands):
    pts = {(0, 0)}
    for S in summands:
        pts = {(p[0] + s[0], p[1] + s[1]) for p in pts for s in S}
    P = np.array(sorted(pts))
    P -= P.min(0)
    M = np.zeros((P[:, 1].max() + 1, P[:, 0].max() + 1), bool)
    M[P[:, 1], P[:, 0]] = True
    return M
