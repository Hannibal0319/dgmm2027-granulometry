"""Do the granulometries of current software increase on real images?

DIPlib 3.6.1 dip.Granulometry (isotropic, default options and scales, and integer diameters) on grey-value and
Otsu-binarised scikit-image sample images, and the Scientific Python Lectures code on its own example image
(seeded exactly as in the current lecture notes) with every integer radius instead of the 5 radii it plots."""
import diplib as dip
import numpy as np
from scipy import ndimage as ndi
from skimage import data, filters


def disk_structure(n):  # verbatim from Scientific Python Lectures
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n) ** 2 + (y - n) ** 2 <= n ** 2
    struct[mask] = 1
    return struct.astype(bool)


def increases(seq, tol=1e-9):
    return [i for i in range(len(seq) - 1) if seq[i + 1] > seq[i] + tol]


# 1) the lecture notes' own example image
rng = np.random.default_rng(27446968)
n, l = 10, 256
im = np.zeros((l, l))
points = l * rng.random((2, n ** 2))
im[(points[0]).astype(int), (points[1]).astype(int)] = 1
im = ndi.gaussian_filter(im, sigma=l / (4. * n))
mask = im > im.mean()
sizes = list(range(1, 25))
g = [int(ndi.binary_opening(mask, structure=disk_structure(k)).sum()) for k in sizes]
inc = increases(g)
print("Lectures example image, radii 1..24: increases at", [(sizes[i], g[i], g[i + 1]) for i in inc])

# 2) DIPlib on real images
imgs = {"coins": data.coins(), "human_mitosis": data.human_mitosis(), "gravel": data.gravel(),
        "camera": data.camera(), "moon": data.moon(), "cells3d_slice": data.cells3d()[30, 1]}
for name, a in imgs.items():
    a = a.astype(np.float32)
    b = (a > filters.threshold_otsu(a)).astype(np.float32)
    for kind, x in (("grey", a), ("binary", b)):
        out = []
        for lab, sc in (("default", None), ("diam 3..41", [2.0 * r + 1 for r in range(1, 21)])):
            d = dip.Granulometry(dip.Image(x)) if sc is None else dip.Granulometry(dip.Image(x), scales=sc)
            Y = list(d.Y())  # cumulative distribution, should be non-decreasing
            dec = [(round(d.X()[i], 2), round(Y[i], 4), round(Y[i + 1], 4)) for i in range(len(Y) - 1)
                   if Y[i + 1] < Y[i] - 1e-9]
            out.append(f"{lab}: {len(dec)} decreases {dec[:3]}")
        print(f"{name:14s} {kind:6s} | " + " | ".join(out))
