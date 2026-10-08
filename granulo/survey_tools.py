"""Software survey: do granulometries as implemented in widely used tools give non-monotone size distributions?

Tested on the 400 random images of Table 1 (same generator and seed as nonmono_sup.py), integer radii 1..15:
  - Scientific Python Lectures, "granulometry" example (code copied verbatim: disk_structure + ndimage.binary_opening)
  - DIPlib 3.6.1, dip.Granulometry(type="isotropic"), scales = diameters 2r+1 (the default options),
    and its default scales (12 values, sqrt(2)-spaced)
  - PoreSpy local thickness semantics (largest inscribed disk covering a pixel) = sup-closure, for reference
A size distribution is non-monotone if the opened area increases from one scale to the next."""
import diplib as dip
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk


def disk_structure(n):  # verbatim from Scientific Python Lectures, advanced/image_processing
    struct = np.zeros((2 * n + 1, 2 * n + 1))
    x, y = np.indices((2 * n + 1, 2 * n + 1))
    mask = (x - n) ** 2 + (y - n) ** 2 <= n ** 2
    struct[mask] = 1
    return struct.astype(bool)


def images():
    rng = np.random.default_rng(0)
    for trial in range(400):
        n = 120
        img = np.zeros((n, n), bool)
        kind = trial % 3
        for _ in range(rng.integers(3, 12)):
            y, x = rng.integers(10, n - 10, 2)
            if kind == 0:
                rr = rng.integers(3, 15); yy, xx = np.ogrid[:n, :n]; img |= (yy - y) ** 2 + (xx - x) ** 2 <= rr ** 2
            elif kind == 1:
                h, w = rng.integers(3, 25, 2); img[y:y + h, x:x + w] = True
            else:
                rr = int(rng.integers(2, 14)); D = disk(rr); y0, x0 = max(0, y - rr), max(0, x - rr)
                sub = img[y0:y0 + D.shape[0], x0:x0 + D.shape[1]]
                sub |= D[:sub.shape[0], :sub.shape[1]].astype(bool)
        yield img


def increases(seq, tol=1e-9):
    return sum(seq[i + 1] > seq[i] + tol for i in range(len(seq) - 1))


R = range(1, 16)
res = {"lectures": [0, 0], "dip_int": [0, 0], "dip_default": [0, 0]}
example = None
for k, img in enumerate(images()):
    a = [int(ndi.binary_opening(img, structure=disk_structure(n)).sum()) for n in R]
    f = img.astype(np.float32)
    # DIPlib returns Y = (mean(opened) - mean(in)) / (min(in) - mean(in)): it increases as the opened area decreases
    dI = dip.Granulometry(dip.Image(f), scales=[2.0 * r + 1 for r in R])
    yI = [-y for y in dI.Y()]  # so that, like the area, it should be non-increasing
    dD = dip.Granulometry(dip.Image(f))
    yD = [-y for y in dD.Y()]
    for key, seq in (("lectures", a), ("dip_int", yI), ("dip_default", yD)):
        c = increases(seq)
        res[key][0] += c > 0
        res[key][1] += c
    if example is None and increases(yI) and increases(a):
        example = (k, a, [round(-v, 4) for v in yI])
for key, (imgs, total) in res.items():
    print(f"{key:12s}: non-monotone on {imgs}/400 images ({total} increases)")
print("example image", example[0], "\n  lectures areas:", example[1], "\n  DIPlib Y:", example[2])
print("DIPlib default scales:", [round(x, 2) for x in dD.X()])
