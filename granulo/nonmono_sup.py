"""Same 400 random images as Table 1 (nonmono.py, seed 0): single disk openings vs their sup-closure."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

rng = np.random.default_rng(0)
plain = sup = 0
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
            sub = img[y0:y0 + D.shape[0], x0:x0 + D.shape[1]]; sub |= D[:sub.shape[0], :sub.shape[1]].astype(bool)
    O = [ndi.binary_opening(img, structure=disk(r).astype(bool)) for r in range(1, 16)]
    a = [int(o.sum()) for o in O]
    acc = np.zeros_like(img); b = [0] * 15
    for i in range(14, -1, -1):
        acc |= O[i]; b[i] = int(acc.sum())
    plain += any(a[i + 1] > a[i] for i in range(14))
    sup += any(b[i + 1] > b[i] for i in range(14))
print(f"non-monotone: single openings {plain}/400, sup-closure {sup}/400")
