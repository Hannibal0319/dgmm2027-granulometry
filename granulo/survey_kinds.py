"""Which shapes make single disk openings non-monotone? The 400 random images of Table 1 by object type
(Euclidean disks, rectangles, digital disks), with the size of the negative part of the pattern spectrum
relative to its positive part."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

rng = np.random.default_rng(0)
names = ["Euclidean disks", "rectangles", "digital disks"]
cnt = [[0, 0] for _ in names]
neg = [[] for _ in names]
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
    a = np.array([img.sum()] + [ndi.binary_opening(img, structure=disk(r).astype(bool)).sum() for r in range(1, 16)],
                 dtype=float)
    ps = -np.diff(a)
    cnt[kind][1] += 1
    if (ps < 0).any():
        cnt[kind][0] += 1
        neg[kind].append(-ps[ps < 0].sum() / ps[ps > 0].sum())
for k, nm in enumerate(names):
    m = f"median {np.median(neg[k]):.3f}, max {max(neg[k]):.3f}" if neg[k] else "-"
    print(f"{nm:16s}: non-monotone {cnt[k][0]}/{cnt[k][1]}; negative/positive spectrum mass {m}")
