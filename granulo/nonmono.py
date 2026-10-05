import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk
rng = np.random.default_rng(0)
found = 0
for trial in range(400):
    n = 120
    img = np.zeros((n, n), bool)
    kind = trial % 3
    for _ in range(rng.integers(3, 12)):
        y, x = rng.integers(10, n - 10, 2)
        if kind == 0:
            rr = rng.integers(3, 15); yy, xx = np.ogrid[:n, :n]; img |= (yy-y)**2 + (xx-x)**2 <= rr**2
        elif kind == 1:
            h, w = rng.integers(3, 25, 2); img[y:y+h, x:x+w] = True
        else:
            # random digital disk from the library itself (worst case: SE-shaped grains)
            rr = int(rng.integers(2, 14)); D = disk(rr); yy0, xx0 = max(0, y-rr), max(0, x-rr)
            sub = img[yy0:yy0+D.shape[0], xx0:xx0+D.shape[1]]; sub |= D[:sub.shape[0], :sub.shape[1]].astype(bool)
    areas = [int(ndi.binary_opening(img, structure=disk(r)).sum()) for r in range(1, 16)]
    inc = [(r + 2, areas[r + 1] - areas[r]) for r in range(len(areas) - 1) if areas[r + 1] > areas[r]]
    if inc:
        found += 1
        if found <= 3: print("trial", trial, "kind", kind, "area increases at r:", inc)
print("images with non-monotone size distribution:", found, "/ 400")
