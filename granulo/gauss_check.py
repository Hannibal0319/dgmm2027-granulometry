import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

def is_open(B, A):
    """B is A-open  <=>  B o A == B  (as sets, B padded)"""
    pad = A.shape[0]
    Bp = np.pad(B, pad).astype(bool)
    return np.array_equal(ndi.binary_opening(Bp, structure=A), Bp)

viol = []
R = 30
for r in range(1, R + 1):
    for s in range(1, r):
        if not is_open(disk(r), disk(s)):
            viol.append((r, s))
print(f"skimage disks: {len(viol)} of {R*(R-1)//2} pairs (s<r<= {R}) violate 'disk(r) is disk(s)-open'")
print(viol[:20])
# consequence: non-monotone size distribution on a test image
rng = np.random.default_rng(0)
img = np.zeros((400, 400), bool)
for _ in range(60):
    y, x, rr = rng.integers(30, 370), rng.integers(30, 370), rng.integers(3, 25)
    yy, xx = np.ogrid[:400, :400]; img |= (yy - y) ** 2 + (xx - x) ** 2 <= rr ** 2
areas = [img.sum()] + [ndi.binary_opening(img, structure=disk(r)).sum() for r in range(1, 26)]
d = -np.diff(areas)
print("pattern spectrum negative entries:", [(r + 1, int(v)) for r, v in enumerate(d) if v < 0])
# does opening by disk(r) of the image stay below opening by disk(s)? (absorption)
bad = 0
for r in range(2, 20):
    for s in range(1, r):
        a = ndi.binary_opening(img, structure=disk(r)); b = ndi.binary_opening(img, structure=disk(s))
        bad += (a & ~b).sum() > 0
print("pairs with gamma_r(X) not subset of gamma_s(X):", bad)
