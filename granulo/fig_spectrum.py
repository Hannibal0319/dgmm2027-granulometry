"""Fig. 3: (a) an image of disk particles where a larger disk opening keeps pixels that a smaller one removes;
(b) its pattern spectrum by single disk openings (negative values) and by their sup-closure (non-negative);
(c) rotation spread max_theta F - min_theta F of the size distributions of the rotated squares scene.

The image is a random image of the Table 1 type (overlapping disks with integer centres and radii, seed 0): among
the first 100 with at least 6 disks covering at least 15% of the image, the one whose pattern spectrum has the
largest negative part relative to its positive part."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

import figstyle  # noqa: F401  (fonts, sizes)

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
R = 15


def make(rng, n=120):
    img = np.zeros((n, n), bool)
    yy, xx = np.ogrid[:n, :n]
    k = rng.integers(3, 12)
    for _ in range(k):
        y, x = rng.integers(10, n - 10, 2)
        rr = rng.integers(3, 15)
        img |= (yy - y) ** 2 + (xx - x) ** 2 <= rr ** 2
    return img, k


def openings(img):
    return [img] + [ndi.binary_opening(img, structure=disk(r).astype(bool)) for r in range(1, R + 1)]


rng = np.random.default_rng(0)
best = None
for t in range(100):
    img, k = make(rng)
    if k < 6 or img.mean() < 0.15:
        continue
    O = openings(img)
    ps = -np.diff([o.sum() for o in O]).astype(float)
    score = -ps[ps < 0].sum() / ps[ps > 0].sum()
    if best is None or score > best[0]:
        best = (score, img, O)
score, img, O = best
a = np.array([o.sum() for o in O], float)
acc = np.zeros_like(img)
S = [None] * (R + 1)
for r in range(R, -1, -1):
    acc = acc | O[r]
    S[r] = acc.copy()
ps_single = -np.diff(a)
ps_sup = -np.diff(np.array([s.sum() for s in S], float))
assert (ps_sup >= 0).all()
rb = int(np.argmax(np.diff(a))) + 1  # opening by D_rb keeps more than opening by D_(rb-1)
print(f"neg/pos = {score:.3f}; largest increase r={rb - 1}->{rb}: {int(a[rb - 1])} -> {int(a[rb])}")

fig = plt.figure(figsize=(4.6, 1.95))
gs = fig.add_gridspec(2, 2, width_ratios=[0.8, 1.4], wspace=0.35, hspace=1.1)
ax = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])]

# (a) image: light grey X, dark grey X o D_(rb-1), red = in X o D_rb but not in X o D_(rb-1)
p = ax[0]
rgb = np.ones(img.shape + (3,))
rgb[img] = (0.82, 0.82, 0.82)
rgb[O[rb - 1]] = (0.5, 0.5, 0.5)
gain = O[rb] & ~O[rb - 1]
rgb[gain] = (0.85, 0.15, 0.15)
p.imshow(rgb, interpolation="nearest")
ys, xs = np.nonzero(gain)
p.plot(xs, ys, "s", mfc="none", mec="C3", ms=1.6, mew=0.4)  # second cue for black-and-white print
p.set_xticks([])
p.set_yticks([])
for sp in p.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.4)
p.set_title("(a) disk particles $X$")

# (b) pattern spectrum, y clipped so that the negative values are visible
p = ax[1]
r = np.arange(R)
top = max(4 * -ps_single.min(), 60)
p.axhline(0, color="0.3", lw=0.5)
neg = ps_single < 0
p.bar(r[~neg] - 0.2, np.minimum(ps_single[~neg], top), width=0.4, color="0.6", label="single openings")
p.bar(r[neg] - 0.2, ps_single[neg], width=0.4, color="C3", hatch="////", edgecolor="w", linewidth=0,
      label="negative values")
p.bar(r + 0.2, np.minimum(ps_sup, top), width=0.4, color="k", label="sup-closure")
print("clipped bars at r =", list(np.nonzero(np.maximum(ps_single, ps_sup) > top)[0]), "top =", top)
p.set_ylim(1.15 * ps_single.min(), top * 1.45)
p.set_xlabel("radius $r$")
p.set_ylabel("spectrum (pixels)")
p.set_title("(b) pattern spectrum")
p.legend(frameon=False, loc="upper left", handlelength=1.2, labelspacing=0.2, borderaxespad=0.1, fontsize=6)
p.set_xlim(-0.8, R - 0.2)
p.set_xticks([0, 5, 10, 14])

# (c) rotation spread of the size distributions of the rotated squares scene
p = ax[2]
res = json.load(open("exp_rotation2_squares_mitosis_coins_gravel.json"))["squares"]
sty = {"gauss_sup": ("sup-closure", "k", "-"), "ours": ("periodic, 13 dir.", "C0", "-."),
       "octagon": ("octagons", "C3", "--")}
for k, (lab, c, ls) in sty.items():
    F = np.array(res[k]["F"])
    p.plot(np.arange(F.shape[1]), F.max(axis=0) - F.min(axis=0), color=c, ls=ls, lw=0.9, label=lab)
p.set_xlabel("radius $r$")
p.set_ylabel(r"spread of $F_\theta(r)$")
p.set_title("(c) rotation spread")
p.legend(frameon=False, loc="upper left", handlelength=2.0, labelspacing=0.2, borderaxespad=0.1, fontsize=6)
p.set_xlim(0, 40)
p.set_ylim(0, None)

plt.savefig(os.path.join(OUT, "spectrum.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "spectrum.png"), bbox_inches="tight", dpi=300)
print("ok")
