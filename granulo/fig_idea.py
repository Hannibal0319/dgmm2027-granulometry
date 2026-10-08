"""Fig. 1: (a) D_5 is not D_4-open; (b) direction gap of short lattice vectors;
(c) consequence: over the gap one vertex carries the support, so the polygon leaves the circle."""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from scipy import ndimage as ndi
from skimage.morphology import disk

import figstyle  # noqa: F401  (fonts, sizes)

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
RED, BLUE = "C3", "C0"

fig, ax = plt.subplots(1, 3, figsize=(4.8, 1.6), gridspec_kw=dict(width_ratios=[1, 1.3, 1.3], wspace=0.12))

# (a) violation: pixels of D_5 not covered by any translate of D_4 inside D_5
r, s = 5, 4
B = np.pad(disk(r).astype(bool), 1)
O = ndi.binary_opening(np.pad(B, s), structure=disk(s).astype(bool))[s:-s, s:-s]
img = np.ones(B.shape + (3,))
img[B & O] = (0.6, 0.6, 0.6)
img[B & ~O] = (0.85, 0.15, 0.15)
a = ax[0]
a.imshow(img, interpolation="nearest")
ys_, xs_ = np.nonzero(B & ~O)
a.plot(xs_, ys_, "x", color="w", ms=3.0, mew=0.8)  # second cue for black-and-white print
a.set_xticks(np.arange(-.5, B.shape[1], 1), minor=True)
a.set_yticks(np.arange(-.5, B.shape[0], 1), minor=True)
a.grid(which="minor", color="w", lw=0.4)
a.tick_params(which="both", length=0)
a.set_xticks([])
a.set_yticks([])
for sp in a.spines.values():
    sp.set_visible(False)
a.set_title(r"(a) $D_5\circ D_4\neq D_5$")

# (b) direction gap for l = 3 (upper half plane; the picture is symmetric)
l = 3
a = ax[1]
dirs = set()
for x in range(-l, l + 1):
    for y in range(0, l + 1):
        if (x, y) != (0, 0) and x * x + y * y <= l * l and math.gcd(abs(x), y) == 1 and (y > 0 or x > 0):
            dirs.add((x, y))
for x, y in dirs:
    n = math.hypot(x, y)
    a.plot([0, x / n], [0, y / n], color=BLUE, lw=0.9)
alpha = math.degrees(math.atan(1 / l))
for th0 in (0, 90 - alpha, 90, 180 - alpha):
    a.add_patch(Wedge((0, 0), 1.0, th0, th0 + alpha, color=RED, alpha=0.25, lw=0))
a.text(0.0, -0.08, r"no direction within $\arctan(1/\ell)$ of an axis", color=RED, ha="center", va="top",
       fontsize=6.5)
a.set_xlim(-1.08, 1.08)
a.set_ylim(-0.3, 1.05)
a.set_aspect("equal")
a.axis("off")
a.set_title(r"(b) lattice directions, $\|v\|\leq\ell=3$")

# (c) schematic: two consecutive edge normals separated by the gap alpha (exaggerated)
a = ax[2]
al = 0.75
c0 = math.pi / 2 - al / 2
t = np.linspace(0.25, math.pi - 0.25, 300)
a.plot(np.cos(t), np.sin(t), color="k", lw=1)


def tang(n, s0, s1):
    p = np.array([math.cos(n), math.sin(n)])
    d = np.array([-math.sin(n), math.cos(n)])
    return np.array([p + s0 * d, p + s1 * d])


q = np.array([0.0, 1.0]) / math.cos(al / 2)
e1 = tang(c0, -0.55, math.tan(al / 2))
e2 = tang(c0 + al, -math.tan(al / 2), 0.55)
a.plot(e1[:, 0], e1[:, 1], color=BLUE, lw=1.3)
a.plot(e2[:, 0], e2[:, 1], color=BLUE, lw=1.3)
a.plot(*q, "o", color=RED, ms=3.5)
for n in (c0, c0 + al):
    p = np.array([math.cos(n), math.sin(n)])
    a.annotate("", xy=p, xytext=p * 0.55, arrowprops=dict(arrowstyle="->", lw=0.5, color="0.4"))
a.add_patch(Wedge((0, 0), 0.4, math.degrees(c0), math.degrees(c0 + al), color=RED, alpha=0.25, lw=0))
a.text(0.0, 0.2, r"gap $\geq\alpha$", ha="center", color=RED, fontsize=6.5)
a.text(0.1, 1.17, "vertex $q$ outside", color=RED, va="bottom", fontsize=6.5)
a.text(-0.55, 0.42, "circle", ha="center", fontsize=6.5)
a.text(-0.78, 1.02, "polygon", color=BLUE, ha="center", fontsize=6.5)
a.set_xlim(-1.1, 1.1)
a.set_ylim(-0.05, 1.32)
a.set_aspect("equal")
a.axis("off")
a.set_title(r"(c) gap $\Rightarrow$ not round")

plt.savefig(os.path.join(OUT, "idea.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "idea.png"), bbox_inches="tight", dpi=300)
print("ok")
