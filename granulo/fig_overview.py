"""Fig. 1 (overview): (a) common practice, single openings by digital disks: D_5 o D_4 != D_5;
(b) the classical sup-closure: D_5 o D_4 u D_5 o D_5 = D_5; (c) one element per size: the octagon chain is a
granulometry, but its roundness error grows with the radius, while that of the disks does not; (d) why: short
lattice vectors leave a gap of directions next to the axes (Lemma 1), which no edge of a chain element can fill."""
import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge
from scipy import ndimage as ndi
from skimage.morphology import disk

import figstyle  # noqa: F401  (fonts, sizes)
from all_families import load, R

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
RED, BLUE, GREY = "C3", "C0", (0.6, 0.6, 0.6)
YES, NO = r"$\bf yes$", r"$\bf no$"

fig, ax = plt.subplots(1, 4, figsize=(4.8, 1.7), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25], wspace=0.3))


def pixels(a, D, mask, colour, marker):
    rgb = np.ones(D.shape + (3,))
    rgb[D] = GREY
    rgb[mask] = mcolors.to_rgb(colour)
    ys, xs = np.nonzero(mask)
    a.plot(xs, ys, marker, color="w", ms=2.6, mew=0.7)  # second cue for black-and-white print
    a.imshow(rgb, interpolation="nearest")
    a.set_xticks(np.arange(-.5, D.shape[1], 1), minor=True)
    a.set_yticks(np.arange(-.5, D.shape[0], 1), minor=True)
    a.grid(which="minor", color="w", lw=0.3)
    a.tick_params(which="both", length=0)
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)


VERDICTS = []


def verdict(a, text, y=None):
    VERDICTS.append((a, text))  # placed on a common baseline once the layout is fixed


# (a), (b): D_5 and its opening by D_4
r, s = 5, 4
B = np.pad(disk(r).astype(bool), 1)
O = ndi.binary_opening(np.pad(B, s), structure=disk(s).astype(bool))[s:-s, s:-s]
pixels(ax[0], B, B & ~O, RED, "x")
ax[0].set_title("(a) single openings\n" + r"$D_5\circ D_4\neq D_5$")
verdict(ax[0], f"round: {YES}\ngranulometry: {NO}")

pixels(ax[1], B, B & ~O, BLUE, ".")
ax[1].set_title("(b) sup-closure\n" + r"$D_5\circ D_4\cup D_5\circ D_5=D_5$")
verdict(ax[1], f"round: {YES}\ngranulometry: {YES}")

# (c) one element per size: roundness error against radius, octagon chain vs the disks of (a), (b)
fams = load()
a = ax[2]
for k, lab, col, ls, mk in (("octagon", "octagons", RED, "--", None), ("gauss", "disks", "0.45", "none", ".")):
    info = fams[k][1]
    rho = np.array([info[q][0] for q in range(1, R + 1)])
    eps = np.array([info[q][1] for q in range(1, R + 1)])
    a.plot(rho, eps, color=col, ls=ls, marker=mk, ms=2.0, lw=0.9, label=lab)
a.set_xlim(0, 50)
a.set_ylim(0, 2.0)
a.set_xticks([0, 25, 50])
a.set_yticks([0, 1, 2])
a.tick_params(labelsize=6, length=2, pad=1)
a.set_xlabel(r"radius $\rho$", fontsize=6.5, labelpad=0)
a.set_ylabel(r"error $\varepsilon$", fontsize=6.5, labelpad=1)
a.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.6, borderaxespad=0.1, labelspacing=0.2)
a.set_title("(c) one element per size\noctagon chain")
a.set_box_aspect(1)
verdict(a, f"round: {NO}\ngranulometry: {YES}", y=-0.36)

# (d) direction gap of lattice vectors with norm <= 3
l = 3
a = ax[3]
dirs = set()
for x in range(-l, l + 1):
    for y in range(0, l + 1):
        if (x, y) != (0, 0) and x * x + y * y <= l * l and math.gcd(abs(x), y) == 1 and (y > 0 or x > 0):
            dirs.add((x, y))
for x, y in dirs:
    n = math.hypot(x, y)
    a.plot([0, x / n], [0, y / n], color=BLUE, lw=0.8)
al = math.degrees(math.atan(1 / l))
for th0 in (0, 90 - al, 90, 180 - al):
    a.add_patch(Wedge((0, 0), 1.0, th0, th0 + al, color=RED, alpha=0.3, lw=0))
a.set_xlim(-1.05, 1.05)
a.set_ylim(-0.05, 1.05)
a.set_aspect("equal")
a.axis("off")
a.set_title("(d) why (c) is forced\n" + r"edge directions, $\|v\|\leq3$")
verdict(a, "red: gaps no edge can fill\n" + r"$\Rightarrow\ \varepsilon\gtrsim\rho/(16\pi^2\delta^2)$")

fig.canvas.draw()
rend = fig.canvas.get_renderer()
low = ax[2].xaxis.label.get_window_extent(rend).transformed(fig.transFigure.inverted()).y0
for a, text in VERDICTS:
    box = a.get_position()
    fig.text((box.x0 + box.x1) / 2, low - 0.03, text, ha="center", va="top", fontsize=6.5, linespacing=1.25)
plt.savefig(os.path.join(OUT, "overview.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "overview.png"), bbox_inches="tight", dpi=300)
print("ok")
