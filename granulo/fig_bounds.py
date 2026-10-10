"""Fig. 2: (a) roundness error of practical families; (b) the chain of Theorem 2 with m = 2, normalised as
eps * m^2 / rho, between its exact upper bound (Theorem 2) and lower bound (Corollary 1); (c) the multiscale chain,
normalised as eps / sqrt(rho).

Drawn at the final LNCS size (text width 12.2 cm = 4.8 in, included unscaled), lettering >= 6 pt, direct labels,
line styles and markers so that the curves stay distinguishable in black and white."""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle  # noqa: F401  (fonts, sizes)
from all_families import load, R
from figstyle import FAMILY

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")

fig, ax = plt.subplots(1, 3, figsize=(4.8, 1.6), gridspec_kw=dict(width_ratios=[1, 1, 1], wspace=1.0))

# (a) practical families: the disks for reference, octagons, Bresenham, and the best chain (Table 2 has all)
a = ax[0]
fams = load()
labels = {"gauss": "disks", "octagon": "octagons", "adams": "Bresenham", "ours": "13-dir. lines"}
for k in ("adams", "octagon", "ours", "gauss"):
    _, c, ls, mk = FAMILY[k]
    info = fams[k][1]
    rho = np.array([info[r][0] for r in range(1, R + 1)])
    eps = np.array([info[r][1] for r in range(1, R + 1)])
    a.plot(rho, eps, color=c, ls=ls, marker=mk, ms=2.2, lw=1.0 if k == "ours" else 0.8)
    y = {"gauss": 0.2, "ours": 0.75, "octagon": 1.85, "adams": 3.2}[k]
    a.text(51, y, labels[k], color="k", fontsize=6, va="center", clip_on=False)
a.set_xlim(0, 50)
a.set_ylim(0, 3.5)
a.set_xlabel(r"radius $\rho$")
a.set_ylabel(r"error $\varepsilon$ (pixels)")
a.set_title("(a) practical families")

# (b) the Theorem 2 chain with m = 2, normalised as eps * m^2 / rho, between its exact bounds:
# upper (Thm. 2) 1/8 + 2 m^4 / rho, lower (Cor. 1, delta = sqrt(2) m / pi) m^2 c / (2 + c)
a = ax[1]
d = json.load(open("upper_fixed_m.json"))
m = 2
rho = np.array(d[str(m)]["rho"])
eps = np.array(d[str(m)]["eps"])
k = rho >= 5
rr = np.geomspace(5, 3000, 200)
delta = math.sqrt(2) * m / math.pi
c = 1 - math.cos(math.atan(1 / (math.pi * delta)) / 2)
lo = m * m * c / (2 + c)
up = 1 / 8 + 2 * m ** 4 / rr
a.fill_between(rr, lo, up, color="0.9", lw=0)
a.semilogx(rr, up, color="0.3", lw=0.6, ls="--")
a.axhline(lo, color="0.3", lw=0.6, ls=":")
a.semilogx(rho[k], eps[k] * m * m / rho[k], color="C1", lw=0.8)
a.text(3000, 0.205, "upper bound (Thm. 2)", fontsize=6, ha="right", va="bottom")
a.text(3000, lo - 0.004, "lower bound (Cor. 1)", fontsize=6, ha="right", va="top")
a.text(3000, 0.07, "chain, $m=2$", color="k", fontsize=6, ha="right", va="bottom")
a.set_xlim(5, 3000)
a.set_ylim(0, 0.3)
a.set_yticks([0, 1 / 18, 1 / 8, 0.25], ["0", "1/18", "1/8", "1/4"])
a.set_xlabel(r"radius $\rho$")
a.set_ylabel(r"$\varepsilon\,m^2/\rho$")
a.set_title("(b) Theorem 2 chain")
norm = (eps[k] * m * m / rho[k])[rho[k] > 1000]
print("m=2 eps m^2/rho for rho>1000: %.4f-%.4f, lower %.4f" % (norm.min(), norm.max(), lo))

# (c) multiscale chain, eps / sqrt(rho)
a = ax[2]
mu = json.load(open("upper_multi.json"))
rho = np.array(mu["rho"])
eps = np.array(mu["eps"])
k = rho >= 20
a.semilogx(rho[k], eps[k] / np.sqrt(rho[k]), color="k", lw=0.7)
a.axhline(0.28, color="C3", lw=0.6, ls="--")
a.text(25, 0.29, r"$0.28$", color="k", fontsize=6, va="bottom")
a.set_xlim(20, 5e4)
a.set_ylim(0, 0.35)
a.set_xlabel(r"radius $\rho$")
a.set_ylabel(r"$\varepsilon/\sqrt{\rho}$")
a.set_title("(c) multiscale chain")
print("multiscale max eps/sqrt(rho) for rho>=20: %.3f" % (eps[k] / np.sqrt(rho[k])).max())

plt.savefig(os.path.join(OUT, "error.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "error.png"), bbox_inches="tight", dpi=300)
print("ok")
