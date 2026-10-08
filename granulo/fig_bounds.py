"""Fig. 2: (a) practical families; (b) Theorem 2 chains between the upper bound (Thm. 2) and the lower
bound (Cor. 1), and the multiscale chain vs sqrt(rho).

Drawn at the final LNCS size (text width 12.2 cm = 4.8 in, included unscaled), lettering >= 6 pt,
line styles and markers so that the curves stay distinguishable in black and white."""
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from all_families import load, R

matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "font.family": "serif", "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                     "axes.linewidth": 0.6, "lines.linewidth": 0.9})
# colour, line style, marker, label
STY = {"gauss": ("0.45", "none", ".", "Gauss disks (not gran.)"),
       "octagon": ("C3", "--", None, "octagons"),
       "adams": ("C1", ":", None, "Bresenham, 8 dir."),
       "periodic": ("C4", "-.", None, "periodic lines, 8 dir."),
       "ours": ("C0", "-", None, "periodic lines, 13 dir."),
       "dss_milp": ("C2", "-", "x", "DSS chain (MILP)")}


def lb_mean_step(rho, delta):
    """Cor. 1 (chain from a point): eps >= rho c / (2 + c), c = 1 - cos(arctan(1/(pi delta))/2)."""
    c = 1 - math.cos(math.atan(1 / (math.pi * delta)) / 2)
    return rho * c / (2 + c)


fig, ax = plt.subplots(1, 2, figsize=(5.15, 2.25), gridspec_kw=dict(wspace=0.3))
a = ax[0]
fams = load()
for k, (c, ls, mk, lab) in STY.items():
    info = fams[k][1]
    rho = np.array([info[r][0] for r in range(1, R + 1)])
    eps = np.array([info[r][1] for r in range(1, R + 1)])
    a.plot(rho, eps, color=c, ls=ls, marker=mk, ms=2.5 if mk == "." else 2.2, markevery=1 if mk == "." else 6,
           lw=1.2 if k == "ours" else 0.8, label=lab)
a.set_xlabel(r"radius $\rho$")
a.set_ylabel(r"error $\varepsilon$ (pixels)")
a.set_title(r"(a) families, $\rho\leq48$")
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=2.2, columnspacing=0.8)

a = ax[1]
d = json.load(open("upper_fixed_m.json"))
marks = {"2": ("C1", "o"), "4": ("C2", "s"), "8": ("C0", "^")}
rr = np.geomspace(5, 3000, 80)
for m, (c, mk) in marks.items():
    rho = np.array(d[m]["rho"])
    eps = np.array(d[m]["eps"])
    k = rho >= 5
    a.loglog(rho[k], eps[k], color=c, lw=0.8, marker=mk, ms=2.2, markevery=max(1, k.sum() // 8),
             label=f"chain, $m={m}$")
    mm = int(m)
    delta = math.sqrt(2) * mm / math.pi
    a.loglog(rr, rr / (8 * mm * mm) + 2 * mm * mm, color=c, lw=0.6, ls="--")
    a.loglog(rr, [lb_mean_step(x, delta) for x in rr], color=c, lw=0.6, ls=":")
mu = json.load(open("upper_multi.json"))
rho = np.array(mu["rho"])
eps = np.array(mu["eps"])
k = rho >= 5
a.loglog(rho[k], eps[k], color="k", lw=1.0, label=r"multiscale, $m\sim\rho^{1/4}$")
a.loglog(rr, 0.3 * np.sqrt(rr), color="k", lw=0.6, ls="-.", label=r"$0.3\sqrt{\rho}$")
a.plot([], [], color="0.3", ls="--", lw=0.6, label="upper bound")
a.plot([], [], color="0.3", ls=":", lw=0.6, label="lower bound")
a.set_ylim(1e-2, 300)
a.set_xlim(5, 3000)
a.set_xlabel(r"radius $\rho$")
a.set_title("(b) chains of Theorem 2")
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=2.2, columnspacing=0.8)
plt.savefig(os.path.join(OUT, "error.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "error.png"), bbox_inches="tight", dpi=300)
w, h = fig.get_size_inches()
print("ok", w, h)
