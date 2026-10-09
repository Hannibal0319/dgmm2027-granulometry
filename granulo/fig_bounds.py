"""Fig. 2: (a) practical families; (b) fixed-m chains of Theorem 2 between the upper bound (Thm. 2) and the
lower bound (Cor. 1); (c) the multiscale chain against 0.3 sqrt(rho).

Drawn at the final LNCS size (text width 12.2 cm = 4.8 in, included unscaled), lettering >= 6 pt,
line styles and markers so that the curves stay distinguishable in black and white."""
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

from all_families import load, R
from figstyle import FAMILY

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")


def lb_mean_step(rho, delta):
    """Cor. 1 (chain from a point): eps >= rho c / (2 + c), c = 1 - cos(arctan(1/(pi delta))/2)."""
    c = 1 - math.cos(math.atan(1 / (math.pi * delta)) / 2)
    return rho * c / (2 + c)


fig, ax = plt.subplots(1, 3, figsize=(4.8, 1.75), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.42))

# (a) practical families
a = ax[0]
fams = load()
for k in ("gauss", "octagon", "adams", "periodic", "ours", "dss_milp"):
    lab, c, ls, mk = FAMILY[k]
    info = fams[k][1]
    rho = np.array([info[r][0] for r in range(1, R + 1)])
    eps = np.array([info[r][1] for r in range(1, R + 1)])
    a.plot(rho, eps, color=c, ls=ls, marker=mk, ms=2.5 if mk == "." else 2.2, markevery=1 if mk == "." else 6,
           lw=1.1 if k == "ours" else 0.8, label=lab)
a.set_xlabel(r"radius $\rho$")
a.set_ylabel(r"error $\varepsilon$ (pixels)")
a.set_title(r"(a) practical families")
a.set_xlim(0, 50)
a.set_ylim(0, 5.0)
a.legend(frameon=False, loc="upper left", handlelength=2.0, labelspacing=0.2, borderaxespad=0.1, fontsize=6)

# (b) fixed-m chains of Theorem 2 between their bounds
a = ax[1]
d = json.load(open("upper_fixed_m.json"))
marks = {"2": ("C1", "o"), "8": ("C0", "^")}
rr = np.geomspace(5, 3000, 80)
for m, (c, mk) in marks.items():
    rho = np.array(d[m]["rho"])
    eps = np.array(d[m]["eps"])
    k = rho >= 5
    a.loglog(rho[k], eps[k], color=c, lw=0.9, marker=mk, ms=2.2, markevery=max(1, k.sum() // 8),
             label=f"chain, $m={m}$")
    mm = int(m)
    delta = math.sqrt(2) * mm / math.pi
    a.loglog(rr, rr / (8 * mm * mm) + 2 * mm * mm, color=c, lw=0.6, ls="--")
    a.loglog(rr, [lb_mean_step(x, delta) for x in rr], color=c, lw=0.6, ls=":")
a.plot([], [], color="0.3", ls="--", lw=0.6, label="upper (Thm. 2)")
a.plot([], [], color="0.3", ls=":", lw=0.6, label="lower (Cor. 1)")
a.set_ylim(1e-2, 300)
a.set_xlim(5, 3000)
a.set_xlabel(r"radius $\rho$")
a.set_title("(b) chains with fixed $m$")
a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, handlelength=1.8,
         labelspacing=0.25, columnspacing=0.8, fontsize=6)

# (c) multiscale chain
a = ax[2]
mu = json.load(open("upper_multi.json"))
rho = np.array(mu["rho"])
eps = np.array(mu["eps"])
k = rho >= 5
a.loglog(rho[k], eps[k], color="k", lw=0.9, label=r"multiscale, $m\sim\rho^{1/4}$")
a.loglog(rr, 0.3 * np.sqrt(rr), color="C3", lw=0.8, ls="--", label=r"$0.3\sqrt{\rho}$")
a.set_ylim(1e-1, 100)
a.set_xlim(5, 3000)
a.set_xlabel(r"radius $\rho$")
a.set_title("(c) multiscale chain")
a.legend(frameon=False, loc="upper left", handlelength=2.0, labelspacing=0.2, borderaxespad=0.1, fontsize=6)

plt.savefig(os.path.join(OUT, "error.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "error.png"), bbox_inches="tight", dpi=300)
print("ok")
