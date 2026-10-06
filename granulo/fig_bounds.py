"""Fig. 1(b): Theorem 2 chains for fixed m between the lower bound (Cor. 1) and the upper bound
(Thm. 2); multiscale chain vs sqrt(rho)."""
import json, math, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from all_families import load, R
from curves import lower_bound

matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
STY = {"gauss": ("0.5", "Gauss disks (not granulometric)"), "octagon": ("C3", "octagons"),
       "adams": ("C1", "Bresenham segments, 8 dir."), "periodic": ("C4", "periodic lines, 8 dir."),
       "ours": ("C0", "periodic lines, 13 dir."), "dss_milp": ("C2", "DSS chain (MILP)")}


def lb_mean_step(rho, delta):
    """Cor. 1 with D = pi delta: smallest eps with 8 eps >= 2 (rho - 2 eps)(1 - cos(alpha/2))."""
    a = math.atan(1 / (math.pi * delta)); c = 1 - math.cos(a / 2)
    return rho * c / (2 + c)   # chain from a point: eps_K >= rho c / (2 + c)


fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
a = ax[0]
fams = load()
for k, (c, lab) in STY.items():
    info = fams[k][1]
    rho = np.array([info[r][0] for r in range(1, R + 1)]); eps = np.array([info[r][1] for r in range(1, R + 1)])
    if k == "gauss":
        a.plot(rho, eps, ".", color=c, ms=3, label=lab)
    else:
        a.plot(rho, eps, color=c, lw=1.1 if k == "ours" else 0.8, label=lab)
a.set_xlabel(r"radius $\rho$"); a.set_ylabel(r"error $\varepsilon$ (pixels)")
a.set_title(r"(a) practical families, $\rho\leq48$", fontsize=8)
a.legend(fontsize=5.5, frameon=False, loc="upper left")

a = ax[1]
d = json.load(open("upper_fixed_m.json"))
cols = {"2": "C1", "4": "C2", "8": "C0"}
rr = np.geomspace(5, 3000, 80)
for m, c in cols.items():
    rho = np.array(d[m]["rho"]); eps = np.array(d[m]["eps"]); k = rho >= 5
    a.loglog(rho[k], eps[k], color=c, lw=0.9, label=f"Thm. 2 chain, $m={m}$")
    mm = int(m); delta = math.sqrt(2) * mm / math.pi
    a.loglog(rr, rr / (8 * mm * mm) + 2 * mm * mm, color=c, lw=0.6, ls="--")
    a.loglog(rr, [lb_mean_step(x, delta) for x in rr], color=c, lw=0.6, ls=":")
mu = json.load(open("upper_multi.json"))
rho = np.array(mu["rho"]); eps = np.array(mu["eps"]); k = rho >= 5
a.loglog(rho[k], eps[k], color="k", lw=1.0, label=r"multiscale, $m\sim\rho^{1/4}$")
a.loglog(rr, 0.3 * np.sqrt(rr), color="k", lw=0.6, ls="-.", label=r"$0.3\sqrt{\rho}$")
a.set_ylim(1e-2, 300); a.set_xlim(5, 3000)
a.set_xlabel(r"radius $\rho$")
a.set_title("(b) Thm. 2 chains, upper (dashed) and lower (dotted) bounds", fontsize=8)
a.legend(fontsize=5.5, frameon=False, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "error.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "error.png"), bbox_inches="tight", dpi=200)
print("ok")
