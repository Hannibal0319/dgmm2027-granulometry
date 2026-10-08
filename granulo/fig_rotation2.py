"""Fig. 3: size distributions of the rotated synthetic scene for four families."""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7, "font.family": "serif", "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6})

res = json.load(open("exp_rotation2_squares_mitosis_coins_gravel.json"))["squares"]
names = {"gauss": "Gauss disks, single", "gauss_sup": "sup-closure",
         "octagon": "octagons", "ours": "periodic, 13 dir."}
fig, ax = plt.subplots(1, 4, figsize=(5.05, 1.75), sharey=True)
for a, (k, title) in zip(ax, names.items()):
    F = np.array(res[k]["F"])
    r = np.arange(F.shape[1])
    for i in range(F.shape[0]):
        a.plot(r, F[i], color=plt.cm.viridis(i / (F.shape[0] - 1)), lw=0.6)
    a.set_title(f"{title}\n$S$={1000 * res[k]['S']:.1f}e-3, incr.={res[k]['violations']}", fontsize=6.5)
    a.set_xlabel("radius $r$", fontsize=7)
ax[0].set_ylabel(r"$|\gamma_r(X_\theta)|/|X_\theta|$")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 45))
fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01).set_label(r"$\theta$ (deg)")
plt.savefig(os.path.join(OUT, "rotation.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "rotation.png"), bbox_inches="tight", dpi=300)
print("ok")
