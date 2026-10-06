"""Fig. 3: size distributions of the rotated synthetic scene for four families."""
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

res = json.load(open("exp_rotation2_squares_mitosis_coins_gravel.json"))["squares"]
names = {"gauss": "Gauss disks (single)", "gauss_sup": "sup-closure of Gauss disks",
         "octagon": "octagons", "ours": "periodic lines, 13 dir."}
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.9), sharey=True)
for a, (k, title) in zip(ax, names.items()):
    F = np.array(res[k]["F"])
    r = np.arange(F.shape[1])
    for i in range(F.shape[0]):
        a.plot(r, F[i], color=plt.cm.viridis(i / (F.shape[0] - 1)), lw=0.6)
    a.set_title(f"{title}\n$S$={1000 * res[k]['S']:.1f}e-3, increases={res[k]['violations']}", fontsize=6.5)
    a.set_xlabel("radius $r$", fontsize=7)
ax[0].set_ylabel(r"$|\gamma_r(X_\theta)|/|X_\theta|$")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 45))
fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01).set_label(r"$\theta$ (deg)")
plt.savefig(os.path.join(OUT, "rotation.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "rotation.png"), bbox_inches="tight", dpi=200)
print("ok")
