"""Fig. 3: size distributions of the rotated synthetic scene for four families; increases (impossible for a
granulometry) are marked."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import figstyle  # noqa: F401  (fonts, sizes)

OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")

res = json.load(open("exp_rotation2_squares_mitosis_coins_gravel.json"))["squares"]
names = {"gauss": "(a) Gauss disks, single", "gauss_sup": "(b) sup-closure",
         "octagon": "(c) octagons", "ours": "(d) periodic lines, 13 dir."}
fig, ax = plt.subplots(1, 4, figsize=(4.8, 1.55), sharey=True, gridspec_kw=dict(wspace=0.22))
for a, (k, title) in zip(ax, names.items()):
    F = np.array(res[k]["F"])
    r = np.arange(F.shape[1])
    for i in range(F.shape[0]):
        a.plot(r, F[i], color=plt.cm.viridis(i / (F.shape[0] - 1)), lw=0.6)
    inc = np.argwhere(F[:, 1:] > F[:, :-1] + 1e-12)
    if len(inc):
        a.plot(inc[:, 1] + 1, F[inc[:, 0], inc[:, 1] + 1], "o", mfc="none", mec="C3", ms=3.5, mew=0.7,
               label=f"increase ({len(inc)})")
        a.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    a.set_title(title, fontsize=6.5)
    a.set_xlabel("radius $r$")
    a.set_xlim(0, F.shape[1] - 1)
    a.set_xticks([0, 10, 20, 30, 40], ["0", "", "20", "", "40"])
ax[0].set_ylabel(r"$F_\theta(r)$ (area fraction)")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 45))
cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
cb.set_label(r"rotation $\theta$ (deg)")
cb.outline.set_linewidth(0.5)
plt.savefig(os.path.join(OUT, "rotation.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.savefig(os.path.join(OUT, "rotation.png"), bbox_inches="tight", dpi=300)
print("ok", {k: res[k]["violations"] for k in names})
