"""Final figures: error vs radius, shapes, rotation curves (all from all_families / periodic_large)."""
import json, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk
from all_families import load, R
from curves import lower_bound
from families import element_mask

matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
STY = {"gauss": ("0.5", "Gauss disks (not granulometric)"), "octagon": ("C3", "octagons"),
       "adams": ("C1", "Bresenham segments, 8 dir."), "periodic": ("C4", "periodic lines, 8 dir. (LP)"),
       "ours": ("C0", "periodic lines, 13 dir. (LP)"), "dss_milp": ("C2", "DSS chain (MILP)")}


def fig_error():
    fams = load()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    a = ax[0]
    for k, (c, lab) in STY.items():
        info = fams[k][1]
        rho = np.array([info[r][0] for r in range(1, R + 1)]); eps = np.array([info[r][1] for r in range(1, R + 1)])
        if k == "gauss":
            a.plot(rho, eps, ".", color=c, ms=3, label=lab)
        else:
            a.plot(rho, eps, color=c, lw=1.1 if k == "ours" else 0.8, label=lab)
    rr = np.linspace(2, 48, 50)
    a.plot(rr, [lower_bound(x, 1.5) for x in rr], "k--", lw=1, label=r"Theorem 1 ($\Delta=1.5$)")
    a.set_xlabel(r"radius $\rho$"); a.set_ylabel(r"error $\varepsilon$ (pixels)")
    a.set_title(r"(a) radius steps $\leq 2.2$, $\rho\leq48$", fontsize=8)
    a.legend(fontsize=5.5, frameon=False, loc="upper left")
    a = ax[1]
    oc = np.load("octagon_chain.npy"); k = oc[:, 1] >= 5
    a.loglog(oc[k, 1], np.maximum.accumulate(oc[:, 2])[k], color="C3", lw=1, label="octagons")
    pl = json.load(open("periodic_large.json"))
    for key, c, lab in (("sqrt10", "C0", r"periodic, $|v|\leq\sqrt{10}$"), ("5", "C9", r"periodic, $|v|\leq5$"),
                        ("sqrt50", "C5", r"periodic, $|v|\leq\sqrt{50}$")):
        rho = np.array(pl[key]["rho"]); eps = np.maximum.accumulate(np.array(pl[key]["eps"])); k = rho >= 5
        a.loglog(rho[k], eps[k], color=c, lw=1, label=lab)
    rr = np.geomspace(5, 1000, 60)
    a.loglog(rr, [lower_bound(x, 1.0) for x in rr], "k--", lw=1, label=r"Theorem 1, $\Delta=1$")
    a.loglog(rr, [lower_bound(x, 3.0) for x in rr], "k:", lw=1, label=r"Theorem 1, $\Delta=3$")
    a.axhline(0.43, color="C2", lw=0.8, ls="-.", label=r"real relaxation (max, $\rho\leq2000$)")
    a.set_xlabel(r"radius $\rho$"); a.set_ylim(2e-2, 60)
    a.set_title(r"(b) running max of $\varepsilon$, log-log", fontsize=8)
    a.legend(fontsize=5.5, frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "error.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(OUT, "error.png"), bbox_inches="tight", dpi=200)


def fig_shapes(radii=(8, 24, 48)):
    fams = load()
    rows = (("gauss", "Gauss"), ("octagon", "octagons"), ("ours", "periodic, 13 dir."))
    fig, ax = plt.subplots(3, len(radii), figsize=(1.25 * len(radii) + 0.5, 3.9))
    for j, r in enumerate(radii):
        for i, (k, name) in enumerate(rows):
            M = disk(r).astype(bool) if k == "gauss" else element_mask(fams[k][0][r])
            a = ax[i, j]; a.imshow(M, cmap="Greys", interpolation="nearest")
            rho = fams[k][1][r][0]; ys, xs = np.nonzero(M)
            cy, cx = (ys.max() + ys.min()) / 2, (xs.max() + xs.min()) / 2
            t = np.linspace(0, 2 * np.pi, 200); a.plot(cx + (rho + .5) * np.cos(t), cy + (rho + .5) * np.sin(t), color="C3", lw=0.6)
            a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(f"$r={r}$", fontsize=8)
            if j == 0: a.set_ylabel(name, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shapes.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(OUT, "shapes.png"), bbox_inches="tight", dpi=200)


def fig_rotation():
    res = json.load(open("exp_rotation2_squares_mitosis_coins.json"))["squares"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), sharey=True)
    for a, k in zip(ax, ("gauss", "octagon", "ours")):
        F = np.array(res[k]["F"]); r = np.arange(F.shape[1])
        for i in range(F.shape[0]):
            a.plot(r, F[i], color=plt.cm.viridis(i / (F.shape[0] - 1)), lw=0.6)
        a.set_title(f"{STY[k][1].split(' (')[0]}\n$S$={res[k]['S']:.4f}, increases={res[k]['violations']}", fontsize=7)
        a.set_xlabel("radius $r$")
    ax[0].set_ylabel(r"$|X_\theta\circ D_r|/|X_\theta|$")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 45))
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01).set_label(r"$\theta$ (deg)")
    plt.savefig(os.path.join(OUT, "rotation.pdf"), bbox_inches="tight"); plt.savefig(os.path.join(OUT, "rotation.png"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    fig_error(); fig_shapes(); fig_rotation()
