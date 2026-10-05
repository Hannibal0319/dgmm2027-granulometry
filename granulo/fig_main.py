"""Figures for the granulometry paper."""
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk

from curves import lower_bound
from geom import hull_vertices, roundness

matplotlib.use("Agg")
OUT = os.path.join(os.path.dirname(__file__), "..", "paper_granulo", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})


def gauss_roundness(rs):
    out = []
    for r in rs:
        ys, xs = np.nonzero(disk(int(r)))
        V = hull_vertices(np.stack([xs, ys], 1).astype(float))
        V = V - (V.max(0) + V.min(0)) / 2
        out.append(roundness(V)[0])
    return np.array(out)


def fig_error():
    oc = np.load("octagon_chain.npy")                   # (r, rad, err)
    ev = np.load("chain2_M6_R48_eval.npy")              # (r, rho, eps, rho_gauss, eps_gauss)
    tr = np.load("joint_M32_R1000_track2_s1.npy")       # coarse chain (r, rad, err, ...)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    a = ax[0]
    k = oc[:, 1] <= 49
    a.plot(oc[k, 1], oc[k, 2], color="C3", lw=1, label="octagon chain")
    a.plot(ev[:, 1], ev[:, 2], color="C0", lw=1.2, marker=".", ms=3, label="ours (unit steps)")
    a.plot(ev[:, 0], ev[:, 4], ".", color="0.5", ms=3, label="Gauss disks (not granulometric)")
    rr = np.linspace(2, 48, 50)
    a.plot(rr, [lower_bound(x, 1.0) for x in rr], "k--", lw=1, label=r"Theorem 1 ($\Delta=1$)")
    a.set_xlabel(r"radius $\rho$"); a.set_ylabel(r"error $\varepsilon$ (pixels)")
    a.set_title("(a) unit-step chains, practical radii", fontsize=8)
    a.legend(fontsize=6, frameon=False)
    a = ax[1]
    k = oc[:, 1] >= 5
    a.loglog(oc[k, 1], oc[k, 2], color="C3", lw=1, label="octagon chain")
    u, idx = np.unique(tr[:, 1], return_index=True)
    k = u >= 5
    a.loglog(u[k], tr[idx, 2][k], color="C0", lw=1, marker=".", ms=3, label=r"ours, coarse steps ($\Delta\leq 39$)")
    rr = np.geomspace(5, 1000, 60)
    a.loglog(rr, [lower_bound(x, 1.0) for x in rr], "k--", lw=1, label=r"Theorem 1, $\Delta=1$")
    a.loglog(rr, [max(lower_bound(x, 20.0), 1e-3) for x in rr], "k:", lw=1, label=r"Theorem 1, $\Delta=20$")
    a.axhline(0.34, color="C2", lw=0.8, ls="-.", label="real relaxation (max)")
    a.set_xlabel(r"radius $\rho$"); a.set_ylim(1e-2, 60)
    a.set_title("(b) large radii (log-log)", fontsize=8)
    a.legend(fontsize=6, frameon=False, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "error.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "error.png"), bbox_inches="tight", dpi=200)


def fig_rotation():
    res = json.load(open("exp_rotation.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0), sharey=True)
    names = {"Gauss": "Gauss disks", "octagon": "octagon chain", "ours": "our chain"}
    for a, (k, title) in zip(ax, names.items()):
        F = np.array(res[k]["F"])
        r = np.arange(F.shape[1])
        for i in range(F.shape[0]):
            a.plot(r, F[i], color=plt.cm.viridis(i / (F.shape[0] - 1)), lw=0.6)
        a.set_title(f"{title}\nS={res[k]['S']:.4f}, violations={res[k]['violations']}", fontsize=7)
        a.set_xlabel("radius $r$")
    ax[0].set_ylabel(r"$|X_\theta\circ D_r|/|X_\theta|$")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 45))
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(r"$\theta$ (deg)")
    plt.savefig(os.path.join(OUT, "rotation.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "rotation.png"), bbox_inches="tight", dpi=200)


def fig_violations():
    V = np.load("violations_skimage.npy")
    fig, ax = plt.subplots(figsize=(2.2, 2.2))
    ax.imshow(V[1:, 1:], cmap="Greys", origin="lower", extent=(0.5, V.shape[0] - .5, 0.5, V.shape[0] - .5))
    ax.set_xlabel("$s$"); ax.set_ylabel("$r$")
    ax.set_title(r"$D_r$ not $D_s$-open (black)", fontsize=7)
    plt.savefig(os.path.join(OUT, "violations.pdf"), bbox_inches="tight")




def fig_shapes(chain_file="chain2_M6_R48.npz", radii=(8, 24, 48)):
    from families import element_mask, chain_summands
    from exp_rotation import octagon_chain_counts
    d = np.load(chain_file, allow_pickle=True)
    C, sets = d["C"], list(d["sets"])
    k8, C8 = octagon_chain_counts(max(radii))
    fig, ax = plt.subplots(3, len(radii), figsize=(1.25 * len(radii) + 0.4, 3.9))
    for j, r in enumerate(radii):
        summ = []
        for S, k in zip(sets, C[r - 1]): summ += [list(map(tuple, np.asarray(S, int)))] * int(k)
        ours = element_mask(summ)
        octo = element_mask(chain_summands(k8, C8[r - 1]))
        gau = disk(r).astype(bool)
        for i, (M, name) in enumerate(((gau, "Gauss"), (octo, "octagon"), (ours, "ours"))):
            a = ax[i, j]
            n = max(M.shape)
            a.imshow(M, cmap="Greys", interpolation="nearest")
            t = np.linspace(0, 2 * np.pi, 200)
            ys, xs = np.nonzero(M)
            cy, cx = (ys.max() + ys.min()) / 2, (xs.max() + xs.min()) / 2
            rr = (ys.max() - ys.min() + xs.max() - xs.min()) / 4 + 0.5
            a.plot(cx + rr * np.cos(t), cy + rr * np.sin(t), color="C3", lw=0.6)
            a.set_xticks([]); a.set_yticks([])
            if i == 0: a.set_title(f"$r={r}$", fontsize=8)
            if j == 0: a.set_ylabel(name, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "shapes.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "shapes.png"), bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        globals()[f]()
