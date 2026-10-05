"""Fig. 1: provenance of the h-dome on a 1-D signal and on a 2-D image.

The vector-Jacobian product of sum(D_h(u)) is +1 on dome pixels and a negative
mass concentrated on the peak (dynamic >= h) or on the saddle (dynamic < h)."""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geodrec import reconstruct_with_provenance  # noqa

OUT = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

# ---------------------------------------------------------------- 1-D signal (as 1 x n image)
x = np.linspace(0, 1, 400)
u = (1.0 * np.exp(-((x - 0.2) / 0.07) ** 2) + 0.45 * np.exp(-((x - 0.37) / 0.045) ** 2)
     + 0.7 * np.exp(-((x - 0.72) / 0.08) ** 2))
h = 0.35
ut = torch.tensor(u, dtype=torch.float64)[None, None, None]
R, src, _ = reconstruct_with_provenance(ut - h, ut, connectivity=1, cuda_graph=False)
R, src = R[0, 0, 0].numpy(), src[0, 0, 0].numpy()
n = len(u)
D = u - R
dome = D > 1e-9
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw=dict(width_ratios=[1.6, 1]))
a = ax[0]
a.plot(x, u, "k-", lw=1, label="$u$")
a.plot(x, u - h, color="0.6", lw=0.8, ls="--", label="$u-h$ (marker)")
a.fill_between(x, R, u, where=dome, color="C0", alpha=0.35, label="$D_h(u)=u-R_u(u-h)$")
a.plot(x, R, "C0", lw=1, label="$R_u(u-h)$")
for s in np.unique(src[dome]):
    q, from_marker = (s, True) if s < n else (s - n, False)
    a.plot(x[q], u[q], "v" if from_marker else "^", ms=6,
           color="C3" if from_marker else "C2", zorder=5)
a.plot([], [], "v", color="C3", label="source in marker (peak)")
a.plot([], [], "^", color="C2", label="source in mask (saddle)")
a.set_yticks([])
a.set_xticks([])
a.legend(fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
a.set_title("(a) provenance of the $h$-dome, 1-D", fontsize=8)

# ---------------------------------------------------------------- 2-D: VJP of sum(D)
yy, xx = np.mgrid[0:96, 0:96] / 96.0
g2 = (np.exp(-((xx - .3) ** 2 + (yy - .3) ** 2) / .012) + .55 * np.exp(-((xx - .47) ** 2 + (yy - .45) ** 2) / .006)
      + .7 * np.exp(-((xx - .72) ** 2 + (yy - .72) ** 2) / .012))
h2 = 0.3
u2 = torch.tensor(g2, dtype=torch.float64)[None, None].requires_grad_()
from geodrec import hdome  # noqa: E402
hdome(u2, h2).sum().backward()
G = u2.grad[0, 0].numpy()
_, src2, _ = reconstruct_with_provenance(u2.detach() - h2, u2.detach(), cuda_graph=False)
src2 = src2[0, 0].numpy()
a = ax[1]
a.imshow(np.where(G > 0, 1.0, np.nan), cmap="Blues", vmin=0, vmax=2.5)
a.contour(g2, levels=16, colors="0.6", linewidths=0.4)
n2 = g2.size
dome2 = (G > 0)
for sv in np.unique(src2[dome2] if dome2.any() else []):
    q, from_marker = (sv, True) if sv < n2 else (sv - n2, False)
    yv, xv = divmod(int(q), g2.shape[1])
    a.plot(xv, yv, "v" if from_marker else "^", ms=6, color="C3" if from_marker else "C2")
    a.annotate(f"{G[yv, xv]:.0f}", (xv, yv), color="k", fontsize=6, xytext=(4, -8),
               textcoords="offset points")
a.set_xticks([])
a.set_yticks([])
a.set_title(r"(b) $\nabla_u \sum_p D_h(u)(p)$, 2-D", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "illustration.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "illustration.png"), bbox_inches="tight", dpi=200)
print("saved")
