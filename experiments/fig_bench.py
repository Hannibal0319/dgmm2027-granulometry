"""Fig. 2: memory / time vs. image size, and gradient error of truncated unrolling."""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

eff = pd.read_csv(os.path.join(ROOT, "results", "bench_efficiency.csv"))
fid = pd.read_csv(os.path.join(ROOT, "results", "bench_fidelity.csv"))
eff["total"] = eff.fwd + eff.bwd
style = {"provenance": ("C0", "o", "ours (CUDA graph)"), "provenance_eager": ("C0", "s", "ours (eager)"),
         "unrolled": ("C3", "^", "unrolled autograd")}
ls = {"smooth": "-", "serpentine": "--"}

fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.1))
for kind in ("smooth", "serpentine"):
    for m in ("provenance", "provenance_eager", "unrolled"):
        d = eff[(eff.kind == kind) & (eff.method == m) & (eff.status == "ok")]
        c, mk, lab = style[m]
        mfc = "none" if m == "provenance_eager" else c
        if m != "provenance_eager":
            ax[0].loglog(d.n ** 2, d.peak_MB, ls[kind], color=c, marker=mk, ms=3)
        ax[1].loglog(d.n ** 2, d.total, ls[kind], color=c, marker=mk, ms=3, mfc=mfc)
        oom = eff[(eff.kind == kind) & (eff.method == m) & (eff.status != "ok")]
        for a, y in ((ax[0], 6e4), (ax[1], 150)):
            a.plot(oom.n ** 2, [y] * len(oom), "x", color=c, ms=5, mew=1.2)
ax[0].axhline(8188, color="0.5", lw=0.6, ls=":")
ax[0].text(3e5, 8188 * 0.45, "8 GB", fontsize=6, color="0.4")
ax[0].set_xlabel("pixels per image (batch 4)")
ax[0].set_ylabel("peak memory (MB)")
ax[0].set_title("(a) memory, forward+backward", fontsize=8)
ax[1].set_xlabel("pixels per image (batch 4)")
ax[1].set_ylabel("time (s)")
ax[1].set_title("(b) time, forward+backward", fontsize=8)
from matplotlib.lines import Line2D  # noqa: E402
hs = [Line2D([], [], color=style[m][0], marker=style[m][1], ms=3,
             mfc="none" if m == "provenance_eager" else style[m][0], label=style[m][2])
      for m in ("provenance", "provenance_eager", "unrolled")]
hs += [Line2D([], [], color="k", ls="-", label="smooth input"),
       Line2D([], [], color="k", ls="--", label="serpentine input"),
       Line2D([], [], color="C3", marker="x", ls="none", label="out of memory")]
fig.legend(handles=hs, loc="lower center", ncol=6, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.07))

for h, c in ((0.05, "C2"), (0.2, "C4")):
    d = fid[fid.h == h].groupby("K").rel_err.agg(["median", lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
    ax[2].semilogx(d.index, d["median"], "-o", ms=3, color=c, label=f"$h={h}$")
    ax[2].fill_between(d.index, d.iloc[:, 1], d.iloc[:, 2], color=c, alpha=0.2)
    kstar = fid[fid.h == h].groupby("img").n_iter_conv.first().median()
    ax[2].axvline(kstar, color=c, lw=0.6, ls=":")
ax[2].set_xlabel("unrolled iterations $K$")
ax[2].set_ylabel(r"$\|\nabla_K-\nabla\|/\|\nabla\|$")
ax[2].set_title("(c) gradient error of truncation", fontsize=8)
ax[2].legend(fontsize=6, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1))
plt.savefig(os.path.join(OUT, "bench.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(OUT, "bench.png"), bbox_inches="tight", dpi=200)
print(eff.to_string())
print(fid.groupby(["h", "K"]).agg(rel_err=("rel_err", "median"), cos=("cos", "median"),
                                  kstar=("n_iter_conv", "median")).to_string())
