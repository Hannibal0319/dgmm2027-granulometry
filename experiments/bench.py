"""E1: gradient fidelity of truncated unrolling.  E2: time / peak memory vs. size.

Writes results/bench_fidelity.csv and results/bench_efficiency.csv."""
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geodrec import reconstruct_with_provenance, reconstruction_by_dilation, reconstruction_unrolled  # noqa
from experiments.data import load_split  # noqa

DEV = "cuda"
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)


def hdome_with(rec, u, h, **kw):
    return u - rec(u - h, u, **kw)


def fidelity():
    """Relative gradient error of K-step unrolled backprop w.r.t. the exact gradient,
    for the h-dome of real (smoothed) nuclei images."""
    imgs, _ = load_split("validation")
    rows = []
    torch.manual_seed(0)
    for i, im in enumerate(imgs[:20]):
        crop = ndi.gaussian_filter(im, 2.0)[100:356, 200:456]  # 256 x 256
        u0 = torch.tensor(crop, dtype=torch.float32, device=DEV)[None, None]
        u0 = u0 + 1e-5 * torch.rand_like(u0)  # break exact ties of quantised data
        w = torch.randn_like(u0)
        for h in (0.05, 0.2):
            u = u0.clone().requires_grad_()
            (hdome_with(reconstruction_by_dilation, u, h) * w).sum().backward()
            g_ref = u.grad.clone()
            _, _, n_it = reconstruct_with_provenance(u0 - h, u0, check_every=1)
            for K in (1, 2, 5, 10, 20, 50, 100, 200, 400):
                u = u0.clone().requires_grad_()
                (hdome_with(reconstruction_unrolled, u, h, n_iter=K) * w).sum().backward()
                err = ((u.grad - g_ref).norm() / g_ref.norm()).item()
                cos = torch.nn.functional.cosine_similarity(u.grad.flatten(), g_ref.flatten(), 0).item()
                rows.append(dict(img=i, h=h, K=K, n_iter_conv=n_it, rel_err=err, cos=cos))
        print("fidelity", i, n_it, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bench_fidelity.csv"), index=False)


def serpentine_mask(n, width=2):
    """Boustrophedon corridor of the given width: horizontal lanes separated by walls,
    joined alternately at the right / left ends; geodesic length ~ n^2 / (2 width)."""
    m = np.zeros((n, n))
    lanes = list(range(0, n - width + 1, 2 * width))
    for k, r in enumerate(lanes):
        m[r:r + width, :] = 1
        if k + 1 < len(lanes):
            c = slice(n - width, n) if k % 2 == 0 else slice(0, width)
            m[r:lanes[k + 1] + width, c] = 1
    return m


def make_input(kind, n, B):
    rng = np.random.default_rng(n)
    if kind == "smooth":
        g = np.stack([ndi.gaussian_filter(rng.random((n, n)), 4.0) for _ in range(B)])
        g = (g - g.min()) / (g.max() - g.min())
        f = g - 0.05
    else:  # serpentine: marker = one seed at the corridor's start
        sp = serpentine_mask(n)
        g = np.stack([sp * (0.5 + 0.5 * rng.random((n, n))) for _ in range(B)])
        f = np.zeros_like(g)
        f[:, 0, 0] = g[:, 0, 0]
    g = torch.tensor(g, dtype=torch.float32, device=DEV)[:, None]
    f = torch.tensor(f, dtype=torch.float32, device=DEV)[:, None]
    return f, g


def run_once(method, f, g):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fr, gr = f.clone().requires_grad_(), g.clone().requires_grad_()
    t0 = time.perf_counter()
    if method.startswith("provenance"):
        R = reconstruction_by_dilation(fr, gr, cuda_graph=(method == "provenance"))
    else:
        R = reconstruction_unrolled(fr, gr)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    R.sum().backward()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() - base
    return t1 - t0, t2 - t1, peak


def efficiency():
    torch.cuda.set_per_process_memory_fraction(1.0)  # no silent spill to host memory
    rows = []
    for kind in ("smooth", "serpentine"):
        sizes = (64, 128, 256, 512, 1024, 2048) if kind == "smooth" else (32, 64, 128, 256, 512)
        for n in sizes:
            B = 4
            f, g = make_input(kind, n, B)
            _, _, it = reconstruct_with_provenance(f, g, check_every=1)
            for method in ("provenance", "provenance_eager", "unrolled"):
                try:
                    run_once(method, f, g)  # warm-up
                    ts = [run_once(method, f, g) for _ in range(3)]
                    fwd, bwd, peak = np.median(np.array(ts), axis=0)
                    status = "ok"
                except torch.cuda.OutOfMemoryError:
                    fwd = bwd = peak = np.nan
                    status = "OOM"
                torch.cuda.empty_cache()
                rows.append(dict(kind=kind, n=n, B=B, method=method, iters=it, fwd=fwd, bwd=bwd,
                                 peak_MB=peak / 2**20, status=status))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "bench_efficiency.csv"), index=False)


if __name__ == "__main__":
    what = sys.argv[1:] or ["fidelity", "efficiency"]
    if "fidelity" in what:
        fidelity()
    if "efficiency" in what:
        efficiency()
