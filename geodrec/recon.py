"""Exact, memory-efficient backpropagation through morphological reconstruction.

Reconstruction by dilation of a marker f under a mask g is the max-min (bottleneck)
path function

    R_g(f)(p) = max_{paths q=p_0,...,p_m=p} min(f(q), g(p_0), ..., g(p_m)),

so every output value is a *copy* of a single input value, either f(q) or g(r).
The forward pass below tracks, for every pixel, the flat index of the input entry
it copies (its *provenance*).  The backward pass is then a single scatter-add,
whose memory cost does not depend on the number of geodesic iterations.

Tensors are (B, C, *spatial) with 1, 2 or 3 spatial dimensions.  Provenance
indices live in [0, 2n), n = prod(spatial): indices < n refer to the marker,
indices >= n to the mask, both within the same (b, c) plane.
"""
import itertools

import torch
import torch.nn.functional as F


def neighbourhood(ndim, connectivity):
    """Offsets of the elementary structuring element (origin included).

    connectivity=1 -> 4-adjacency in 2D (6 in 3D); connectivity=ndim -> 8 (26)."""
    offs = []
    for d in itertools.product((-1, 0, 1), repeat=ndim):
        if sum(abs(x) for x in d) <= connectivity:
            offs.append(d)
    return offs


def _shift(x, off, fill):
    """y[p] = x[p + off] (out-of-domain entries set to `fill`); x is (P, *spatial)."""
    ndim = x.dim() - 1
    pad = []
    for o in reversed(off):
        pad += [max(-o, 0), max(o, 0)]
    xp = F.pad(x, pad, value=fill) if any(pad) else x
    sl = [slice(None)]
    for i, o in enumerate(off):
        start = max(o, 0)  # original index i sits at i + max(-o, 0) once padded
        sl.append(slice(start, start + x.shape[1 + i]))
    return xp[tuple(sl)]


class _ProvenanceState:
    """Static buffers for one reconstruction; X is a view of the interior of a
    (-inf)-padded buffer so that all neighbour shifts are free views."""

    def __init__(self, f, g, offs):
        P, spatial = g.shape[0], g.shape[1:]
        n = g[0].numel()
        self.P, self.n, self.g = P, n, g
        self.Xp = torch.full((P, *[s + 2 for s in spatial]), float("-inf"),
                             dtype=g.dtype, device=g.device)
        inner = (slice(None),) + tuple(slice(1, s + 1) for s in spatial)
        self.X = self.Xp[inner]
        self.views = [self.Xp[(slice(None),) + tuple(slice(1 + o, 1 + o + s)
                                                     for o, s in zip(off, spatial))]
                      for off in [(0,) * len(spatial)] + offs]
        strides = [1] * len(spatial)
        for i in range(len(spatial) - 2, -1, -1):
            strides[i] = strides[i + 1] * spatial[i + 1]
        self.off_flat = torch.tensor([0] + [sum(o * st for o, st in zip(off, strides))
                                            for off in offs], device=g.device)
        self.idx = torch.arange(n, device=g.device).reshape(1, *spatial).expand(P, *spatial)
        self.g_src = self.idx + n
        from_marker = f <= g
        self.X.copy_(torch.where(from_marker, f, g))
        self.src = torch.where(from_marker, self.idx, self.g_src).contiguous()

    def step(self):
        best, arg = torch.stack(self.views).max(0)
        nb = (self.idx + self.off_flat[arg]).clamp_(0, self.n - 1)
        best_src = torch.gather(self.src.view(self.P, -1), 1, nb.view(self.P, -1)).view_as(self.src)
        clipped = best > self.g
        cand = torch.where(clipped, self.g, best)
        cand_src = torch.where(clipped, self.g_src, best_src)
        upd = cand > self.X  # strict: keeps the invariant X[p] == input[src[p]]
        self.X.copy_(torch.where(upd, cand, self.X))
        self.src.copy_(torch.where(upd, cand_src, self.src))

    def sweep(self, axis, reverse):
        """Propagate along one axis in one direction in ceil(log2 L) doubling steps.
        Solves x_i = max(X_i, min(g_i, x_{i-1})), a linear recurrence in the (max, min)
        semiring: A(i) = max_{i-2d<j<=i} min(X_j, g_{j+1..i}), G(i) = min g over (i-2d, i]."""
        mv = lambda t: t.movedim(1 + axis, -1).flip(-1) if reverse else t.movedim(1 + axis, -1)
        A, As, G, Gs = mv(self.X), mv(self.src), mv(self.g), mv(self.g_src)
        L = A.shape[-1]
        inf = float("inf")
        d = 1
        while d < L:
            A_sh = F.pad(A[..., :-d], (d, 0), value=-inf)
            As_sh = F.pad(As[..., :-d], (d, 0), value=0)
            cand = torch.minimum(A_sh, G)
            cand_src = torch.where(A_sh <= G, As_sh, Gs)
            upd = cand > A
            A = torch.where(upd, cand, A)
            As = torch.where(upd, cand_src, As)
            G_sh = F.pad(G[..., :-d], (d, 0), value=inf)
            Gs_sh = F.pad(Gs[..., :-d], (d, 0), value=0)
            take = G_sh < G
            G = torch.where(take, G_sh, G)
            Gs = torch.where(take, Gs_sh, Gs)
            d *= 2
        back = lambda t: t.flip(-1).movedim(-1, 1 + axis) if reverse else t.movedim(-1, 1 + axis)
        self.X.copy_(back(A))
        self.src.copy_(back(As))

    def round(self):
        """Forward/backward sweeps along every axis, then one geodesic dilation by the
        full neighbourhood (diagonal moves).  Every operation copies a value together
        with its provenance and is bounded by the reconstruction, so iterating rounds
        converges to the same fixed point as plain geodesic dilations."""
        for axis in range(self.X.dim() - 1):
            self.sweep(axis, False)
            self.sweep(axis, True)
        self.step()


@torch.no_grad()
def reconstruct_with_provenance(marker, mask, connectivity=None, max_iter=None,
                                check_every=None, cuda_graph=True, method="dilation"):
    """Reconstruction by dilation of `marker` (clipped to `mask`) under `mask`, by
    parallel operations that carry provenance.

    method="dilation": elementary geodesic dilations (Algorithm 1);
    method="scan": rounds of semiring scans along the axes + one geodesic dilation.
    Returns (R, src, n_iter) where src[p] is the flat provenance index of R[p].
    n_iter counts iterations (dilations or rounds) run; stability is tested every
    `check_every` iterations (default 16 for dilation, 1 for scan)."""
    if check_every is None:
        check_every = 16 if method == "dilation" else 1
    shape = mask.shape
    spatial = shape[2:]
    ndim = len(spatial)
    connectivity = ndim if connectivity is None else connectivity
    offs = [o for o in neighbourhood(ndim, connectivity) if any(o)]
    P = shape[0] * shape[1]
    st = _ProvenanceState(marker.detach().reshape(P, *spatial),
                          mask.detach().reshape(P, *spatial).contiguous(), offs)
    step = st.step if method == "dilation" else st.round
    chunk = check_every if max_iter is None else max(1, min(check_every, max_iter))
    graph = None
    if cuda_graph and mask.is_cuda and max_iter is None:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm-up iterations (these count)
            for _ in range(chunk):
                step()
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(chunk):
                step()
        # capture does not execute: the warm-up chunk is the first chunk
    it = chunk if graph is not None else 0
    snapshot = st.X.clone()
    while max_iter is None or it < max_iter:
        if graph is not None:
            graph.replay()
            it += chunk
        else:
            for _ in range(chunk if max_iter is None else min(chunk, max_iter - it)):
                step()
                it += 1
        if torch.equal(st.X, snapshot):
            break
        snapshot.copy_(st.X)
    return st.X.reshape(shape).clone(), st.src.reshape(shape), it


class _ReconstructionByDilation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, marker, mask, connectivity, max_iter, cuda_graph, method):
        R, src, it = reconstruct_with_provenance(marker, mask, connectivity, max_iter,
                                                 cuda_graph=cuda_graph, method=method)
        ctx.save_for_backward(src)
        ctx.n_iter = it
        return R

    @staticmethod
    def backward(ctx, grad):
        (src,) = ctx.saved_tensors
        shape = grad.shape
        P = shape[0] * shape[1]
        n = grad[0, 0].numel()
        out = torch.zeros(P, 2 * n, dtype=grad.dtype, device=grad.device)
        out.scatter_add_(1, src.reshape(P, n), grad.reshape(P, n))
        gf = out[:, :n].reshape(shape) if ctx.needs_input_grad[0] else None
        gg = out[:, n:].reshape(shape) if ctx.needs_input_grad[1] else None
        return gf, gg, None, None, None, None


def reconstruction_by_dilation(marker, mask, connectivity=None, max_iter=None, cuda_graph=True,
                               method="dilation"):
    """Differentiable R_mask(marker) with provenance-based (exact) backward."""
    return _ReconstructionByDilation.apply(marker, mask, connectivity, max_iter, cuda_graph, method)


def reconstruction_by_erosion(marker, mask, connectivity=None, max_iter=None):
    """Dual reconstruction: R^*_mask(marker) = -R_{-mask}(-marker)."""
    return -reconstruction_by_dilation(-marker, -mask, connectivity, max_iter)


def hdome(u, h, connectivity=None, rec=reconstruction_by_dilation):
    """h-dome transform u - R_u(u - h); h may be a scalar or a broadcastable tensor."""
    return u - rec(u - h, u, connectivity)


def hmaxima_regional(u, h, connectivity=None):
    """Binary h-maxima: regional maxima of R_u(u - h) (non-differentiable)."""
    with torch.no_grad():
        r = reconstruction_by_dilation(u - h, u, connectivity)
        eps = 1e-6 * (1 + r.abs().max())
        return (r - reconstruction_by_dilation(r - eps, r, connectivity)) > 0
