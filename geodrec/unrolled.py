"""Baseline: reconstruction as an unrolled loop of geodesic dilations, backpropagated
by autograd (the approach of current trainable h-maxima pipelines)."""
import torch

from .recon import _shift, neighbourhood


def geodesic_dilation(X, g, offs):
    best = X
    for off in offs:
        best = torch.maximum(best, _shift(X, off, float("-inf")))
    return torch.minimum(best, g)


def reconstruction_unrolled(marker, mask, connectivity=None, n_iter=None, check_every=8):
    """K geodesic dilations with an autograd graph.  n_iter=None iterates to stability
    (stability is tested without building extra graph)."""
    shape = mask.shape
    spatial = shape[2:]
    ndim = len(spatial)
    connectivity = ndim if connectivity is None else connectivity
    offs = [o for o in neighbourhood(ndim, connectivity) if any(o)]
    P = shape[0] * shape[1]
    g = mask.reshape(P, *spatial)
    X = torch.minimum(marker.reshape(P, *spatial), g)
    it = 0
    prev = X.detach()
    while n_iter is None or it < n_iter:
        X = geodesic_dilation(X, g, offs)
        it += 1
        if n_iter is None and it % check_every == 0:
            cur = X.detach()
            if torch.equal(cur, prev):
                break
            prev = cur
    return X.reshape(shape)
