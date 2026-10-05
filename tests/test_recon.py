import numpy as np
import pytest
import torch
from scipy import ndimage as ndi
from skimage.morphology import reconstruction as sk_rec

from geodrec import (hdome, reconstruct_with_provenance, reconstruction_by_dilation,
                     reconstruction_by_erosion, reconstruction_unrolled)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def footprint(ndim, conn):
    return ndi.generate_binary_structure(ndim, conn)


def smooth_field(shape, seed, sigma=2.0):
    rng = np.random.default_rng(seed)
    return ndi.gaussian_filter(rng.random(shape), sigma)


METHODS = ["dilation", "scan"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape,conn", [((40, 50), 1), ((40, 50), 2), ((12, 14, 16), 1),
                                        ((12, 14, 16), 3)])
@pytest.mark.parametrize("seed", range(3))
def test_forward_matches_skimage(shape, conn, seed, method):
    g = smooth_field(shape, seed)
    f = g - 0.02
    ref = sk_rec(f, g, method="dilation", footprint=footprint(len(shape), conn))
    out = reconstruction_by_dilation(torch.tensor(f)[None, None], torch.tensor(g)[None, None], conn,
                                     method=method)
    np.testing.assert_allclose(out[0, 0].numpy(), ref, atol=1e-12)
    ref_e = sk_rec(g + 0.02, g, method="erosion", footprint=footprint(len(shape), conn))
    out_e = reconstruction_by_erosion(torch.tensor(g + 0.02)[None, None], torch.tensor(g)[None, None], conn)
    np.testing.assert_allclose(out_e[0, 0].numpy(), ref_e, atol=1e-12)


def test_marker_above_mask_is_clipped():
    rng = np.random.default_rng(0)
    f, g = rng.random((30, 30)), rng.random((30, 30))
    ref = sk_rec(np.minimum(f, g), g, method="dilation", footprint=footprint(2, 2))
    out = reconstruction_by_dilation(torch.tensor(f)[None, None], torch.tensor(g)[None, None])
    np.testing.assert_allclose(out[0, 0].numpy(), ref, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("conn", [1, 2])
def test_provenance_invariant(conn, method):
    g = torch.rand(3, 2, 33, 47, device=DEV)
    f = g - 0.3 * torch.rand_like(g)
    R, src, _ = reconstruct_with_provenance(f, g, conn, method=method)
    R2, _, _ = reconstruct_with_provenance(f, g, conn, method="dilation")
    assert torch.equal(R, R2)
    vals = torch.cat([f.flatten(2), g.flatten(2)], dim=2)
    picked = torch.gather(vals, 2, src.flatten(2)).reshape(R.shape)
    assert torch.equal(picked, R)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("conn", [1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_gradient_equals_full_unrolled(conn, seed, method):
    torch.manual_seed(seed)
    g = torch.rand(2, 1, 40, 40, dtype=torch.float64, device=DEV)
    f = (g - 0.5 * torch.rand_like(g)).requires_grad_()
    g.requires_grad_()
    w = torch.randn_like(g)
    (reconstruction_by_dilation(f, g, conn, method=method) * w).sum().backward()
    gf, gg = f.grad.clone(), g.grad.clone()
    f.grad = g.grad = None
    (reconstruction_unrolled(f, g, conn) * w).sum().backward()
    torch.testing.assert_close(gf, f.grad, rtol=0, atol=1e-12)
    torch.testing.assert_close(gg, g.grad, rtol=0, atol=1e-12)


def test_gradcheck_finite_differences():
    torch.manual_seed(1)
    g = torch.rand(1, 1, 9, 11, dtype=torch.float64)
    f = (g - 0.4 * torch.rand_like(g)).requires_grad_()
    g.requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: reconstruction_by_dilation(a, b), (f, g),
                                    eps=1e-7, atol=1e-6)


def test_hdome_gradients_wrt_u_and_h():
    torch.manual_seed(2)
    u = torch.rand(1, 1, 24, 24, dtype=torch.float64).requires_grad_()
    h = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: hdome(a, b), (u, h), eps=1e-7, atol=1e-6)
    d = hdome(u, h)
    assert float(d.min()) >= -1e-12 and float(d.max()) <= 0.2 + 1e-12


def test_serpentine_scan_matches_dilation():
    from experiments.bench import serpentine_mask
    g = torch.tensor(serpentine_mask(64) * (0.5 + 0.5 * np.random.default_rng(0).random((64, 64))),
                     device=DEV)[None, None]
    f = torch.zeros_like(g)
    f[..., 0, 0] = g[..., 0, 0]
    Ra, sa, ka = reconstruct_with_provenance(f, g, method="dilation", check_every=1)
    Rb, sb, kb = reconstruct_with_provenance(f, g, method="scan")
    assert torch.equal(Ra, Rb) and torch.equal(sa, sb)
    assert kb < ka / 10
