"""BBBC039 (U2OS nuclei, Hoechst) loading: normalised images and nucleus centroids."""
import os

import numpy as np
from scipy import ndimage as ndi
from skimage import io

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "BBBC039")
MIN_AREA = 20


def instances(mask_rgba):
    """Masks colour-code nuclei with few labels so that touching nuclei differ;
    instances are the connected components of each label value."""
    m = mask_rgba[..., 0]
    lab = np.zeros(m.shape, np.int32)
    nxt = 0
    for v in np.unique(m):
        if v == 0:
            continue
        cc, k = ndi.label(m == v)
        lab[cc > 0] = cc[cc > 0] + nxt
        nxt += k
    areas = np.bincount(lab.ravel())
    small = np.where(areas < MIN_AREA)[0]
    lab[np.isin(lab, small[small > 0])] = 0
    return lab


def centroids(lab):
    ids = np.unique(lab)
    ids = ids[ids > 0]
    return np.array(ndi.center_of_mass(lab > 0, lab, ids)).reshape(-1, 2)


def normalise(img):
    lo, hi = np.percentile(img, (1, 99.8))
    return np.clip((img.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1.5).astype(np.float32)


_cache = {}


def load_split(split):
    """Returns (images, list of (k,2) centroid arrays)."""
    if split in _cache:
        return _cache[split]
    names = [l.strip() for l in open(os.path.join(ROOT, "metadata", f"{split}.txt")) if l.strip()]
    imgs, pts = [], []
    for nm in names:
        stem = os.path.splitext(nm)[0]
        imgs.append(normalise(io.imread(os.path.join(ROOT, "images", stem + ".tif"))))
        pts.append(centroids(instances(io.imread(os.path.join(ROOT, "masks", stem + ".png")))))
    _cache[split] = (imgs, pts)
    return imgs, pts
