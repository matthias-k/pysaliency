from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np


def normalize_saliency_map(saliency_map, cdf, cdf_bins):
    """ Normalize saliency to make saliency values distributed according to a given CDF
    """

    smap = saliency_map.copy()
    shape = smap.shape
    smap = smap.flatten()
    smap = np.argsort(np.argsort(smap)).astype(float)
    smap /= 1.0*len(smap)

    inds = np.searchsorted(cdf, smap, side='right')
    smap = cdf_bins[inds]
    smap = smap.reshape(shape)
    smap = smap.reshape(shape)
    return smap


def NSS(saliency_map, xs, ys):
    xs = np.asarray(xs, dtype=np.int)
    ys = np.asarray(ys, dtype=np.int)

    mean = saliency_map.mean()
    std = saliency_map.std()

    value = saliency_map[ys, xs].copy()
    value -= mean

    if std:
        value /= std

    return value
