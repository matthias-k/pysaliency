from __future__ import absolute_import, print_function, division, unicode_literals

import numpy as np


def normalize_saliency_map(saliency_map, cdf, cdf_bins):
    """ Normalize saliency to make saliency values distributed according to a given CDF
    """

    smap = saliency_map.copy()
    shape = smap.shape
    smap = smap.flatten()
    smap = np.argsort(np.argsort(smap)).astype(float)
    smap /= 1.0 * len(smap)

    inds = np.searchsorted(cdf, smap, side='right')
    smap = cdf_bins[inds]
    smap = smap.reshape(shape)
    smap = smap.reshape(shape)
    return smap


def convert_saliency_map_to_density(saliency_map, minimum_value=0.0):
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value

    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()

    return saliency_map


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


def CC(saliency_map_1, saliency_map_2):
    def normalize(saliency_map):
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 0.0
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]


def probabilistic_image_based_kl_divergence(logp1, logp2, log_regularization=0, quotient_regularization=0):
    if log_regularization or quotient_regularization:
        return (np.exp(logp2) * np.log(log_regularization + np.exp(logp2) / (np.exp(logp1) + quotient_regularization))).sum()
    else:
        return (np.exp(logp2) * (logp2 - logp1)).sum()


def image_based_kl_divergence(saliency_map_1, saliency_map_2, minimum_value=1e-20, log_regularization=0, quotient_regularization=0):
    """ KLDiv. Function is not symmetric. saliency_map_2 is treated as empirical saliency map. """
    log_density_1 = np.log(convert_saliency_map_to_density(saliency_map_1, minimum_value=minimum_value))
    log_density_2 = np.log(convert_saliency_map_to_density(saliency_map_2, minimum_value=minimum_value))

    return probabilistic_image_based_kl_divergence(log_density_1, log_density_2, log_regularization=log_regularization, quotient_regularization=quotient_regularization)


def MIT_KLDiv(saliency_map_1, saliency_map_2):
    """ compute image-based KL divergence with same hyperparameters as in Tuebingen/MIT Saliency Benchmark """
    return image_based_kl_divergence(
        saliency_map_1,
        saliency_map_2,
        minimum_value=0,
        log_regularization=2.2204e-16,
        quotient_regularization=2.2204e-16
    )


def SIM(saliency_map_1, saliency_map_2):
    """ Compute similiarity metric. """
    density_1 = convert_saliency_map_to_density(saliency_map_1, minimum_value=0)
    density_2 = convert_saliency_map_to_density(saliency_map_2, minimum_value=0)

    return np.min([density_1, density_2], axis=0).sum()
