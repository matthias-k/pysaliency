# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function  # , unicode_literals

import numpy as np

from tqdm import tqdm


def optimize_for_information_gain(
        model, fit_stimuli, fit_fixations,
        nonlinearity_target='density',
        nonlinearity_values='logdensity',
        num_nonlinearity=20,
        num_centerbias=12,
        blur_radius=0,
        optimize=None,
        average='image',
        saliency_min=None,
        saliency_max=None,
        batch_size=1,
        verbose=0,
        return_optimization_result=False,
        maxiter=1000,
        method='trust-constr',
        minimize_options=None,
        framework='torch'):
    """ convert saliency map model into probabilistic model as described in Kümmerer et al, PNAS 2015.
    """

    if saliency_min is None or saliency_max is None:
        smax = -np.inf
        smin = np.inf
        for s in tqdm(fit_stimuli, disable=verbose < 2):
            smap = model.saliency_map(s)
            smax = np.max([smax, smap.max()])
            smin = np.min([smin, smap.min()])

        if saliency_min is None:
            saliency_min = smin
        if saliency_max is None:
            saliency_max = smax

    if framework == 'theano':
        assert nonlinearity_target == 'density'
        assert nonlinearity_values == 'logdensity'
        assert average == 'fixations'
        assert batch_size == 1
        assert minimize_options is None

        from .saliency_map_conversion_theano import optimize_for_information_gain
        return optimize_for_information_gain(
            model, fit_stimuli, fit_fixations,
            num_nonlinearity=num_nonlinearity,
            num_centerbias=num_centerbias,
            blur_radius=blur_radius,
            optimize=optimize,
            saliency_min=saliency_min,
            saliency_max=saliency_max,
            verbose=verbose,
            return_optimization_result=return_optimization_result,
            maxiter=maxiter,
            method=method
        )
    elif framework == 'torch':
        from .saliency_map_conversion_torch import optimize_saliency_map_conversion
        return optimize_saliency_map_conversion(
            model, fit_stimuli, fit_fixations,
            nonlinearity_target=nonlinearity_target,
            nonlinearity_values=nonlinearity_values,
            saliency_min=saliency_min,
            saliency_max=saliency_max,
            optimize=optimize,
            average=average,
            batch_size=batch_size,
            num_nonlinearity=num_nonlinearity,
            num_centerbias=num_centerbias,
            blur_radius=blur_radius,
            verbose=verbose,
            return_optimization_result=return_optimization_result,
            maxiter=maxiter,
            minimize_options=minimize_options,
            method=method
        )
