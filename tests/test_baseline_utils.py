from __future__ import absolute_import, division, print_function, unicode_literals

from collections import OrderedDict

import numpy as np
import pytest

import pysaliency
from pysaliency.baseline_utils import (
    CrossvalMultipleRegularizations,
    GeneralMixtureKernelDensityEstimator,
    KDEGoldModel,
    MixtureKernelDensityEstimator,
    ScikitLearnImageCrossValidationGenerator,
    fill_fixation_map,
)


@pytest.fixture
def fixation_trains():
    xs_trains = [
        [15, 20, 25],
        [10, 30],
        [30, 20, 10]]
    ys_trains = [
        [13, 21, 10],
        [15, 35],
        [22, 5, 18]]
    ts_trains = [
        [0, 200, 600],
        [100, 400],
        [50, 500, 900]]
    ns = [0, 0, 1]
    subjects = [0, 1, 1]
    return pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)


@pytest.fixture
def stimuli():
    return pysaliency.Stimuli([np.random.randn(40, 40, 3),
                               np.random.randn(40, 40, 3)])


def test_fixation_map():
    fixations = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 2],
        [1, 2],
        [2, 1]])

    fixation_map = np.zeros((3, 3))
    fill_fixation_map(fixation_map, fixations)

    np.testing.assert_allclose(fixation_map, np.array([
        [1, 0, 0],
        [0, 2, 2],
        [0, 1, 0]]))


def test_kde_gold_model(stimuli, fixation_trains):
    bandwidth = 0.1
    kde_gold_model = KDEGoldModel(stimuli, fixation_trains, bandwidth=bandwidth)
    spaced_kde_gold_model = KDEGoldModel(stimuli, fixation_trains, bandwidth=bandwidth, grid_spacing=2)

    full_log_density = kde_gold_model.log_density(stimuli[0])
    spaced_log_density = spaced_kde_gold_model.log_density(stimuli[0])

    kl_div1 = np.sum(np.exp(full_log_density) * (full_log_density - spaced_log_density)) / np.log(2)
    kl_div2 = np.sum(np.exp(spaced_log_density) * (spaced_log_density - full_log_density)) / np.log(2)

    assert kl_div1 < 0.002
    assert kl_div2 < 0.002

    full_ll = kde_gold_model.information_gain(stimuli, fixation_trains, average='image')
    spaced_ll = spaced_kde_gold_model.information_gain(stimuli, fixation_trains, average='image')
    print(full_ll, spaced_ll)
    np.testing.assert_allclose(full_ll, 2.1912009255501252)
    np.testing.assert_allclose(spaced_ll, 2.191055750664578)


def test_general_mixture_kernel_density_estimator():
    # Test initialization
    estimator = GeneralMixtureKernelDensityEstimator(bandwidth=1.0, regularizations=[0.2, 0.1], regularizing_log_likelihoods=[[-1, 0.0], [-0.1, -10.0], [-10, -0.1]])
    assert estimator.bandwidth == 1.0
    assert np.allclose(estimator.regularizations, [0.2, 0.1])
    assert np.allclose(estimator.regularizing_log_likelihoods, [[-1, 0.0], [-0.1, -10.0], [-10, -0.1]])

    # Test setup
    estimator.setup()
    assert estimator.kde is not None
    assert estimator.kde_constant is not None
    assert estimator.regularization_constants is not None

    # Test fit
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    estimator.fit(X)
    assert estimator.kde is not None

    # Test score_samples
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    logliks = estimator.score_samples(X)
    assert logliks.shape == (3,)
    np.testing.assert_allclose(logliks, [-1.49141561, -1.40473767, -1.95213405])

    # Test score
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    score = estimator.score(X)
    assert isinstance(score, float)


def test_mixture_kernel_density_estimator():
    # Test initialization
    estimator = MixtureKernelDensityEstimator(bandwidth=1.0, regularization=1.0e-5, regularizing_log_likelihoods=[-0.3, -0.2, -0.1])
    assert estimator.bandwidth == 1.0
    assert estimator.regularization == 1.0e-5

    # Test setup
    estimator.setup()
    assert estimator.kde is not None
    assert estimator.kde_constant is not None
    assert estimator.uniform_constant is not None

    # Test fit
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    estimator.fit(X)
    assert estimator.kde is not None

    # Test score_samples
    X = np.array([[0, 0.2, 0], [0.3, 1, 1], [1, 1, 2]])
    logliks = estimator.score_samples(X)
    assert logliks.shape == (3,)
    np.testing.assert_allclose(logliks, [-2.56662505, -2.5272495,  -2.38495638])

    # Test score
    X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    score = estimator.score(X)
    assert isinstance(score, float)


def test_crossval_multiple_regularizations(stimuli, fixation_trains):
    # Test initialization
    regularization_models = OrderedDict([('model1', pysaliency.UniformModel()), ('model2', pysaliency.models.GaussianModel())])
    crossvalidation = ScikitLearnImageCrossValidationGenerator(stimuli, fixation_trains)
    estimator = CrossvalMultipleRegularizations(stimuli, fixation_trains, regularization_models, crossvalidation)
    assert estimator.cv is crossvalidation
    assert estimator.mean_area is not None
    assert estimator.X is not None
    assert estimator.regularization_log_likelihoods is not None

    # Test score
    log_bandwidth = 0.1
    log_regularizations = [0.1, 0.2]

    score = estimator.score(log_bandwidth, *log_regularizations)
    assert isinstance(score, float)
    np.testing.assert_allclose(score, -1.4673831679692528e-10)