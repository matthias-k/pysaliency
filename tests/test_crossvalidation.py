from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pytest
from sklearn.model_selection import cross_val_score

import pysaliency
from pysaliency.baseline_utils import RegularizedKernelDensityEstimator, ScikitLearnImageCrossValidationGenerator, ScikitLearnImageSubjectCrossValidationGenerator, fixations_to_scikit_learn


class ConstantSaliencyModel(pysaliency.Model):
    def _log_density(self, stimulus):
        return np.zeros((stimulus.shape[0], stimulus.shape[1])) - np.log(stimulus.shape[0]) - np.log(stimulus.shape[1])


class GaussianSaliencyModel(pysaliency.Model):
    def _log_density(self, stimulus):
        height = stimulus.shape[0]
        width = stimulus.shape[1]
        YS, XS = np.mgrid[:height, :width]
        r_squared = (XS-0.5*width)**2 + (YS-0.5*height)**2
        size = np.sqrt(width**2+height**2)
        values = np.ones((stimulus.shape[0], stimulus.shape[1]))*np.exp(-0.5*(r_squared/size))
        density = values / values.sum()
        return np.log(density)


@pytest.fixture
def fixation_trains():
    xs_trains = [
        [0, 1, 2],
        [2, 2],
        [1, 5, 3],
        [10]]
    ys_trains = [
        [10, 11, 12],
        [12, 12],
        [21, 25, 33],
        [11]]
    ts_trains = [
        [0, 200, 600],
        [100, 400],
        [50, 500, 900],
        [100]]
    ns = [0, 0, 1, 2]
    subjects = [0, 1, 1, 0]
    return pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)


@pytest.fixture
def stimuli():
    return pysaliency.Stimuli([np.random.randn(40, 40, 3),
                               np.random.randn(40, 40, 3),
                               np.random.randn(40, 40, 3)])


def _unpack_crossval(cv):
    for train_inds, test_inds in cv:
        yield list(train_inds), list(test_inds)


def unpack_crossval(cv):
    return list(_unpack_crossval(cv))


def test_image_crossvalidation(stimuli, fixation_trains):
    gsmm = GaussianSaliencyModel()

    cv = ScikitLearnImageCrossValidationGenerator(stimuli, fixation_trains)

    assert unpack_crossval(cv) == [
        ([False, False, False, False, False, True, True, True, True],
         [True, True, True, True, True, False, False, False, False]),
        ([True, True, True, True, True, False, False, False, True],
         [False, False, False, False, False, True, True, True, False]),
        ([True, True, True, True, True, True, True, True, False],
         [False, False, False, False, False, False, False, False, True])
    ]

    X = fixations_to_scikit_learn(fixation_trains, normalize=stimuli, add_shape=True)

    assert cross_val_score(
        RegularizedKernelDensityEstimator(bandwidth=0.1, regularization=0.1),
        X,
        cv=cv,
        verbose=0).sum()


def test_image_subject_crossvalidation(stimuli, fixation_trains):
    gsmm = GaussianSaliencyModel()

    cv = ScikitLearnImageSubjectCrossValidationGenerator(stimuli, fixation_trains)

    assert unpack_crossval(cv) == [
        ([3, 4], [0, 1, 2]),
        ([0, 1, 2], [3, 4])
    ]

    X = fixations_to_scikit_learn(fixation_trains, normalize=stimuli, add_shape=True)

    assert cross_val_score(
        RegularizedKernelDensityEstimator(bandwidth=0.1, regularization=0.1),
        X,
        cv=cv,
        verbose=0).sum()
