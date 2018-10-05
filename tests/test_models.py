from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

import pysaliency


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
        [1, 5, 3]]
    ys_trains = [
        [10, 11, 12],
        [12, 12],
        [21, 25, 33]]
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


def test_log_likelihood_constant(stimuli, fixation_trains):
    csmm = ConstantSaliencyModel()

    log_likelihoods = csmm.log_likelihoods(stimuli, fixation_trains)
    np.testing.assert_allclose(log_likelihoods, -np.log(40*40))


def test_log_likelihood_gauss(stimuli, fixation_trains):
    gsmm = GaussianSaliencyModel()

    log_likelihoods = gsmm.log_likelihoods(stimuli, fixation_trains)
    np.testing.assert_allclose(log_likelihoods, np.array([-10.276835,  -9.764182,  -9.286885,  -9.286885,
                                                          -9.286885,   -9.057075,  -8.067126,  -9.905604]))
