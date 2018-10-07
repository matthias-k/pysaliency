from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

import pysaliency


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
def stimuli():
    return pysaliency.Stimuli([np.random.randn(40, 40, 3),
                               np.random.randn(40, 40, 3)])


def test_sim_saliency_map(stimuli):
    gsmm = GaussianSaliencyModel()

    sim_model = pysaliency.SIMSaliencyMapModel(gsmm, kernel_size=2, max_iter=100, initial_learning_rate=1e-6,
        learning_rate_decay_scheme='validation_loss')

    smap = sim_model.saliency_map(stimuli[0])
    assert smap.shape == (40, 40)
