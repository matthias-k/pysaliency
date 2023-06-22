import os

import numpy as np

import pysaliency
from pysaliency.external_models.deepgaze import DeepGazeI, DeepGazeIIE

import pytest

@pytest.fixture(scope='module')
def color_stimulus():
    return np.load(os.path.join('tests', 'external_models', 'color_stimulus.npy'))


@pytest.fixture(scope='module')
def grayscale_stimulus():
    return np.load(os.path.join('tests', 'external_models', 'grayscale_stimulus.npy'))


@pytest.fixture
def stimuli(color_stimulus, grayscale_stimulus):
    return pysaliency.Stimuli([color_stimulus, grayscale_stimulus])


@pytest.fixture
def fixations():
    return pysaliency.FixationTrains.from_fixation_trains(
        [[700, 730], [430, 450]],
        [[300, 300], [500, 500]],
        [[0, 1], [0, 1]],
        ns=[0, 1],
        subjects=[0, 0],
    )


def test_deepgaze1(stimuli, fixations):
    model = DeepGazeI(centerbias_model=pysaliency.UniformModel(), device='cpu')

    ig = model.information_gain(stimuli, fixations)

    np.testing.assert_allclose(ig, 0.9455161648442227, rtol=5e-6)


def test_deepgaze2e(stimuli, fixations):
    model = DeepGazeIIE(centerbias_model=pysaliency.UniformModel(), device='cpu')

    ig = model.information_gain(stimuli, fixations)

    np.testing.assert_allclose(ig, 3.918556860669079)