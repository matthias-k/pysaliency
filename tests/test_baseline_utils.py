from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

import pysaliency
from pysaliency.baseline_utils import fill_fixation_map, KDEGoldModel


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