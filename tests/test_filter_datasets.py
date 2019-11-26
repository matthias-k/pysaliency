from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np

import pysaliency
import pysaliency.filter_datasets as filter_datasets


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


def test_filter_fixations_by_number(fixation_trains):
    fixations = filter_datasets.filter_fixations_by_number(fixation_trains, 0)
    assert len(fixations.x) == 3
    np.testing.assert_allclose(fixations.lengths, 0)

    fixations = filter_datasets.filter_fixations_by_number(fixation_trains, 1)
    assert len(fixations.x) == 3
    np.testing.assert_allclose(fixations.lengths, 1)

    fixations = filter_datasets.filter_fixations_by_number(fixation_trains, [[0, 2]])
    assert len(fixations.x) == 6
    assert np.all(fixations.lengths < 2)

    fixations = filter_datasets.filter_fixations_by_number(fixation_trains, [[0, 2], 2])
    assert len(fixations.x) == 8
    np.testing.assert_allclose(fixations.x, fixation_trains.x)
