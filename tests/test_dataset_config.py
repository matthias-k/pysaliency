from __future__ import absolute_import, division, print_function, unicode_literals

import os

import pytest
import numpy as np

import pysaliency
import pysaliency.dataset_config as dc


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


@pytest.fixture
def hdf5_dataset(tmpdir, fixation_trains, stimuli):
    stimuli.to_hdf5(os.path.join(str(tmpdir), 'stimuli.hdf5'))
    fixation_trains.to_hdf5(os.path.join(str(tmpdir), 'fixations.hdf5'))
    return str(tmpdir)


def test_load_dataset(hdf5_dataset, stimuli, fixation_trains):
    loaded_stimuli, loaded_fixations = dc.load_dataset_from_config({
        'stimuli': os.path.join(hdf5_dataset, 'stimuli.hdf5'),
        'fixations': os.path.join(hdf5_dataset, 'fixations.hdf5'),
    })

    assert len(loaded_stimuli) == len(stimuli)
    np.testing.assert_allclose(loaded_fixations.x, fixation_trains.x)


def test_load_dataset_with_filter(hdf5_dataset, stimuli, fixation_trains):
    loaded_stimuli, loaded_fixations = dc.load_dataset_from_config({
        'stimuli': os.path.join(hdf5_dataset, 'stimuli.hdf5'),
        'fixations': os.path.join(hdf5_dataset, 'fixations.hdf5'),
        'filters': [{
            'type': 'filter_fixations_by_number',
            'parameters': {
                'intervals': [[0, 2]],
            },
        }],
    })

    assert len(loaded_stimuli) == len(stimuli)
    assert len(loaded_fixations.x) == 6
    assert np.all(loaded_fixations.lengths < 2)
