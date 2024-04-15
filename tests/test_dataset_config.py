from __future__ import absolute_import, division, print_function, unicode_literals

import os

import numpy as np
import pytest
from imageio import imwrite
from test_filter_datasets import assert_stimuli_equal

import pysaliency
import pysaliency.dataset_config as dc
from pysaliency.filter_datasets import create_subset
from tests.datasets.utils import assert_scanpath_fixations_equal
from tests.datasets.utils import assert_fixations_equal


@pytest.fixture
def scanpath_fixations():
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
    tasks = [0, 1, 0]
    multi_dim_attribute = [[0.0, 1],[0, 3], [4, 5.5]]
    durations_train = [
        [42, 25, 100],
        [99, 98],
        [200, 150, 120]
    ]
    return pysaliency.ScanpathFixations(pysaliency.Scanpaths(
        xs=xs_trains,
        ys=ys_trains,
        ts=ts_trains,
        n=ns,
        subject=subjects,
        scanpath_attributes={
            'task': tasks,
            'multi_dim_attribute': multi_dim_attribute,
        },
        fixation_attributes={'durations': durations_train},
        attribute_mapping={'durations': 'duration'},
    ))


@pytest.fixture
def file_stimuli_with_attributes(tmpdir):
    filenames = []
    for i in range(3):
        filename = tmpdir.join('stimulus_{:04d}.png'.format(i))
        imwrite(str(filename), np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
        filenames.append(str(filename))

    for sub_directory_index in range(3):
        sub_directory = tmpdir.join('sub_directory_{:04d}'.format(sub_directory_index))
        sub_directory.mkdir()
        for i in range(5):
            filename = sub_directory.join('stimulus_{:04d}.png'.format(i))
            imwrite(str(filename), np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8))
            filenames.append(str(filename))
    attributes = {
        'dva': list(range(len(filenames))),
        'other_stuff': np.random.randn(len(filenames)),
        'some_strings': list('abcdefghijklmnopqr'),
    }
    return pysaliency.FileStimuli(filenames=filenames, attributes=attributes)


@pytest.fixture
def stimuli():
    return pysaliency.Stimuli([np.random.randn(40, 40, 3),
                               np.random.randn(40, 40, 3)])


@pytest.fixture
def hdf5_dataset(tmpdir, scanpath_fixations, stimuli):
    stimuli.to_hdf5(os.path.join(str(tmpdir), 'stimuli.hdf5'))
    scanpath_fixations.to_hdf5(os.path.join(str(tmpdir), 'fixations.hdf5'))
    return str(tmpdir)


def test_load_dataset(hdf5_dataset, stimuli, scanpath_fixations):
    loaded_stimuli, loaded_fixations = dc.load_dataset_from_config({
        'stimuli': os.path.join(hdf5_dataset, 'stimuli.hdf5'),
        'fixations': os.path.join(hdf5_dataset, 'fixations.hdf5'),
    })

    assert len(loaded_stimuli) == len(stimuli)
    np.testing.assert_allclose(loaded_fixations.x, scanpath_fixations.x)


def test_load_dataset_with_filter(hdf5_dataset, stimuli, scanpath_fixations):
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
    assert np.all(loaded_fixations.scanpath_history_length < 2)


def test_apply_dataset_filter_config_filter_scanpaths_by_attribute_task(stimuli, scanpath_fixations):
    scanpaths = scanpath_fixations
    filter_config = {
        'type': 'filter_scanpaths_by_attribute',
        'parameters': {
            'attribute_name': 'task',
            'attribute_value': 0,
            'invert_match': False,
        }
    }
    filtered_stimuli, filtered_scanpaths = dc.apply_dataset_filter_config(stimuli, scanpaths, filter_config)
    inds = [0, 2]
    expected_scanpaths = scanpaths.filter_scanpaths(inds)
    assert_scanpath_fixations_equal(filtered_scanpaths, expected_scanpaths)
    assert_stimuli_equal(filtered_stimuli, stimuli)


def test_apply_dataset_filter_config_filter_scanpaths_by_attribute_multi_dim_attribute_invert_match(stimuli, scanpath_fixations):
    scanpaths = scanpath_fixations
    filter_config = {
        'type': 'filter_scanpaths_by_attribute',
        'parameters': {
            'attribute_name': 'multi_dim_attribute',
            'attribute_value': [0, 1],
            'invert_match': True,
        }
    }
    filtered_stimuli, filtered_scanpaths = dc.apply_dataset_filter_config(stimuli, scanpaths, filter_config)
    inds = [1, 2]
    expected_scanpaths = scanpaths.filter_scanpaths(inds)
    assert_scanpath_fixations_equal(filtered_scanpaths, expected_scanpaths)
    assert_stimuli_equal(filtered_stimuli, stimuli)


def test_apply_dataset_filter_config_filter_fixations_by_attribute_subject_invert_match(stimuli, scanpath_fixations):
    fixations = scanpath_fixations[:]
    filter_config = {
        'type': 'filter_fixations_by_attribute',
        'parameters': {
            'attribute_name': 'subject',
            'attribute_value': 0,
            'invert_match': True,
        }
    }
    filtered_stimuli, filtered_fixations = dc.apply_dataset_filter_config(stimuli, fixations, filter_config)
    inds = [3, 4, 5, 6, 7]
    expected_fixations = fixations[inds]
    assert_fixations_equal(filtered_fixations, expected_fixations)
    assert_stimuli_equal(filtered_stimuli, stimuli)


def test_apply_dataset_filter_config_filter_stimuli_by_attribute_dva(file_stimuli_with_attributes, scanpath_fixations):
    fixations = scanpath_fixations[:]
    filter_config = {
        'type': 'filter_stimuli_by_attribute',
        'parameters': {
            'attribute_name': 'dva',
            'attribute_value': 1,
            'invert_match': False,
        }
    }
    filtered_stimuli, filtered_fixations = dc.apply_dataset_filter_config(file_stimuli_with_attributes, fixations, filter_config)
    inds = [1]
    expected_stimuli, expected_fixations = create_subset(file_stimuli_with_attributes, fixations, inds)
    assert_fixations_equal(filtered_fixations, expected_fixations)
    assert_stimuli_equal(filtered_stimuli, expected_stimuli)


def test_apply_dataset_filter_config_filter_scanpaths_by_length_multiple_inputs(stimuli, scanpath_fixations):
    scanpaths = scanpath_fixations
    filter_config = {
        'type': 'filter_scanpaths_by_length',
        'parameters': {
            'intervals': [(1, 2), (2, 3)]
        }
    }
    filtered_stimuli, filtered_scanpaths = dc.apply_dataset_filter_config(stimuli, scanpaths, filter_config)
    inds = [1]
    expected_scanpaths = scanpaths.filter_scanpaths(inds)
    assert_scanpath_fixations_equal(filtered_scanpaths, expected_scanpaths)
    assert_stimuli_equal(filtered_stimuli, stimuli)


def test_apply_dataset_filter_config_filter_scanpaths_by_length_single_input(stimuli, scanpath_fixations):
    scanpaths = scanpath_fixations
    filter_config = {
        'type': 'filter_scanpaths_by_length',
        'parameters': {
            'intervals': [(3)]
        }
    }
    filtered_stimuli, filtered_scanpaths = dc.apply_dataset_filter_config(stimuli, scanpaths, filter_config)
    inds = [0, 2]
    expected_scanpaths = scanpaths.filter_scanpaths(inds)
    assert_scanpath_fixations_equal(filtered_scanpaths, expected_scanpaths)
    assert_stimuli_equal(filtered_stimuli, stimuli)
