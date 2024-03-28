from copy import deepcopy

import numpy as np
import pytest

import pysaliency
from pysaliency.datasets import Scanpaths
from pysaliency.utils.variable_length_array import VariableLengthArray


def assert_variable_length_array_equal(array1, array2):
    assert len(array1) == len(array2)

    for i in range(len(array1)):
        np.testing.assert_array_equal(array1[i], array2[i], err_msg=f'arrays not equal at index {i}')


def assert_scanpaths_equal(scanpaths1: Scanpaths, scanpaths2: Scanpaths, scanpaths2_inds=None):

    if scanpaths2_inds is None:
        scanpaths2_inds = slice(None)

    assert isinstance(scanpaths1, Scanpaths)
    assert isinstance(scanpaths2, Scanpaths)

    assert_variable_length_array_equal(scanpaths1.xs, scanpaths2.xs[scanpaths2_inds])
    assert_variable_length_array_equal(scanpaths1.ys, scanpaths2.ys[scanpaths2_inds])

    assert scanpaths1.scanpath_attributes.keys() == scanpaths2.scanpath_attributes.keys()
    for attribute_name in scanpaths1.scanpath_attributes.keys():
        np.testing.assert_array_equal(scanpaths1.scanpath_attributes[attribute_name], scanpaths2.scanpath_attributes[attribute_name][scanpaths2_inds])

    assert scanpaths1.fixation_attributes.keys() == scanpaths2.fixation_attributes.keys()
    for attribute_name in scanpaths1.fixation_attributes.keys():
        assert_variable_length_array_equal(scanpaths1.fixation_attributes[attribute_name], scanpaths2.fixation_attributes[attribute_name][scanpaths2_inds])

    assert scanpaths1.attribute_mapping == scanpaths2.attribute_mapping


def test_scanpaths():
    xs = np.array([[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
    ys = np.array([[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
    ns = np.array([0, 0, 1])
    lengths = np.array([3, 2, 3])
    scanpath_attributes = {'task': np.array([0, 1, 0])}
    fixation_attributes = {'attribute1': np.array([[1, 1, 2], [2, 2, np.nan], [0, 1, 3]]), 'attribute2': np.array([[3, 1.3, 5], [1, 42, np.nan], [0, -1, -3]])}
    attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    scanpaths = Scanpaths(xs, ys, ns, lengths, scanpath_attributes, fixation_attributes, attribute_mapping)

    assert isinstance(scanpaths.xs, VariableLengthArray)
    assert isinstance(scanpaths.ys, VariableLengthArray)
    assert isinstance(scanpaths.ns, np.ndarray)
    assert isinstance(scanpaths.lengths, np.ndarray)
    assert isinstance(scanpaths.scanpath_attributes, dict)
    assert isinstance(scanpaths.scanpath_attributes['task'], np.ndarray)
    assert isinstance(scanpaths.fixation_attributes, dict)
    assert isinstance(scanpaths.fixation_attributes['attribute1'], VariableLengthArray)
    assert isinstance(scanpaths.fixation_attributes['attribute2'], VariableLengthArray)
    assert isinstance(scanpaths.attribute_mapping, dict)

    np.testing.assert_array_equal(scanpaths.xs._data, xs)
    np.testing.assert_array_equal(scanpaths.ys._data, ys)
    np.testing.assert_array_equal(scanpaths.ns, ns)
    np.testing.assert_array_equal(scanpaths.lengths, lengths)
    np.testing.assert_array_equal(scanpaths.scanpath_attributes['task'], np.array([0, 1, 0]))
    np.testing.assert_array_equal(scanpaths.fixation_attributes['attribute1']._data, np.array([[1, 1, 2], [2, 2, np.nan], [0, 1, 3]]))
    np.testing.assert_array_equal(scanpaths.fixation_attributes['attribute2']._data, np.array([[3, 1.3, 5], [1, 42, np.nan], [0, -1, -3]]))
    assert scanpaths.attribute_mapping == {'attribute1': 'attr1', 'attribute2': 'attr2'}


def test_scanpaths_from_lists():
    xs = [[0, 1, 2], [2, 2], [1, 5, 3]]
    ys = [[10, 11, 12], [12, 12], [21, 25, 33]]
    ns = [0, 0, 1]
    expected_lengths = np.array([3, 2, 3])
    scanpath_attributes = {'task': [0, 1, 0]}
    fixation_attributes = {'attribute1': [[1, 1, 2], [2, 2], [0, 1, 3]], 'attribute2': [[3, 1.3, 5], [1, 42], [0, -1, -3]]}
    attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    scanpaths = Scanpaths(xs, ys, ns, lengths=None, scanpath_attributes=scanpath_attributes, fixation_attributes=fixation_attributes, attribute_mapping=attribute_mapping)

    np.testing.assert_array_equal(scanpaths.xs._data, np.array([[0, 1, 2], [2, 2, np.nan], [1, 5, 3]]))
    np.testing.assert_array_equal(scanpaths.ys._data, np.array([[10, 11, 12], [12, 12, np.nan], [21, 25, 33]]))
    np.testing.assert_array_equal(scanpaths.ns, ns)
    np.testing.assert_array_equal(scanpaths.lengths, expected_lengths)
    np.testing.assert_array_equal(scanpaths.scanpath_attributes['task'], np.array([0, 1, 0]))
    np.testing.assert_array_equal(scanpaths.fixation_attributes['attribute1']._data, np.array([[1, 1, 2], [2, 2, np.nan], [0, 1, 3]]))
    np.testing.assert_array_equal(scanpaths.fixation_attributes['attribute2']._data, np.array([[3, 1.3, 5], [1, 42, np.nan], [0, -1, -3]]))
    assert scanpaths.attribute_mapping == {'attribute1': 'attr1', 'attribute2': 'attr2'}


def test_scanpaths_init_inconsistent_lengths():
    xs = np.array([[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
    ys = np.array([[10, 11, 12], [12, 12, np.nan]])  # too short, should fail
    ns = np.array([0, 0, 1])
    lengths = np.array([3, 2, 3])
    scanpath_attributes = {'task': np.array([0, 1, 0])}
    fixation_attributes = {'attribute1': np.array([[1, 1, 2], [2, 2, np.nan], [0, 1, 3]]), 'attribute2': np.array([[3, 1.3, 5], [1, 42, np.nan], [0, -1, -3]])}
    attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    with pytest.raises(ValueError):
        Scanpaths(xs, ys, ns, lengths, scanpath_attributes, fixation_attributes, attribute_mapping)

def test_scanpaths_init_invalid_scanpath_attributes():
    xs = np.array([[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
    ys = np.array([[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
    ns = np.array([0, 0, 1])
    lengths = np.array([3, 2, 3])
    scanpath_attributes = {'invalid_attribute': np.array([1, 2]), 'attribute2': np.array([4, 5, 6])}  # Invalid attribute length
    scanpath_fixation_attributes = {'fixation_attribute1': np.array([[1, 2], [3, 4], [5, 6]]), 'fixation_attribute2': np.array([[7, 8], [9, 10], [11, 12]])}
    scanpath_attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    with pytest.raises(ValueError):
        Scanpaths(xs, ys, ns, lengths, scanpath_attributes, scanpath_fixation_attributes, scanpath_attribute_mapping)

def test_scanpaths_init_invalid_scanpath_fixation_attributes():
    xs = np.array([[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
    ys = np.array([[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
    ns = np.array([0, 0, 1])
    lengths = np.array([3, 2, 3])
    scanpath_attributes = {'attribute1': np.array([1, 2, 3]), 'attribute2': np.array([4, 5, 6])}
    scanpath_fixation_attributes = {'valid_fixation_attribute': np.array([[1, 2], [3, 4], [5, 6]]), 'invalid_fixation_attribute': np.array([[7, 8], [9, 10]])}  # Invalid fixation attribute length
    scanpath_attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    with pytest.raises(ValueError):
        Scanpaths(xs, ys, ns, lengths, scanpath_attributes, scanpath_fixation_attributes, scanpath_attribute_mapping)

def test_scanpaths_init_invalid_scanpath_fixation_attributes_dimensions():
    xs = np.array([[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
    ys = np.array([[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
    ns = np.array([0, 0, 1])
    lengths = np.array([3, 2, 3])
    scanpath_attributes = {'attribute1': np.array([1, 2, 3]), 'attribute2': np.array([4, 5, 6])}
    scanpath_fixation_attributes = {'fixation_attribute1': np.array([1, 2, 3]), 'fixation_attribute2': np.array([[7, 8], [9, 10], [11, 12]])}  # Invalid fixation attribute dimensions
    scanpath_attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    with pytest.raises(ValueError):
        Scanpaths(xs, ys, ns, lengths, scanpath_attributes, scanpath_fixation_attributes, scanpath_attribute_mapping)

def test_scanpaths_init_invalid_scanpath_lengths():

    data = {
        'xs': [[0, 1, 2], [2, 2], [1, 5, 3]],
        'ys': [[10, 11, 12], [12, 12], [21, 25, 33]],
        'ns': [0, 0, 1],
        'scanpath_attributes': {'task': [0, 1, 0]},
        'fixation_attributes': {'attribute1': [[1, 1, 2], [2, 2], [0, 1, 3]], 'attribute2': [[3, 1.3, 5], [1, 42], [0, -1, -3]]},
        'attribute_mapping': {'attribute1': 'attr1', 'attribute2': 'attr2'},
    }

    # make sure original data works
    Scanpaths(**data)


    for scanpath_attribute in ['xs', 'ys']:
        data_copy = deepcopy(data)
        data_copy[scanpath_attribute][-1].append(4)
        with pytest.raises(ValueError):
            Scanpaths(**data_copy)

    for scanpath_attribute in data['fixation_attributes'].keys():
        data_copy = deepcopy(data)
        data_copy['fixation_attributes'][scanpath_attribute][-1].append(4)
        with pytest.raises(ValueError):
            Scanpaths(**data_copy)



@pytest.mark.parametrize('inds', [
    slice(None, 2),
    slice(1, None),
    [0, 1],
    [1, 2],
    [0, 2],
    [2, 1],
    [False, True, True],
])
def test_scanpaths_slicing(inds):
    xs = [[0, 1, 2], [2, 2], [1, 5, 3]]
    ys = [[10, 11, 12], [12, 12], [21, 25, 33]]
    ns = [0, 0, 1]
    scanpath_attributes = {'task': [0, 1, 0]}
    fixation_attributes = {'attribute1': [[1, 1, 2], [2, 2], [0, 1, 3]], 'attribute2': [[3, 1.3, 5], [1, 42], [0, -1, -3]]}
    attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    scanpaths = Scanpaths(xs, ys, ns, lengths=None, scanpath_attributes=scanpath_attributes, fixation_attributes=fixation_attributes, attribute_mapping=attribute_mapping)

    sliced_scanpaths = scanpaths[inds]
    assert_scanpaths_equal(sliced_scanpaths, scanpaths, inds)


def test_write_read_scanpaths_pathlib(tmp_path):
    filename = tmp_path / 'scanpaths.hdf5'

    xs = [[0, 1, 2], [2, 2], [1, 5, 3]]
    ys = [[10, 11, 12], [12, 12], [21, 25, 33]]
    ns = [0, 0, 1]
    scanpath_attributes = {'task': [0, 1, 0]}
    fixation_attributes = {'attribute1': [[1, 1, 2], [2, 2], [0, 1, 3]], 'attribute2': [[3, 1.3, 5], [1, 42], [0, -1, -3]]}
    attribute_mapping = {'attribute1': 'attr1', 'attribute2': 'attr2'}

    scanpaths = Scanpaths(xs, ys, ns, lengths=None, scanpath_attributes=scanpath_attributes, fixation_attributes=fixation_attributes, attribute_mapping=attribute_mapping)

    scanpaths.to_hdf5(filename)

    # test loading via class method

    new_scanpaths = Scanpaths.read_hdf5(filename)

    assert scanpaths is not new_scanpaths  # make sure there is no sophisticated caching...
    assert_scanpaths_equal(scanpaths, new_scanpaths)

    # test loading via pysaliency

    new_scanpaths = pysaliency.read_hdf5(filename)

    assert scanpaths is not new_scanpaths # make sure there is no sophisticated caching...
    assert_scanpaths_equal(scanpaths, new_scanpaths)