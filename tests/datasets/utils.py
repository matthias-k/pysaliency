from pysaliency.datasets import ScanpathFixations
from pysaliency.datasets.scanpaths import Scanpaths


import numpy as np

from pysaliency.utils.variable_length_array import VariableLengthArray


def assert_variable_length_array_equal(array1, array2):
    assert isinstance(array1, VariableLengthArray)
    assert isinstance(array2, VariableLengthArray)
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


def compare_fixations_subset(f1, f2, f2_inds):
    np.testing.assert_allclose(f1.x, f2.x[f2_inds])
    np.testing.assert_allclose(f1.y, f2.y[f2_inds])
    np.testing.assert_allclose(f1.t, f2.t[f2_inds])
    np.testing.assert_allclose(f1.n, f2.n[f2_inds])
    np.testing.assert_allclose(f1.subject, f2.subject[f2_inds])

    assert f1.__attributes__ == f2.__attributes__
    for attribute in f1.__attributes__:
        if attribute == 'scanpath_index':
            continue
        np.testing.assert_array_equal(getattr(f1, attribute), getattr(f2, attribute)[f2_inds])


def assert_fixations_equal(f1, f2, crop_length=False):
    if crop_length:
        maximum_length = np.max(f2.scanpath_history_length)
    else:
        maximum_length = max(np.max(f1.scanpath_history_length), np.max(f2.scanpath_history_length))
    np.testing.assert_array_equal(f1.x, f2.x)
    np.testing.assert_array_equal(f1.y, f2.y)
    np.testing.assert_array_equal(f1.t, f2.t)
    np.testing.assert_array_equal(f1.n, f2.n)
    assert_variable_length_array_equal(f1.x_hist, f2.x_hist)
    assert_variable_length_array_equal(f1.y_hist, f2.y_hist)
    assert_variable_length_array_equal(f1.t_hist, f2.t_hist)

    f1_attributes = set(f1.__attributes__)
    f2_attributes = set(f2.__attributes__)

    assert set(f1_attributes) == set(f2_attributes)
    for attribute in f1.__attributes__:
        if attribute == 'scanpath_index':
            continue
        attribute1 = getattr(f1, attribute)
        attribute2 = getattr(f2, attribute)

        if isinstance(attribute1, VariableLengthArray):
            assert_variable_length_array_equal(attribute1, attribute2)
            continue
        elif attribute.endswith('_hist'):
            attribute1 = attribute1[:, :maximum_length]
            attribute2 = attribute2[:, :maximum_length]

        np.testing.assert_array_equal(attribute1, attribute2, err_msg=f'attributes not equal: {attribute}')


def assert_fixation_trains_equal(fixation_trains1, fixation_trains2):
    assert_variable_length_array_equal(fixation_trains1.train_xs, fixation_trains2.train_xs)
    assert_variable_length_array_equal(fixation_trains1.train_ys, fixation_trains2.train_ys)
    assert_variable_length_array_equal(fixation_trains1.train_ts, fixation_trains2.train_ts)

    np.testing.assert_array_equal(fixation_trains1.train_ns, fixation_trains2.train_ns)
    np.testing.assert_array_equal(fixation_trains1.train_subjects, fixation_trains2.train_subjects)
    np.testing.assert_array_equal(fixation_trains1.train_lengths, fixation_trains2.train_lengths)

    assert fixation_trains1.scanpath_attribute_mapping == fixation_trains2.scanpath_attribute_mapping

    assert fixation_trains1.scanpath_attributes.keys() == fixation_trains2.scanpath_attributes.keys()
    for attribute_name in fixation_trains1.scanpath_attributes.keys():
        np.testing.assert_array_equal(fixation_trains1.scanpath_attributes[attribute_name], fixation_trains2.scanpath_attributes[attribute_name])

    assert fixation_trains1.scanpath_fixation_attributes.keys() == fixation_trains2.scanpath_fixation_attributes.keys()
    for attribute_name in fixation_trains1.scanpath_fixation_attributes.keys():
        assert_variable_length_array_equal(fixation_trains1.scanpath_fixation_attributes[attribute_name], fixation_trains2.scanpath_fixation_attributes[attribute_name])

    assert_fixations_equal(fixation_trains1, fixation_trains2)


def assert_scanpath_fixations_equal(scanpath_fixations1: ScanpathFixations, scanpath_fixations2: ScanpathFixations):
    assert isinstance(scanpath_fixations1, ScanpathFixations)
    assert isinstance(scanpath_fixations2, ScanpathFixations)
    assert_scanpaths_equal(scanpath_fixations1.scanpaths, scanpath_fixations2.scanpaths)
    assert_fixations_equal(scanpath_fixations1, scanpath_fixations2)