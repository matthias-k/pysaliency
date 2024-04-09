import os.path
import pickle

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from imageio import imwrite
from test_helpers import TestWithData

import pysaliency
from pysaliency.datasets import Fixations, FixationTrains, ScanpathFixations, scanpaths_from_fixations
from pysaliency.datasets.scanpaths import Scanpaths
from pysaliency.utils.variable_length_array import VariableLengthArray


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


def assert_variable_length_array_equal(array1, array2):
    assert isinstance(array1, VariableLengthArray)
    assert isinstance(array2, VariableLengthArray)
    assert len(array1) == len(array2)

    for i in range(len(array1)):
        np.testing.assert_array_equal(array1[i], array2[i], err_msg=f'arrays not equal at index {i}')


def compare_fixations_subset(f1, f2, f2_inds):
    np.testing.assert_allclose(f1.x, f2.x[f2_inds])
    np.testing.assert_allclose(f1.y, f2.y[f2_inds])
    np.testing.assert_allclose(f1.t, f2.t[f2_inds])
    np.testing.assert_allclose(f1.n, f2.n[f2_inds])
    np.testing.assert_allclose(f1.subjects, f2.subjects[f2_inds])

    assert f1.__attributes__ == f2.__attributes__
    for attribute in f1.__attributes__:
        if attribute == 'scanpath_index':
            continue
        np.testing.assert_array_equal(getattr(f1, attribute), getattr(f2, attribute)[f2_inds])


def assert_fixations_equal(f1, f2, crop_length=False):
    if crop_length:
        maximum_length = np.max(f2.lengths)
    else:
        maximum_length = max(np.max(f1.lengths), np.max(f2.lengths))
    np.testing.assert_array_equal(f1.x, f2.x)
    np.testing.assert_array_equal(f1.y, f2.y)
    np.testing.assert_array_equal(f1.t, f2.t)
    assert_variable_length_array_equal(f1.x_hist, f2.x_hist)
    assert_variable_length_array_equal(f1.y_hist, f2.y_hist)
    assert_variable_length_array_equal(f1.t_hist, f2.t_hist)

    assert set(f1.__attributes__) == set(f2.__attributes__)
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


class TestFixations(TestWithData):
    def test_from_fixations(self):
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
        some_attribute = np.arange(len(sum(xs_trains, [])))
        # Create Fixations
        f = pysaliency.FixationTrains.from_fixation_trains(
            xs_trains,
            ys_trains,
            ts_trains,
            ns,
            subjects,
            attributes={'some_attribute': some_attribute},
            scanpath_attributes={'task': tasks},
        )

        # Test fixation trains

        assert_variable_length_array_equal(f.train_xs, VariableLengthArray(xs_trains))
        assert_variable_length_array_equal(f.train_ys, VariableLengthArray(ys_trains))
        assert_variable_length_array_equal(f.train_ts, VariableLengthArray(ts_trains))

        np.testing.assert_allclose(f.train_ns, ns)
        np.testing.assert_allclose(f.train_subjects, subjects)

        # Test conditional fixations
        np.testing.assert_allclose(f.x, [0, 1, 2, 2, 2, 1, 5, 3])
        np.testing.assert_allclose(f.y, [10, 11, 12, 12, 12, 21, 25, 33])
        np.testing.assert_allclose(f.t, [0, 200, 600, 100, 400, 50, 500, 900])
        np.testing.assert_allclose(f.n, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(f.subjects, [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(f.lengths, [0, 1, 2, 0, 1, 0, 1, 2])

        assert_variable_length_array_equal(
            f.x_hist,
            VariableLengthArray([
                [],
                [0],
                [0, 1],
                [],
                [2],
                [],
                [1],
                [1, 5]
            ])
        )

    def test_filter(self):
        xs_trains = []
        ys_trains = []
        ts_trains = []
        ns = []
        subjects = []
        for n in range(1000):
            size = np.random.randint(10)
            xs_trains.append(np.random.randn(size))
            ys_trains.append(np.random.randn(size))
            ts_trains.append(np.cumsum(np.square(np.random.randn(size))))
            ns.append(np.random.randint(20))
            subjects.append(np.random.randint(20))
        f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)
        # First order filtering
        inds = f.n == 10
        _f = f.filter(inds)
        self.assertNotIsInstance(_f, pysaliency.FixationTrains)
        compare_fixations_subset(_f, f, inds)

        # second order filtering
        inds = np.nonzero(f.n == 10)[0]
        _f = f.filter(inds)
        inds2 = np.nonzero(_f.subjects == 0)[0]
        __f = _f.filter(inds2)
        cum_inds = inds[inds2]
        compare_fixations_subset(__f, f, cum_inds)

    def test_filter_trains(self):
        xs_trains = []
        ys_trains = []
        ts_trains = []
        ns = []
        subjects = []
        for n in range(1000):
            size = np.random.randint(10)
            xs_trains.append(np.random.randn(size))
            ys_trains.append(np.random.randn(size))
            ts_trains.append(np.cumsum(np.square(np.random.randn(size))))
            ns.append(np.random.randint(20))
            subjects.append(np.random.randint(20))

        f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)
        # First order filtering
        inds = f.train_ns == 10
        _f = f.filter_fixation_trains(inds)
        self.assertIsInstance(_f, pysaliency.FixationTrains)
        equivalent_indices = f.n == 10
        compare_fixations_subset(_f, f, equivalent_indices)

        ## second order filtering
        # inds = np.nonzero(f.n == 10)[0]
        # _f = f.filter(inds)
        # inds2 = np.nonzero(_f.subjects == 0)[0]
        # __f = _f.filter(inds2)
        # cum_inds = inds[inds2]
        # compare_fixations_subset(__f, f, cum_inds)

    def test_save_and_load(self):
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
        # Create /Fixations
        f = pysaliency.FixationTrains.from_fixation_trains(xs_trains, ys_trains, ts_trains, ns, subjects)

        filename = os.path.join(self.data_path, 'fixation.pydat')
        with open(filename, 'wb') as out_file:
            pickle.dump(f, out_file)

        with open(filename, 'rb') as in_file:
            f = pickle.load(in_file)
        # Test fixation trains

        assert_variable_length_array_equal(f.train_xs, VariableLengthArray(xs_trains))
        assert_variable_length_array_equal(f.train_ys, VariableLengthArray(ys_trains))
        assert_variable_length_array_equal(f.train_ts, VariableLengthArray(ts_trains))

        np.testing.assert_allclose(f.train_ns, [0, 0, 1])
        np.testing.assert_allclose(f.train_subjects, [0, 1, 1])

        # Test conditional fixations
        np.testing.assert_allclose(f.x, [0, 1, 2, 2, 2, 1, 5, 3])
        np.testing.assert_allclose(f.y, [10, 11, 12, 12, 12, 21, 25, 33])
        np.testing.assert_allclose(f.t, [0, 200, 600, 100, 400, 50, 500, 900])
        np.testing.assert_allclose(f.n, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(f.subjects, [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(f.lengths, [0, 1, 2, 0, 1, 0, 1, 2])
        np.testing.assert_allclose(f.x_hist._data,
                                   [[np.nan, np.nan],
                                   [0, np.nan],
                                   [0, 1],
                                   [np.nan, np.nan],
                                   [2, np.nan],
                                   [np.nan, np.nan],
                                   [1, np.nan],
                                   [1, 5]])



@pytest.fixture
def scanpath_fixations() -> ScanpathFixations:
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
    multi_dim_attribute = [[0.0, 1],[2, 3], [4, 5.5]]
    durations_train = [
        [42, 25, 100],
        [99, 98],
        [200, 150, 120]
    ]
    scanpaths = Scanpaths(
        xs=xs_trains,
        ys=ys_trains,
        n=ns,
        scanpath_attributes={
            'task': tasks,
            'multi_dim_attribute': multi_dim_attribute,
            'subject': subjects,
        },
        fixation_attributes={'durations': durations_train, 'ts': ts_trains},
        attribute_mapping={'durations': 'duration', 'ts': 't'},
    )

    return ScanpathFixations(scanpaths=scanpaths)


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
    tasks = [0, 1, 0]
    multi_dim_attribute = [[0.0, 1],[2, 3], [4, 5.5]]
    durations_train = [
        [42, 25, 100],
        [99, 98],
        [200, 150, 120]
    ]
    some_attribute = np.arange(len(sum(xs_trains, [])))
    return pysaliency.FixationTrains.from_fixation_trains(
        xs_trains,
        ys_trains,
        ts_trains,
        ns,
        subjects,
        attributes={'some_attribute': some_attribute},
        scanpath_attributes={
            'task': tasks,
            'multi_dim_attribute': multi_dim_attribute
        },
        scanpath_fixation_attributes={'durations': durations_train},
        scanpath_attribute_mapping={'durations': 'duration'},
    )


def test_copy_scanpath_fixations(scanpath_fixations):
    copied_scanpath_fixations = scanpath_fixations.copy()
    assert_scanpath_fixations_equal(copied_scanpath_fixations, scanpath_fixations)
    assert copied_scanpath_fixations is not scanpath_fixations


def test_copy_fixation_trains(fixation_trains):
    copied_fixation_trains = fixation_trains.copy()
    assert_fixation_trains_equal(copied_fixation_trains, fixation_trains)


def test_copy_fixations(fixation_trains):
    fixations = fixation_trains[:]
    copied_fixations = fixations.copy()
    assert_fixations_equal(copied_fixations, fixations)


def test_write_read_scanpath_fixations_pathlib(tmp_path, scanpath_fixations):
    filename = tmp_path / 'scanpath_fixations.hdf5'
    scanpath_fixations.to_hdf5(filename)

    new_scanpath_fixations = pysaliency.read_hdf5(filename)

    # make sure there is no sophisticated caching...
    assert scanpath_fixations is not new_scanpath_fixations
    assert_scanpath_fixations_equal(scanpath_fixations, new_scanpath_fixations)


def test_write_read_fixation_trains_pathlib(tmp_path, fixation_trains):
    filename = tmp_path / 'fixation_trains.hdf5'
    fixation_trains.to_hdf5(filename)

    new_fixation_trains = pysaliency.read_hdf5(filename)

    # make sure there is no sophisticated caching...
    assert fixation_trains is not new_fixation_trains
    assert_fixation_trains_equal(fixation_trains, new_fixation_trains)


def test_write_read_fixation_trains(tmp_path, fixation_trains):
    filename = tmp_path / 'fixation_trains.hdf5'
    fixation_trains.to_hdf5(str(filename))

    new_fixation_trains = pysaliency.read_hdf5(str(filename))

    # make sure there is no sophisticated caching...
    assert fixation_trains is not new_fixation_trains
    assert_fixation_trains_equal(fixation_trains, new_fixation_trains)


def test_scanpath_lengths(fixation_trains):
    np.testing.assert_array_equal(fixation_trains.train_lengths, [3, 2, 3])


def test_scanpath_fixations_scanpath_attributes(scanpath_fixations):
    assert "task" in scanpath_fixations.scanpaths.scanpath_attributes
    assert "task" in scanpath_fixations.__attributes__

    np.testing.assert_array_equal(scanpath_fixations.scanpaths.scanpath_attributes['multi_dim_attribute'][0], [0, 1])
    np.testing.assert_array_equal(scanpath_fixations.multi_dim_attribute[2], [0, 1])


def test_fixation_trains_scanpath_attributes(fixation_trains):
    assert "task" in fixation_trains.scanpath_attributes
    assert "task" in fixation_trains.__attributes__

    np.testing.assert_array_equal(fixation_trains.scanpath_attributes['multi_dim_attribute'][0], [0, 1])
    np.testing.assert_array_equal(fixation_trains.multi_dim_attribute[2], [0, 1])


def test_scanpath_fixations_scanpath_fixation_attributes(scanpath_fixations):
    # test attribute itself
    assert "durations" in scanpath_fixations.scanpaths.fixation_attributes
    assert isinstance(scanpath_fixations.scanpaths.fixation_attributes['durations'], VariableLengthArray)
    np.testing.assert_array_equal(scanpath_fixations.scanpaths.fixation_attributes['durations'][0], [42, 25, 100])
    np.testing.assert_array_equal(scanpath_fixations.scanpaths.fixation_attributes['durations'][1], [99, 98])
    np.testing.assert_array_equal(scanpath_fixations.scanpaths.fixation_attributes['durations'][2], [200, 150, 120])

    # test derived fixation attribute
    assert "duration" in scanpath_fixations.__attributes__
    np.testing.assert_array_equal(scanpath_fixations.duration, np.array([
            42, 25, 100,
            99, 98,
            200, 150, 120
        ]))

    # test derived history attribute
    assert "duration_hist" in scanpath_fixations.__attributes__
    assert isinstance(scanpath_fixations.duration_hist, VariableLengthArray)
    np.testing.assert_array_equal(scanpath_fixations.duration_hist[0], [])
    np.testing.assert_array_equal(scanpath_fixations.duration_hist[1], [42])
    np.testing.assert_array_equal(scanpath_fixations.duration_hist[2], [42, 25])

    np.testing.assert_array_equal(scanpath_fixations.duration_hist[3], [])
    np.testing.assert_array_equal(scanpath_fixations.duration_hist[4], [99])

    np.testing.assert_array_equal(scanpath_fixations.duration_hist[5], [])
    np.testing.assert_array_equal(scanpath_fixations.duration_hist[6], [200])
    np.testing.assert_array_equal(scanpath_fixations.duration_hist[7], [200, 150])


def test_fixation_trains_scanpath_fixation_attributes(fixation_trains):
    # test attribute itself
    assert "durations" in fixation_trains.scanpath_fixation_attributes
    assert isinstance(fixation_trains.scanpath_fixation_attributes['durations'], VariableLengthArray)
    np.testing.assert_array_equal(fixation_trains.scanpath_fixation_attributes['durations'][0], [42, 25, 100])
    np.testing.assert_array_equal(fixation_trains.scanpath_fixation_attributes['durations'][1], [99, 98])
    np.testing.assert_array_equal(fixation_trains.scanpath_fixation_attributes['durations'][2], [200, 150, 120])

    # test derived fixation attribute
    assert "duration" in fixation_trains.__attributes__
    np.testing.assert_array_equal(fixation_trains.duration, np.array([
            42, 25, 100,
            99, 98,
            200, 150, 120
        ]))

    # test derived history attribute
    assert "duration_hist" in fixation_trains.__attributes__
    assert isinstance(fixation_trains.duration_hist, VariableLengthArray)
    np.testing.assert_array_equal(fixation_trains.duration_hist[0], [])
    np.testing.assert_array_equal(fixation_trains.duration_hist[1], [42])
    np.testing.assert_array_equal(fixation_trains.duration_hist[2], [42, 25])

    np.testing.assert_array_equal(fixation_trains.duration_hist[3], [])
    np.testing.assert_array_equal(fixation_trains.duration_hist[4], [99])

    np.testing.assert_array_equal(fixation_trains.duration_hist[5], [])
    np.testing.assert_array_equal(fixation_trains.duration_hist[6], [200])
    np.testing.assert_array_equal(fixation_trains.duration_hist[7], [200, 150])



@pytest.mark.parametrize('scanpath_indices,fixation_indices', [
    ([0, 2], [0, 1, 2, 5, 6, 7]),
    ([1, 2], [3, 4, 5, 6, 7]),
    ([2], [5, 6, 7]),
])
def test_scanpath_fixations_filter_scanpaths(scanpath_fixations, scanpath_indices, fixation_indices):
    sub_fixations = scanpath_fixations.filter_scanpaths(scanpath_indices)

    assert_scanpaths_equal(sub_fixations.scanpaths, scanpath_fixations.scanpaths[scanpath_indices])

    assert_fixations_equal(sub_fixations, scanpath_fixations[fixation_indices])


@pytest.mark.parametrize('scanpath_indices,fixation_indices', [
    ([0, 2], [0, 1, 2, 5, 6, 7]),
    ([1, 2], [3, 4, 5, 6, 7]),
    ([2], [5, 6, 7]),
])
def test_filter_fixation_trains(fixation_trains, scanpath_indices, fixation_indices):
    sub_fixations = fixation_trains.filter_fixation_trains(scanpath_indices)

    assert_variable_length_array_equal(
        sub_fixations.train_xs,
        fixation_trains.train_xs[scanpath_indices]
    )

    assert_variable_length_array_equal(
        sub_fixations.train_ys,
        fixation_trains.train_ys[scanpath_indices]
    )

    assert_variable_length_array_equal(
        sub_fixations.train_ts,
        fixation_trains.train_ts[scanpath_indices]
    )

    np.testing.assert_array_equal(
        sub_fixations.train_ns,
        fixation_trains.train_ns[scanpath_indices]
    )

    np.testing.assert_array_equal(
        sub_fixations.some_attribute,
        fixation_trains.some_attribute[fixation_indices]
    )

    np.testing.assert_array_equal(
        sub_fixations.scanpath_attributes['task'],
        fixation_trains.scanpath_attributes['task'][scanpath_indices]
    )

    assert_variable_length_array_equal(
        sub_fixations.scanpath_fixation_attributes['durations'],
        fixation_trains.scanpath_fixation_attributes['durations'][scanpath_indices]
    )

    assert_fixations_equal(sub_fixations, fixation_trains[fixation_indices])


def test_read_hdf5_caching(fixation_trains, tmp_path):
    filename = tmp_path / 'fixations.hdf5'
    fixation_trains.to_hdf5(str(filename))

    new_fixations = pysaliency.read_hdf5(str(filename))

    assert new_fixations is not fixation_trains

    new_fixations2 = pysaliency.read_hdf5(str(filename))
    assert new_fixations2 is new_fixations, "objects should not be read into memory multiple times"


def test_fixations_save_load(tmp_path, fixation_trains):
    fixations = fixation_trains[:-1]

    assert isinstance(fixations, Fixations)

    filename = tmp_path / 'fixations.hdf5'
    fixations.to_hdf5(filename)
    new_fixations = pysaliency.read_hdf5(filename)

    assert_fixations_equal(fixations, new_fixations)


def test_concatenate_fixations(fixation_trains):
    fixations = fixation_trains[:]
    new_fixations = pysaliency.Fixations.concatenate((fixations, fixations))
    assert isinstance(new_fixations, pysaliency.Fixations)
    np.testing.assert_allclose(
        new_fixations.x,
        np.concatenate((fixation_trains.x, fixation_trains.x))
    )

    np.testing.assert_allclose(
        new_fixations.n,
        np.concatenate((fixation_trains.n, fixation_trains.n))
    )

    assert new_fixations.__attributes__ == ['subjects', 'duration', 'duration_hist', 'multi_dim_attribute', 'scanpath_index', 'some_attribute', 'task']

    np.testing.assert_allclose(
        new_fixations.some_attribute,
        np.concatenate((fixation_trains.some_attribute, fixation_trains.some_attribute))
    )


def test_concatenate_scanpath_fixations(scanpath_fixations):
    new_scanpath_fixations = pysaliency.ScanpathFixations.concatenate((scanpath_fixations, scanpath_fixations))
    assert isinstance(new_scanpath_fixations, pysaliency.ScanpathFixations)
    np.testing.assert_allclose(
        new_scanpath_fixations.x,
        np.concatenate((scanpath_fixations.x, scanpath_fixations.x))
    )

    np.testing.assert_allclose(
        new_scanpath_fixations.n,
        np.concatenate((scanpath_fixations.n, scanpath_fixations.n))
    )

    assert set(new_scanpath_fixations.__attributes__) == {'subjects', 'duration', 'duration_hist', 'multi_dim_attribute', 'scanpath_index', 'task'}


def test_concatenate_scanpath_fixations_partial_attributes(scanpath_fixations):
    scanpath_fixations2 = scanpath_fixations.copy()

    del scanpath_fixations2.scanpaths.scanpath_attributes['task']
    delattr(scanpath_fixations2, 'task')
    scanpath_fixations2.auto_attributes.remove('task')
    scanpath_fixations2.__attributes__.remove('task')

    new_fixation_trains = pysaliency.ScanpathFixations.concatenate((scanpath_fixations, scanpath_fixations2))
    assert isinstance(new_fixation_trains, pysaliency.ScanpathFixations)
    np.testing.assert_allclose(
        new_fixation_trains.x,
        np.concatenate((scanpath_fixations.x, scanpath_fixations2.x))
    )

    np.testing.assert_allclose(
        new_fixation_trains.n,
        np.concatenate((scanpath_fixations.n, scanpath_fixations2.n))
    )

    assert set(new_fixation_trains.__attributes__) == {'subjects', 'duration', 'duration_hist', 'multi_dim_attribute', 'scanpath_index'}


def test_concatenate_fixation_trains_partial_attributes(fixation_trains):
    fixation_trains2 = fixation_trains.copy()

    del fixation_trains2.scanpaths.scanpath_attributes['task']
    delattr(fixation_trains2, 'task')
    fixation_trains2.auto_attributes.remove('task')
    fixation_trains2.__attributes__.remove('task')

    new_fixation_trains = pysaliency.FixationTrains.concatenate((fixation_trains, fixation_trains2))
    assert isinstance(new_fixation_trains, pysaliency.Fixations)
    np.testing.assert_allclose(
        new_fixation_trains.x,
        np.concatenate((fixation_trains.x, fixation_trains2.x))
    )

    np.testing.assert_allclose(
        new_fixation_trains.n,
        np.concatenate((fixation_trains.n, fixation_trains2.n))
    )

    assert set(new_fixation_trains.__attributes__) == {'subjects', 'duration', 'duration_hist', 'multi_dim_attribute', 'scanpath_index', 'some_attribute'}

    np.testing.assert_allclose(
        new_fixation_trains.some_attribute,
        np.concatenate((fixation_trains.some_attribute, fixation_trains2.some_attribute))
    )


@given(st.lists(elements=st.integers(min_value=0, max_value=7), min_size=1))
def test_fixation_trains_from_fixations(fixation_indices):
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
    #some_attribute = np.arange(len(sum(xs_trains, [])))
    some_attribute = [
        [0, 1, 3],
        [6, 10],
        [15, 21, 28]
    ]
    fixation_trains = pysaliency.FixationTrains.from_fixation_trains(
        xs_trains,
        ys_trains,
        ts_trains,
        ns,
        subjects,
        scanpath_fixation_attributes={'some_attribute': some_attribute},
        scanpath_attributes={'task': tasks},
    )

    sub_fixations = fixation_trains[fixation_indices]
    new_fixation_trains, new_indices = scanpaths_from_fixations(sub_fixations)
    new_sub_fixations = new_fixation_trains[new_indices]

    assert_fixations_equal(sub_fixations, new_sub_fixations, crop_length=True)