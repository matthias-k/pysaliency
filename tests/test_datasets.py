from __future__ import absolute_import, print_function, division

import unittest
import os.path
import dill
import pickle
import pytest

import numpy as np
from imageio import imwrite

from hypothesis import given, strategies as st

import pysaliency
from pysaliency.datasets import FixationTrains, Fixations, scanpaths_from_fixations
from test_helpers import TestWithData


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


def compare_fixations(f1, f2, crop_length=False):
    if crop_length:
        maximum_length = np.max(f2.lengths)
    else:
        maximum_length = max(np.max(f1.lengths), np.max(f2.lengths))
    np.testing.assert_array_equal(f1.x, f2.x)
    np.testing.assert_array_equal(f1.y, f2.y)
    np.testing.assert_array_equal(f1.t, f2.t)
    np.testing.assert_array_equal(f1.x_hist[:, :maximum_length], f2.x_hist)
    np.testing.assert_array_equal(f1.y_hist[:, :maximum_length], f2.y_hist)
    np.testing.assert_array_equal(f1.t_hist[:, :maximum_length], f2.t_hist)

    assert set(f1.__attributes__) == set(f2.__attributes__)
    for attribute in f1.__attributes__:
        if attribute == 'scanpath_index':
            continue
        attribute1 = getattr(f1, attribute)
        attribute2 = getattr(f2, attribute)

        if attribute.endswith('_hist'):
            attribute1 = attribute1[:, :maximum_length]

        np.testing.assert_array_equal(attribute1, attribute2, err_msg=f'attributes not equal: {attribute}')


def compare_scanpaths(scanpaths1, scanpaths2):
    np.testing.assert_array_equal(scanpaths1.train_xs, scanpaths2.train_xs)
    np.testing.assert_array_equal(scanpaths1.train_ys, scanpaths2.train_ys)
    np.testing.assert_array_equal(scanpaths1.train_xs, scanpaths2.train_xs)
    np.testing.assert_array_equal(scanpaths1.train_ns, scanpaths2.train_ns)
    np.testing.assert_array_equal(scanpaths1.train_subjects, scanpaths2.train_subjects)
    np.testing.assert_array_equal(scanpaths1.train_lengths, scanpaths2.train_lengths)

    assert scanpaths1.scanpath_attribute_mapping == scanpaths2.scanpath_attribute_mapping

    assert scanpaths1.scanpath_attributes.keys() == scanpaths2.scanpath_attributes.keys()
    for attribute_name in scanpaths1.scanpath_attributes.keys():
        np.testing.assert_array_equal(scanpaths1.scanpath_attributes[attribute_name], scanpaths2.scanpath_attributes[attribute_name])

    assert scanpaths1.scanpath_fixation_attributes.keys() == scanpaths2.scanpath_fixation_attributes.keys()
    for attribute_name in scanpaths1.scanpath_fixation_attributes.keys():
        np.testing.assert_array_equal(scanpaths1.scanpath_fixation_attributes[attribute_name], scanpaths2.scanpath_fixation_attributes[attribute_name])

    compare_fixations(scanpaths1, scanpaths2)



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
        np.testing.assert_allclose(f.train_xs, [[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
        np.testing.assert_allclose(f.train_ys, [[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
        np.testing.assert_allclose(f.train_ts, [[0, 200, 600], [100, 400, np.nan], [50, 500, 900]])
        np.testing.assert_allclose(f.train_ns, [0, 0, 1])
        np.testing.assert_allclose(f.train_subjects, [0, 1, 1])

        # Test conditional fixations
        np.testing.assert_allclose(f.x, [0, 1, 2, 2, 2, 1, 5, 3])
        np.testing.assert_allclose(f.y, [10, 11, 12, 12, 12, 21, 25, 33])
        np.testing.assert_allclose(f.t, [0, 200, 600, 100, 400, 50, 500, 900])
        np.testing.assert_allclose(f.n, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(f.subjects, [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(f.lengths, [0, 1, 2, 0, 1, 0, 1, 2])
        np.testing.assert_allclose(f.x_hist, [[np.nan, np.nan],
                                              [0, np.nan],
                                              [0, 1],
                                              [np.nan, np.nan],
                                              [2, np.nan],
                                              [np.nan, np.nan],
                                              [1, np.nan],
                                              [1, 5]])

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
        np.testing.assert_allclose(f.train_xs, [[0, 1, 2], [2, 2, np.nan], [1, 5, 3]])
        np.testing.assert_allclose(f.train_ys, [[10, 11, 12], [12, 12, np.nan], [21, 25, 33]])
        np.testing.assert_allclose(f.train_ts, [[0, 200, 600], [100, 400, np.nan], [50, 500, 900]])
        np.testing.assert_allclose(f.train_ns, [0, 0, 1])
        np.testing.assert_allclose(f.train_subjects, [0, 1, 1])

        # Test conditional fixations
        np.testing.assert_allclose(f.x, [0, 1, 2, 2, 2, 1, 5, 3])
        np.testing.assert_allclose(f.y, [10, 11, 12, 12, 12, 21, 25, 33])
        np.testing.assert_allclose(f.t, [0, 200, 600, 100, 400, 50, 500, 900])
        np.testing.assert_allclose(f.n, [0, 0, 0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(f.subjects, [0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_allclose(f.lengths, [0, 1, 2, 0, 1, 0, 1, 2])
        np.testing.assert_allclose(f.x_hist, [[np.nan, np.nan],
                                              [0, np.nan],
                                              [0, 1],
                                              [np.nan, np.nan],
                                              [2, np.nan],
                                              [np.nan, np.nan],
                                              [1, np.nan],
                                              [1, 5]])


class TestStimuli(TestWithData):
    def test_stimuli(self):
        img1 = np.random.randn(100, 200, 3)
        img2 = np.random.randn(50, 150)
        stimuli = pysaliency.Stimuli([img1, img2])

        self.assertEqual(stimuli.stimuli, [img1, img2])
        self.assertEqual(stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(list(stimuli.sizes), [(100, 200), (50, 150)])
        self.assertEqual(stimuli.stimulus_ids[1], pysaliency.datasets.get_image_hash(img2))
        np.testing.assert_allclose(stimuli.stimulus_objects[1].stimulus_data, img2)
        self.assertEqual(stimuli.stimulus_objects[1].stimulus_id, stimuli.stimulus_ids[1])

        new_stimuli = self.pickle_and_reload(stimuli, pickler=dill)

        self.assertEqual(len(new_stimuli.stimuli), 2)
        for s1, s2 in zip(new_stimuli.stimuli, [img1, img2]):
            np.testing.assert_allclose(s1, s2)
        self.assertEqual(new_stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(list(new_stimuli.sizes), [(100, 200), (50, 150)])
        self.assertEqual(new_stimuli.stimulus_ids[1], pysaliency.datasets.get_image_hash(img2))
        self.assertEqual(new_stimuli.stimulus_objects[1].stimulus_id, stimuli.stimulus_ids[1])

    def test_slicing(self):
        count = 10
        widths = np.random.randint(20, 200, size=count)
        heights = np.random.randint(20, 200, size=count)
        images = [np.random.randn(h, w, 3) for h, w in zip(heights, widths)]

        stimuli = pysaliency.Stimuli(images)
        for i in range(count):
            s = stimuli[i]
            np.testing.assert_allclose(s.stimulus_data, stimuli.stimuli[i])
            self.assertEqual(s.stimulus_id, stimuli.stimulus_ids[i])
            self.assertEqual(s.shape, stimuli.shapes[i])
            self.assertEqual(s.size, stimuli.sizes[i])

        indices = [2, 4, 7]
        ss = stimuli[indices]
        for k, i in enumerate(indices):
            np.testing.assert_allclose(ss.stimuli[k], stimuli.stimuli[i])
            self.assertEqual(ss.stimulus_ids[k], stimuli.stimulus_ids[i])
            self.assertEqual(ss.shapes[k], stimuli.shapes[i])
            self.assertEqual(ss.sizes[k], stimuli.sizes[i])

        slc = slice(2, 8, 3)
        ss = stimuli[slc]
        indices = range(len(stimuli))[slc]
        for k, i in enumerate(indices):
            np.testing.assert_allclose(ss.stimuli[k], stimuli.stimuli[i])
            self.assertEqual(ss.stimulus_ids[k], stimuli.stimulus_ids[i])
            self.assertEqual(ss.shapes[k], stimuli.shapes[i])
            self.assertEqual(ss.sizes[k], stimuli.sizes[i])


class TestFileStimuli(TestWithData):
    def test_file_stimuli(self):
        img1 = np.random.randint(255, size=(100, 200, 3)).astype('uint8')
        filename1 = os.path.join(self.data_path, 'img1.png')
        imwrite(filename1, img1)

        img2 = np.random.randint(255, size=(50, 150)).astype('uint8')
        filename2 = os.path.join(self.data_path, 'img2.png')
        imwrite(filename2, img2)

        stimuli = pysaliency.FileStimuli([filename1, filename2])

        self.assertEqual(len(stimuli.stimuli), 2)
        for s1, s2 in zip(stimuli.stimuli, [img1, img2]):
            np.testing.assert_allclose(s1, s2)
        self.assertEqual(stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(list(stimuli.sizes), [(100, 200), (50, 150)])
        self.assertEqual(stimuli.stimulus_ids[1], pysaliency.datasets.get_image_hash(img2))
        self.assertEqual(stimuli.stimulus_objects[1].stimulus_id, stimuli.stimulus_ids[1])

        new_stimuli = self.pickle_and_reload(stimuli, pickler=dill)

        self.assertEqual(len(new_stimuli.stimuli), 2)
        for s1, s2 in zip(new_stimuli.stimuli, [img1, img2]):
            np.testing.assert_allclose(s1, s2)
        self.assertEqual(new_stimuli.shapes, [(100, 200, 3), (50, 150)])
        self.assertEqual(list(new_stimuli.sizes), [(100, 200), (50, 150)])
        self.assertEqual(new_stimuli.stimulus_ids[1], pysaliency.datasets.get_image_hash(img2))
        self.assertEqual(new_stimuli.stimulus_objects[1].stimulus_id, stimuli.stimulus_ids[1])

    def test_slicing(self):
        count = 10
        widths = np.random.randint(20, 200, size=count)
        heights = np.random.randint(20, 200, size=count)
        images = [np.random.randint(255, size=(h, w, 3)).astype(np.uint8) for h, w in zip(heights, widths)]
        filenames = []
        for i, img in enumerate(images):
            filename = os.path.join(self.data_path, 'img{}.png'.format(i))
            print(filename)
            print(img.shape)
            print(img.dtype)
            imwrite(filename, img)
            filenames.append(filename)

        stimuli = pysaliency.FileStimuli(filenames)
        for i in range(count):
            s = stimuli[i]
            np.testing.assert_allclose(s.stimulus_data, stimuli.stimuli[i])
            self.assertEqual(s.stimulus_id, stimuli.stimulus_ids[i])
            self.assertEqual(s.shape, stimuli.shapes[i])
            self.assertEqual(s.size, stimuli.sizes[i])

        indices = [2, 4, 7]
        ss = stimuli[indices]
        for k, i in enumerate(indices):
            np.testing.assert_allclose(ss.stimuli[k], stimuli.stimuli[i])
            self.assertEqual(ss.stimulus_ids[k], stimuli.stimulus_ids[i])
            self.assertEqual(ss.shapes[k], stimuli.shapes[i])
            self.assertEqual(list(ss.sizes[k]), list(stimuli.sizes[i]))

        slc = slice(2, 8, 3)
        ss = stimuli[slc]
        indices = range(len(stimuli))[slc]
        for k, i in enumerate(indices):
            np.testing.assert_allclose(ss.stimuli[k], stimuli.stimuli[i])
            self.assertEqual(ss.stimulus_ids[k], stimuli.stimulus_ids[i])
            self.assertEqual(ss.shapes[k], stimuli.shapes[i])
            self.assertEqual(list(ss.sizes[k]), list(stimuli.sizes[i]))


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


def test_copy_scanpaths(fixation_trains):
    copied_fixation_trains = fixation_trains.copy()
    compare_scanpaths(copied_fixation_trains, fixation_trains)


def test_copy_fixations(fixation_trains):
    fixations = fixation_trains[:]
    copied_fixations = fixations.copy()
    compare_fixations(copied_fixations, fixations)


def test_write_read_scanpaths_pathlib(tmp_path, fixation_trains):
    filename = tmp_path / 'scanpaths.hdf5'
    fixation_trains.to_hdf5(filename)

    new_fixation_trains = pysaliency.read_hdf5(filename)

    # make sure there is no sophisticated caching...
    assert fixation_trains is not new_fixation_trains
    compare_scanpaths(fixation_trains, new_fixation_trains)


def test_write_read_scanpaths(tmp_path, fixation_trains):
    filename = tmp_path / 'scanpaths.hdf5'
    fixation_trains.to_hdf5(str(filename))

    new_fixation_trains = pysaliency.read_hdf5(str(filename))

    # make sure there is no sophisticated caching...
    assert fixation_trains is not new_fixation_trains
    compare_scanpaths(fixation_trains, new_fixation_trains)


def test_scanpath_lengths(fixation_trains):
    np.testing.assert_array_equal(fixation_trains.train_lengths, [3, 2, 3])


def test_scanpath_attributes(fixation_trains):
    assert "task" in fixation_trains.scanpath_attributes
    assert "task" in fixation_trains.__attributes__

    np.testing.assert_array_equal(fixation_trains.scanpath_attributes['multi_dim_attribute'][0], [0, 1])
    np.testing.assert_array_equal(fixation_trains.multi_dim_attribute[2], [0, 1])


def test_scanpath_fixation_attributes(fixation_trains):
    assert "durations" in fixation_trains.scanpath_fixation_attributes
    np.testing.assert_array_equal(
        fixation_trains.scanpath_fixation_attributes['durations'],
        np.array([
            [42, 25, 100],
            [99, 98, np.nan],
            [200, 150, 120]
        ])
    )

    assert "duration" in fixation_trains.__attributes__
    np.testing.assert_array_equal(fixation_trains.duration, np.array([
            42, 25, 100,
            99, 98,
            200, 150, 120
        ]))
    assert "duration_hist" in fixation_trains.__attributes__
    np.testing.assert_array_equal(fixation_trains.duration_hist[6], [200, np.nan])


@pytest.mark.parametrize('scanpath_indices,fixation_indices', [
    ([0, 2], [0, 1, 2, 5, 6, 7]),
    ([1, 2], [3, 4, 5, 6, 7]),
    ([2], [5, 6, 7]),
])
def test_filter_fixation_trains(fixation_trains, scanpath_indices, fixation_indices):
    sub_fixations = fixation_trains.filter_fixation_trains(scanpath_indices)

    np.testing.assert_array_equal(
        sub_fixations.train_xs,
        fixation_trains.train_xs[scanpath_indices]
    )
    np.testing.assert_array_equal(
        sub_fixations.train_ys,
        fixation_trains.train_ys[scanpath_indices]
    )
    np.testing.assert_array_equal(
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

    np.testing.assert_array_equal(
        sub_fixations.scanpath_fixation_attributes['durations'],
        fixation_trains.scanpath_fixation_attributes['durations'][scanpath_indices]
    )

    compare_fixations(sub_fixations, fixation_trains[fixation_indices])


def test_read_hdf5_caching(fixation_trains, tmp_path):
    filename = tmp_path / 'fixations.hdf5'
    fixation_trains.to_hdf5(str(filename))

    new_fixations = pysaliency.read_hdf5(str(filename))

    assert new_fixations is not fixation_trains

    new_fixations2 = pysaliency.read_hdf5(str(filename))
    assert new_fixations2 is new_fixations, "objects should not be read into memory multiple times"


def test_fixation_trains_copy(fixation_trains):
    copied_fixation_trains = fixation_trains.copy()
    assert isinstance(copied_fixation_trains, FixationTrains)
    compare_fixations(fixation_trains, copied_fixation_trains)


def test_fixations_copy(fixation_trains):
    fixations = fixation_trains[:-1]
    assert isinstance(fixations, Fixations)
    copied_fixations = fixations.copy()
    assert isinstance(copied_fixations, Fixations)
    compare_fixations(fixations, copied_fixations)


@pytest.fixture
def stimuli_with_attributes():
    stimuli_data = [np.random.randint(0, 255, size=(25, 30, 3)) for i in range(10)]
    attributes = {
        'dva': list(range(10)),
        'other_stuff': np.random.randn(10),
        'some_strings': list('abcdefghij'),
    }
    return pysaliency.Stimuli(stimuli_data, attributes=attributes)


def test_stimuli_attributes(stimuli_with_attributes, tmp_path):
    filename = tmp_path / 'stimuli.hdf5'
    stimuli_with_attributes.to_hdf5(str(filename))

    new_stimuli = pysaliency.read_hdf5(str(filename))

    assert stimuli_with_attributes.attributes.keys() == new_stimuli.attributes.keys()
    np.testing.assert_array_equal(stimuli_with_attributes.attributes['dva'], new_stimuli.attributes['dva'])
    np.testing.assert_array_equal(stimuli_with_attributes.attributes['other_stuff'], new_stimuli.attributes['other_stuff'])
    np.testing.assert_array_equal(stimuli_with_attributes.attributes['some_strings'], new_stimuli.attributes['some_strings'])

    partial_stimuli = stimuli_with_attributes[:5]
    assert stimuli_with_attributes.attributes.keys() == partial_stimuli.attributes.keys()
    assert stimuli_with_attributes.attributes['dva'][:5] == partial_stimuli.attributes['dva']
    assert stimuli_with_attributes.attributes['some_strings'][:5] == partial_stimuli.attributes['some_strings']


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


def test_file_stimuli_attributes(file_stimuli_with_attributes, tmp_path):
    filename = tmp_path / 'stimuli.hdf5'
    file_stimuli_with_attributes.to_hdf5(str(filename))

    new_stimuli = pysaliency.read_hdf5(str(filename))

    assert file_stimuli_with_attributes.attributes.keys() == new_stimuli.attributes.keys()
    np.testing.assert_array_equal(file_stimuli_with_attributes.attributes['dva'], new_stimuli.attributes['dva'])
    np.testing.assert_array_equal(file_stimuli_with_attributes.attributes['other_stuff'], new_stimuli.attributes['other_stuff'])
    np.testing.assert_array_equal(file_stimuli_with_attributes.attributes['some_strings'], new_stimuli.attributes['some_strings'])

    partial_stimuli = file_stimuli_with_attributes[:5]
    assert file_stimuli_with_attributes.attributes.keys() == partial_stimuli.attributes.keys()
    assert file_stimuli_with_attributes.attributes['dva'][:5] == partial_stimuli.attributes['dva']
    assert file_stimuli_with_attributes.attributes['some_strings'][:5] == partial_stimuli.attributes['some_strings']


def test_concatenate_stimuli_with_attributes(stimuli_with_attributes, file_stimuli_with_attributes):
    concatenated_stimuli = pysaliency.datasets.concatenate_stimuli([stimuli_with_attributes, file_stimuli_with_attributes])

    assert file_stimuli_with_attributes.attributes.keys() == concatenated_stimuli.attributes.keys()
    np.testing.assert_allclose(stimuli_with_attributes.attributes['dva'], concatenated_stimuli.attributes['dva'][:len(stimuli_with_attributes)])
    np.testing.assert_allclose(file_stimuli_with_attributes.attributes['dva'], concatenated_stimuli.attributes['dva'][len(stimuli_with_attributes):])


def test_concatenate_file_stimuli(file_stimuli_with_attributes):
    concatenated_stimuli = pysaliency.datasets.concatenate_stimuli([file_stimuli_with_attributes, file_stimuli_with_attributes])

    assert isinstance(concatenated_stimuli, pysaliency.FileStimuli)
    assert concatenated_stimuli.filenames == file_stimuli_with_attributes.filenames + file_stimuli_with_attributes.filenames


def test_concatenate_fixations(fixation_trains):
    new_fixations = pysaliency.Fixations.concatenate((fixation_trains, fixation_trains))
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

def test_concatenate_scanpaths(fixation_trains):
    fixation_trains2 = fixation_trains.copy()

    del fixation_trains2.scanpath_attributes['task']
    delattr(fixation_trains2, 'task')
    fixation_trains2.auto_attributes.remove('task')
    fixation_trains2.__attributes__.remove('task')

    new_scanpaths = pysaliency.FixationTrains.concatenate((fixation_trains, fixation_trains2))
    assert isinstance(new_scanpaths, pysaliency.Fixations)
    np.testing.assert_allclose(
        new_scanpaths.x,
        np.concatenate((fixation_trains.x, fixation_trains2.x))
    )

    np.testing.assert_allclose(
        new_scanpaths.n,
        np.concatenate((fixation_trains.n, fixation_trains2.n))
    )

    assert set(new_scanpaths.__attributes__) == {'subjects', 'duration', 'duration_hist', 'multi_dim_attribute', 'scanpath_index', 'some_attribute'}

    np.testing.assert_allclose(
        new_scanpaths.some_attribute,
        np.concatenate((fixation_trains.some_attribute, fixation_trains2.some_attribute))
    )


@pytest.mark.parametrize('stimulus_indices', [[0], [1], [0, 1]])
def test_create_subset_fixation_trains(file_stimuli_with_attributes, fixation_trains, stimulus_indices):
    sub_stimuli, sub_fixations = pysaliency.datasets.create_subset(file_stimuli_with_attributes, fixation_trains, stimulus_indices)

    assert isinstance(sub_fixations, pysaliency.FixationTrains)
    assert len(sub_stimuli) == len(stimulus_indices)
    np.testing.assert_array_equal(sub_fixations.x, fixation_trains.x[np.isin(fixation_trains.n, stimulus_indices)])


@pytest.mark.parametrize('stimulus_indices', [[0], [1], [0, 1]])
def test_create_subset_fixations(file_stimuli_with_attributes, fixation_trains, stimulus_indices):
    # convert to fixations
    fixations = fixation_trains[np.arange(len(fixation_trains))]
    sub_stimuli, sub_fixations = pysaliency.datasets.create_subset(file_stimuli_with_attributes, fixations, stimulus_indices)

    assert not isinstance(sub_fixations, pysaliency.FixationTrains)
    assert len(sub_stimuli) == len(stimulus_indices)
    np.testing.assert_array_equal(sub_fixations.x, fixations.x[np.isin(fixations.n, stimulus_indices)])


def test_create_subset_numpy_indices(file_stimuli_with_attributes, fixation_trains):
    stimulus_indices = np.array([0, 3])

    sub_stimuli, sub_fixations = pysaliency.datasets.create_subset(file_stimuli_with_attributes, fixation_trains, stimulus_indices)

    assert isinstance(sub_fixations, pysaliency.FixationTrains)
    assert len(sub_stimuli) == 2
    np.testing.assert_array_equal(sub_fixations.x, fixation_trains.x[np.isin(fixation_trains.n, stimulus_indices)])


def test_create_subset_numpy_mask(file_stimuli_with_attributes, fixation_trains):
    print(len(file_stimuli_with_attributes))
    stimulus_indices = np.zeros(len(file_stimuli_with_attributes), dtype=bool)
    stimulus_indices[0] = True
    stimulus_indices[2] = True

    sub_stimuli, sub_fixations = pysaliency.datasets.create_subset(file_stimuli_with_attributes, fixation_trains, stimulus_indices)

    assert isinstance(sub_fixations, pysaliency.FixationTrains)
    assert len(sub_stimuli) == 2
    np.testing.assert_array_equal(sub_fixations.x, fixation_trains.x[np.isin(fixation_trains.n, [0, 2])])


@given(st.lists(elements=st.integers(min_value=0, max_value=7), min_size=1))
def test_scanpaths_from_fixations(fixation_indices):
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
    new_scanpaths, new_indices = scanpaths_from_fixations(sub_fixations)
    new_sub_fixations = new_scanpaths[new_indices]

    compare_fixations(sub_fixations, new_sub_fixations, crop_length=True)


if __name__ == '__main__':
    unittest.main()
