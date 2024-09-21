from __future__ import absolute_import, division, print_function

import os.path
import pickle
import unittest
from copy import deepcopy

import dill
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from imageio import imwrite
from test_helpers import TestWithData

import pysaliency
from pysaliency.datasets import Fixations, FixationTrains, Scanpaths, Stimulus, check_prediction_shape, scanpaths_from_fixations
from pysaliency.utils.variable_length_array import VariableLengthArray


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

    partial_stimuli = stimuli_with_attributes[[1, 2, 6]]
    assert stimuli_with_attributes.attributes.keys() == partial_stimuli.attributes.keys()
    assert list(np.array(stimuli_with_attributes.attributes['dva'])[[1, 2, 6]]) == partial_stimuli.attributes['dva']
    assert list(np.array(stimuli_with_attributes.attributes['some_strings'])[[1, 2, 6]]) == partial_stimuli.attributes['some_strings']

    mask = np.array([True, False, True, False, True, False, True, False, True, False, True, False])
    with pytest.raises(ValueError):
        partial_stimuli = stimuli_with_attributes[mask]

    mask = np.array([True, False, True, False, True, False, True, False, True, False])
    partial_stimuli = stimuli_with_attributes[mask]
    assert stimuli_with_attributes.attributes.keys() == partial_stimuli.attributes.keys()
    assert list(np.array(stimuli_with_attributes.attributes['dva'])[mask]) == partial_stimuli.attributes['dva']
    assert list(np.array(stimuli_with_attributes.attributes['some_strings'])[mask]) == partial_stimuli.attributes['some_strings']



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

    partial_stimuli = file_stimuli_with_attributes[[1, 2, 6]]
    assert file_stimuli_with_attributes.attributes.keys() == partial_stimuli.attributes.keys()
    assert list(np.array(file_stimuli_with_attributes.attributes['dva'])[[1, 2, 6]]) == partial_stimuli.attributes['dva']
    assert list(np.array(file_stimuli_with_attributes.attributes['some_strings'])[[1, 2, 6]]) == partial_stimuli.attributes['some_strings']

    mask = np.array([True, False, True, False, True, False, True, False, True, False])
    with pytest.raises(ValueError):
        partial_stimuli = file_stimuli_with_attributes[mask]

    mask = np.array([True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False])
    partial_stimuli = file_stimuli_with_attributes[mask]

    assert file_stimuli_with_attributes.attributes.keys() == partial_stimuli.attributes.keys()
    assert list(np.array(file_stimuli_with_attributes.attributes['dva'])[mask]) == partial_stimuli.attributes['dva']
    assert list(np.array(file_stimuli_with_attributes.attributes['some_strings'])[mask]) == partial_stimuli.attributes['some_strings']


def test_concatenate_stimuli_with_attributes(stimuli_with_attributes, file_stimuli_with_attributes):
    concatenated_stimuli = pysaliency.datasets.concatenate_stimuli([stimuli_with_attributes, file_stimuli_with_attributes])

    assert file_stimuli_with_attributes.attributes.keys() == concatenated_stimuli.attributes.keys()
    np.testing.assert_allclose(stimuli_with_attributes.attributes['dva'], concatenated_stimuli.attributes['dva'][:len(stimuli_with_attributes)])
    np.testing.assert_allclose(file_stimuli_with_attributes.attributes['dva'], concatenated_stimuli.attributes['dva'][len(stimuli_with_attributes):])


def test_concatenate_file_stimuli(file_stimuli_with_attributes):
    concatenated_stimuli = pysaliency.datasets.concatenate_stimuli([file_stimuli_with_attributes, file_stimuli_with_attributes])

    assert isinstance(concatenated_stimuli, pysaliency.FileStimuli)
    assert concatenated_stimuli.filenames == file_stimuli_with_attributes.filenames + file_stimuli_with_attributes.filenames


def test_check_prediction_shape():
    # Test with matching shapes
    prediction = np.random.rand(10, 10)
    stimulus = np.random.rand(10, 10)
    check_prediction_shape(prediction, stimulus)  # Should not raise any exception

    # Test with matching shapes, colorimage
    prediction = np.random.rand(10, 10)
    stimulus = np.random.rand(10, 10, 3)
    check_prediction_shape(prediction, stimulus)  # Should not raise any exception

    # Test with mismatching shapes
    prediction = np.random.rand(10, 10)
    stimulus = np.random.rand(10, 11)
    with pytest.raises(ValueError) as excinfo:
        check_prediction_shape(prediction, stimulus)
    assert str(excinfo.value) == "Prediction shape (10, 10) does not match stimulus shape (10, 11)"

    # Test with Stimulus object
    prediction = np.random.rand(10, 10)
    stimulus = Stimulus(np.random.rand(10, 10))
    check_prediction_shape(prediction, stimulus)  # Should not raise any exception

    # Test with Stimulus object
    prediction = np.random.rand(10, 10)
    stimulus = Stimulus(np.random.rand(10, 10, 3))
    check_prediction_shape(prediction, stimulus)  # Should not raise any exception

    # Test with mismatching shapes and Stimulus object
    prediction = np.random.rand(10, 10)
    stimulus = Stimulus(np.random.rand(10, 11))
    with pytest.raises(ValueError) as excinfo:
        check_prediction_shape(prediction, stimulus)
    assert str(excinfo.value) == "Prediction shape (10, 10) does not match stimulus shape (10, 11)"


@pytest.mark.parametrize(
        'stimuli',
        ['stimuli_with_attributes', 'file_stimuli_with_attributes']
)
def test_substimuli_inherit_cachedstimulus_ids(stimuli, request):
    _stimuli = request.getfixturevalue(stimuli)
    # load some stimulus ids
    cache_stimulus_indices = [1, 2, 5]
    # make sure the ids are cached
    for i in cache_stimulus_indices:
        _stimuli.stimulus_ids[i]

    assert len(_stimuli.stimulus_ids._cache) == len(cache_stimulus_indices)

    sub_stimuli = _stimuli[1:5]
    assert set(sub_stimuli.stimulus_ids._cache.keys()) == {0, 1}

