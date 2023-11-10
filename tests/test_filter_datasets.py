from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np
from imageio import imwrite

import pysaliency
import pysaliency.filter_datasets as filter_datasets
from pysaliency.filter_datasets import filter_fixations_by_attribute, filter_stimuli_by_attribute, filter_scanpaths_by_attribute, filter_scanpaths_by_lengths, create_subset
from test_datasets import compare_fixations, compare_scanpaths


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
    multi_dim_attribute = [[0.0, 1],[0, 3], [4, 5.5]]
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


@pytest.fixture
def stimuli_with_different_sizes():
    return pysaliency.Stimuli([
        np.random.randn(40, 40, 3),
        np.random.randn(40, 40, 3),
        np.random.randn(20, 40, 3),
        np.random.randn(20, 40, 3),
        np.random.randn(40, 20, 3),
        np.random.randn(40, 20, 3),
        np.random.randn(40, 20),
        np.random.randn(20, 20, 3),
        np.random.randn(20, 20, 3),
    ])


def assert_stimuli_equal(actual, expected):
    assert list(actual.stimulus_ids) == list(expected.stimulus_ids)


@pytest.mark.parametrize('size,indices', [
    ((40, 40), [0, 1]),
    ((20, 40), [2, 3]),
    ((40, 20), [4, 5, 6]),
    ((20, 20), [7, 8]),
])
def test_filter_stimuli_by_size_tuple(stimuli_with_different_sizes, fixation_trains, size, indices):
    filtered_stimuli, _ = filter_datasets.filter_stimuli_by_size(
        stimuli_with_different_sizes,
        fixation_trains,
        size=size
    )

    assert_stimuli_equal(
        filtered_stimuli,
        stimuli_with_different_sizes[indices]
    )


def test_filter_stimuli_by_size_array(stimuli_with_different_sizes, fixation_trains):
    filtered_stimuli, _ = filter_datasets.filter_stimuli_by_size(
        stimuli_with_different_sizes,
        fixation_trains,
        size=[40, 40]
    )

    assert_stimuli_equal(
        filtered_stimuli,
        stimuli_with_different_sizes[[0, 1]]
    )


@pytest.mark.parametrize('sizes,indices', [
    ([(40, 40)], [0, 1]),
    ([(20, 40), (40, 40)], [0, 1, 2, 3]),
])
def test_filter_stimuli_by_size_multiple(stimuli_with_different_sizes, fixation_trains, sizes, indices):
    filtered_stimuli, _ = filter_datasets.filter_stimuli_by_size(
        stimuli_with_different_sizes,
        fixation_trains,
        sizes=sizes
    )

    assert_stimuli_equal(
        filtered_stimuli,
        stimuli_with_different_sizes[indices]
    )


@pytest.fixture
def many_stimuli():
    stimuli = [np.random.randn(40, 40, 3) for i in range(1003)]
    category = np.array([i % 10 for i in range(len(stimuli))])
    category2 = np.array([i // 500 for i in range(len(stimuli))])
    return pysaliency.Stimuli(stimuli, attributes={'category': category, 'category2': category2})


@pytest.mark.parametrize('crossval_folds', [10, 11, 12])
@pytest.mark.parametrize('val_folds', [1, 2, 3])
@pytest.mark.parametrize('test_folds', [1])
def test_crossval_splits(many_stimuli, crossval_folds, val_folds, test_folds):
    if not test_folds and val_folds != 1:
        return  # this case is raises an implementation error right now

    tmp_model = pysaliency.UniformModel()
    fixations = tmp_model.sample(many_stimuli, 100)

    train_stimuli = []
    train_fixations = []
    val_stimuli = []
    val_fixations = []
    test_stimuli = []
    test_fixations = []

    for _train_stimuli, _train_fixations, _val_stimuli, _val_fixations, _test_stimuli, _test_fixations in \
            filter_datasets.iterate_crossvalidation(many_stimuli, fixations, crossval_folds=crossval_folds, val_folds=val_folds, test_folds=test_folds, random=True):

        assert not set(_train_stimuli.stimulus_ids).intersection(_val_stimuli.stimulus_ids)
        assert not set(_train_stimuli.stimulus_ids).intersection(_test_stimuli.stimulus_ids)

        if not test_folds:  # otherwise test is validation
            assert not set(_val_stimuli.stimulus_ids).intersection(_test_stimuli.stimulus_ids)

        train_stimuli.append(_train_stimuli)
        train_fixations.append(_train_fixations)

        val_stimuli.append(_val_stimuli)
        val_fixations.append(_val_fixations)

        test_stimuli.append(_test_stimuli)
        test_fixations.append(_test_fixations)

    assert sum(len(s) for s in val_stimuli) == val_folds * len(many_stimuli)
    assert sum(len(s) for s in test_stimuli) == test_folds * len(many_stimuli)
    assert sum(len(s) for s in train_stimuli) == (crossval_folds - val_folds - test_folds) * len(many_stimuli)

    assert sum(len(f.x) for f in val_fixations) == val_folds * len(fixations.x)
    assert sum(len(f.x) for f in test_fixations) == test_folds * len(fixations.x)
    assert sum(len(f.x) for f in train_fixations) == (crossval_folds - val_folds - test_folds) * len(fixations.x)

    assert len(train_stimuli) == crossval_folds


@pytest.mark.parametrize('crossval_folds', [10])
@pytest.mark.parametrize('val_folds', [1, 2, 3])
@pytest.mark.parametrize('test_folds', [1])
def test_stratified_crossval_splits(many_stimuli, crossval_folds, val_folds, test_folds):
    if not test_folds and val_folds != 1:
        return  # this case is raises an implementation error right now

    tmp_model = pysaliency.UniformModel()
    fixations = tmp_model.sample(many_stimuli, 100)

    train_stimuli = []
    train_fixations = []
    val_stimuli = []
    val_fixations = []
    test_stimuli = []
    test_fixations = []

    for _train_stimuli, _train_fixations, _val_stimuli, _val_fixations, _test_stimuli, _test_fixations in \
            filter_datasets.iterate_crossvalidation(many_stimuli, fixations, crossval_folds=crossval_folds, val_folds=val_folds, test_folds=test_folds, random=True, stratified_attributes=['category']):

        assert not set(_train_stimuli.stimulus_ids).intersection(_val_stimuli.stimulus_ids)
        assert not set(_train_stimuli.stimulus_ids).intersection(_test_stimuli.stimulus_ids)

        if not test_folds:  # otherwise test is validation
            assert not set(_val_stimuli.stimulus_ids).intersection(_test_stimuli.stimulus_ids)

        np.testing.assert_allclose(
            np.sum(_train_stimuli.attributes['category'] == 0),
            len(many_stimuli) / crossval_folds * (crossval_folds - val_folds - test_folds) * 0.1,
            atol=1
        )
        np.testing.assert_allclose(
            np.sum(_val_stimuli.attributes['category'] == 0),
            len(many_stimuli) / crossval_folds * val_folds * 0.1,
            atol=1
        )
        np.testing.assert_allclose(
            np.sum(_test_stimuli.attributes['category'] == 0),
            len(many_stimuli) / crossval_folds * test_folds * 0.1,
            atol=1
        )

        train_stimuli.append(_train_stimuli)
        train_fixations.append(_train_fixations)

        val_stimuli.append(_val_stimuli)
        val_fixations.append(_val_fixations)

        test_stimuli.append(_test_stimuli)
        test_fixations.append(_test_fixations)

    assert sum(len(s) for s in val_stimuli) == val_folds * len(many_stimuli)
    assert sum(len(s) for s in test_stimuli) == test_folds * len(many_stimuli)
    assert sum(len(s) for s in train_stimuli) == (crossval_folds - val_folds - test_folds) * len(many_stimuli)

    assert sum(len(f.x) for f in val_fixations) == val_folds * len(fixations.x)
    assert sum(len(f.x) for f in test_fixations) == test_folds * len(fixations.x)
    assert sum(len(f.x) for f in train_fixations) == (crossval_folds - val_folds - test_folds) * len(fixations.x)

    assert len(train_stimuli) == crossval_folds


@pytest.mark.parametrize('crossval_folds', [10])
@pytest.mark.parametrize('val_folds', [1, 2, 3])
@pytest.mark.parametrize('test_folds', [1])
def test_stratified_crossval_splits_multiple_attributes(many_stimuli, crossval_folds, val_folds, test_folds):
    if not test_folds and val_folds != 1:
        return  # this case is raises an implementation error right now

    tmp_model = pysaliency.UniformModel()
    fixations = tmp_model.sample(many_stimuli, 100)

    train_stimuli = []
    train_fixations = []
    val_stimuli = []
    val_fixations = []
    test_stimuli = []
    test_fixations = []

    for _train_stimuli, _train_fixations, _val_stimuli, _val_fixations, _test_stimuli, _test_fixations in \
            filter_datasets.iterate_crossvalidation(many_stimuli, fixations, crossval_folds=crossval_folds, val_folds=val_folds, test_folds=test_folds, random=True, stratified_attributes=['category', 'category2']):

        assert not set(_train_stimuli.stimulus_ids).intersection(_val_stimuli.stimulus_ids)
        assert not set(_train_stimuli.stimulus_ids).intersection(_test_stimuli.stimulus_ids)

        if not test_folds:  # otherwise test is validation
            assert not set(_val_stimuli.stimulus_ids).intersection(_test_stimuli.stimulus_ids)

        np.testing.assert_allclose(
            np.sum(_train_stimuli.attributes['category'] == 0),
            len(many_stimuli) / crossval_folds * (crossval_folds - val_folds - test_folds) * 0.1,
            atol=1
        )
        np.testing.assert_allclose(
            np.sum(_val_stimuli.attributes['category'] == 0),
            len(many_stimuli) / crossval_folds * val_folds * 0.1,
            atol=1
        )
        np.testing.assert_allclose(
            np.sum(_test_stimuli.attributes['category'] == 0),
            len(many_stimuli) / crossval_folds * test_folds * 0.1,
            atol=1
        )

        train_stimuli.append(_train_stimuli)
        train_fixations.append(_train_fixations)

        val_stimuli.append(_val_stimuli)
        val_fixations.append(_val_fixations)

        test_stimuli.append(_test_stimuli)
        test_fixations.append(_test_fixations)

    assert sum(len(s) for s in val_stimuli) == val_folds * len(many_stimuli)
    assert sum(len(s) for s in test_stimuli) == test_folds * len(many_stimuli)
    assert sum(len(s) for s in train_stimuli) == (crossval_folds - val_folds - test_folds) * len(many_stimuli)

    assert sum(len(f.x) for f in val_fixations) == val_folds * len(fixations.x)
    assert sum(len(f.x) for f in test_fixations) == test_folds * len(fixations.x)
    assert sum(len(f.x) for f in train_fixations) == (crossval_folds - val_folds - test_folds) * len(fixations.x)

    assert len(train_stimuli) == crossval_folds


def test_filter_stimuli_by_attribute_dva(file_stimuli_with_attributes, fixation_trains):
    fixations = fixation_trains[:]
    attribute_name = 'dva' 
    attribute_value = 1
    invert_match = False
    filtered_stimuli, filtered_fixations = filter_stimuli_by_attribute(file_stimuli_with_attributes, fixations, attribute_name, attribute_value, invert_match)
    inds = [1]
    expected_stimuli, expected_fixations = create_subset(file_stimuli_with_attributes, fixations, inds)
    compare_fixations(filtered_fixations, expected_fixations)
    assert_stimuli_equal(filtered_stimuli, expected_stimuli)


def test_filter_stimuli_by_attribute_some_strings_invert_match(file_stimuli_with_attributes, fixation_trains):
    fixations = fixation_trains[:]
    attribute_name = 'some_strings' 
    attribute_value = 'n'
    invert_match = True
    filtered_stimuli, filtered_fixations = filter_stimuli_by_attribute(file_stimuli_with_attributes, fixations, attribute_name, attribute_value, invert_match)
    inds = list(range(0, 13)) + list(range(14, 18))
    expected_stimuli, expected_fixations = create_subset(file_stimuli_with_attributes, fixations, inds)
    compare_fixations(filtered_fixations, expected_fixations)
    assert_stimuli_equal(filtered_stimuli, expected_stimuli)


def test_filter_fixations_by_attribute_subject_invert_match(fixation_trains):
    fixations = fixation_trains[:]
    attribute_name = 'subjects' 
    attribute_value = 0
    invert_match = True
    filtered_fixations = filter_fixations_by_attribute(fixations, attribute_name, attribute_value, invert_match)
    inds = [3, 4, 5, 6, 7]
    expected_fixations = fixations[inds]
    compare_fixations(filtered_fixations, expected_fixations)


def test_filter_fixations_by_attribute_some_attribute(fixation_trains):
    fixations = fixation_trains[:]
    attribute_name = 'some_attribute' 
    attribute_value = 2
    invert_match = False
    filtered_fixations = filter_fixations_by_attribute(fixations, attribute_name, attribute_value, invert_match)
    inds = [2]
    expected_fixations = fixations[inds]
    compare_fixations(filtered_fixations, expected_fixations)


def test_filter_fixations_by_attribute_some_attribute_invert_match(fixation_trains):
    fixations = fixation_trains[:]
    attribute_name = 'some_attribute' 
    attribute_value = 3
    invert_match = True
    filtered_fixations = filter_fixations_by_attribute(fixations, attribute_name, attribute_value, invert_match)
    inds = list(range(0, 3)) + list(range(4, 8))
    expected_fixations = fixations[inds]
    compare_fixations(filtered_fixations, expected_fixations)


def test_filter_scanpaths_by_attribute_task(fixation_trains):
    scanpaths = fixation_trains
    attribute_name = 'task' 
    attribute_value = 0
    invert_match = False
    filtered_scanpaths = filter_scanpaths_by_attribute(scanpaths, attribute_name, attribute_value, invert_match)
    inds = [0, 2]
    expected_scanpaths = scanpaths.filter_fixation_trains(inds)
    compare_scanpaths(filtered_scanpaths, expected_scanpaths)


def test_filter_scanpaths_by_attribute_multi_dim_attribute(fixation_trains):
    scanpaths = fixation_trains
    attribute_name = 'multi_dim_attribute' 
    attribute_value = [0, 3]
    invert_match = False
    filtered_scanpaths = filter_scanpaths_by_attribute(scanpaths, attribute_name, attribute_value, invert_match)
    inds = [1]
    expected_scanpaths = scanpaths.filter_fixation_trains(inds)
    compare_scanpaths(filtered_scanpaths, expected_scanpaths)


def test_filter_scanpaths_by_attribute_multi_dim_attribute_invert_match(fixation_trains):
    scanpaths = fixation_trains
    attribute_name = 'multi_dim_attribute' 
    attribute_value = [0, 1]
    invert_match = True
    filtered_scanpaths = filter_scanpaths_by_attribute(scanpaths, attribute_name, attribute_value, invert_match)
    inds = [1, 2]
    expected_scanpaths = scanpaths.filter_fixation_trains(inds)
    compare_scanpaths(filtered_scanpaths, expected_scanpaths)


@pytest.mark.parametrize('intervals', [([(1, 2), (2, 3)]), ([(2, 3), (3, 4)]), ([(2)]), ([(3)])])
def test_filter_scanpaths_by_lengths(fixation_trains, intervals):
    scanpaths = fixation_trains
    filtered_scanpaths = filter_scanpaths_by_lengths(scanpaths, intervals)
    if intervals == [(1, 2), (2, 3)]:
        inds = [1]
        expected_scanpaths = scanpaths.filter_fixation_trains(inds)
        compare_scanpaths(filtered_scanpaths, expected_scanpaths)
    if intervals == [(2, 3), (3, 4)]:
        inds = [0, 1, 2]
        expected_scanpaths = scanpaths.filter_fixation_trains(inds)
        compare_scanpaths(filtered_scanpaths, expected_scanpaths)
    if intervals == [(2)]:
        inds = [1]
        expected_scanpaths = scanpaths.filter_fixation_trains(inds)
        compare_scanpaths(filtered_scanpaths, expected_scanpaths)
    if intervals == [(3)]:
        inds = [0, 2]
        expected_scanpaths = scanpaths.filter_fixation_trains(inds)
        compare_scanpaths(filtered_scanpaths, expected_scanpaths)
