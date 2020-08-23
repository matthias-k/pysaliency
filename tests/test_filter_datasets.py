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
