import numpy as np
import pytest
from pytest import approx
import pysaliency
from scipy.stats import kurtosis, skew

from tests.test_external_datasets import _location, entropy


@pytest.mark.slow
@pytest.mark.download
def test_COCO_Search18_task_merge(location):
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_val, fixations_val = pysaliency.external_datasets.get_COCO_Search18(location=real_location)
    if location is None:
        assert isinstance(stimuli_train, pysaliency.Stimuli)
        assert not isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.Stimuli)
        assert not isinstance(stimuli_val, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.FileStimuli)
        assert location.join('COCO-Search18/stimuli_train.hdf5').check()
        assert location.join('COCO-Search18/stimuli_validation.hdf5').check()
        assert location.join('COCO-Search18/fixations_train.hdf5').check()
        assert location.join('COCO-Search18/fixations_validation.hdf5').check()

    assert len(stimuli_train) == 3714
    assert len(stimuli_val) == 623
    assert set(stimuli_train.sizes) == {(1050, 1680)}
    assert set(stimuli_val.sizes) == {(1050, 1680)}

    assert len(fixations_train.x) == 207970

    assert np.mean(fixations_train.x) == approx(835.8440337548686)
    assert np.mean(fixations_train.y) == approx(509.6030908304083)
    assert np.mean(fixations_train.t) == approx(3.0987979035437805)
    assert np.mean(fixations_train.scanpath_history_length) == approx(3.0987979035437805)

    assert np.std(fixations_train.x) == approx(336.5760343388881)
    assert np.std(fixations_train.y) == approx(193.04654731407436)
    assert np.std(fixations_train.t) == approx(3.8411822348178664)
    assert np.std(fixations_train.scanpath_history_length) == approx(3.8411822348178664)

    assert kurtosis(fixations_train.x) == approx(-0.6283401149747818)
    assert kurtosis(fixations_train.y) == approx(0.15947671647330974)
    assert kurtosis(fixations_train.t) == approx(12.038491881119654)
    assert kurtosis(fixations_train.scanpath_history_length) == approx(12.038491881119654)

    assert skew(fixations_train.x) == approx(0.1706207784149093)
    assert skew(fixations_train.y) == approx(-0.07268825958515616)
    assert skew(fixations_train.t) == approx(2.804671690266736)
    assert skew(fixations_train.scanpath_history_length) == approx(2.804671690266736)

    assert entropy(fixations_train.n) == approx(11.654309812153487)
    assert (fixations_train.n == 0).sum() == 48

    assert len(fixations_val.x) == 31761

    assert np.mean(fixations_val.x) == approx(841.0752652624287)
    assert np.mean(fixations_val.y) == approx(498.3305594911999)
    assert np.mean(fixations_val.t) == approx(3.107994080790907)
    assert np.mean(fixations_val.scanpath_history_length) == approx(3.107994080790907)

    assert np.std(fixations_val.x) == approx(331.6328528765362)
    assert np.std(fixations_val.y) == approx(195.86110035077112)
    assert np.std(fixations_val.t) == approx(3.7502120687824454)
    assert np.std(fixations_val.scanpath_history_length) == approx(3.7502120687824454)

    assert kurtosis(fixations_val.x) == approx(-0.5973130907561486)
    assert kurtosis(fixations_val.y) == approx(-0.2797786304225598)
    assert kurtosis(fixations_val.t) == approx(11.250011182161305)
    assert kurtosis(fixations_val.scanpath_history_length) == approx(11.250011182161305)

    assert skew(fixations_val.x) == approx(0.14886675209256964)
    assert skew(fixations_val.y) == approx(-0.04086275403802345)
    assert skew(fixations_val.t) == approx(2.671653646130074)
    assert skew(fixations_val.scanpath_history_length) == approx(2.671653646130074)

    assert entropy(fixations_val.n) == approx(9.159600084079305)
    assert (fixations_val.n == 0).sum() == 52

    #assert len(fixations_train) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_train, fixations_train))
    #assert len(fixations_val) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_val, fixations_val))


@pytest.mark.slow
@pytest.mark.download
def test_COCO_Search18_no_task_merge_redundant_images(location):
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_val, fixations_val = pysaliency.external_datasets.get_COCO_Search18(location=real_location, merge_tasks=False, unique_images=False)
    if location is None:
        assert isinstance(stimuli_train, pysaliency.Stimuli)
        assert not isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.Stimuli)
        assert not isinstance(stimuli_val, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.FileStimuli)
        assert location.join('COCO-Search18_no-task-merge_duplicate-images/stimuli_train.hdf5').check()
        assert location.join('COCO-Search18_no-task-merge_duplicate-images/stimuli_validation.hdf5').check()
        assert location.join('COCO-Search18_no-task-merge_duplicate-images/fixations_train.hdf5').check()
        assert location.join('COCO-Search18_no-task-merge_duplicate-images/fixations_validation.hdf5').check()

    #assert len(stimuli_train) == 3714
    #assert len(stimuli_val) == 623
    assert len(stimuli_train) == 4326
    assert len(stimuli_val) == 652
    assert set(stimuli_train.sizes) == {(1050, 1680)}
    assert set(stimuli_val.sizes) == {(1050, 1680)}
    #assert len(set(stimuli_train.stimulus_ids)) == 4326
    #assert len(set(stimuli_val.stimulus_ids)) == 652
    assert len(set(stimuli_train.stimulus_ids)) == 3714
    assert len(set(stimuli_val.stimulus_ids)) == 623

    assert 'task' in stimuli_train.__attributes__
    assert 'task' in stimuli_val.__attributes__
    assert len(np.unique(stimuli_train.attributes['task'])) == 18
    assert len(np.unique(stimuli_val.attributes['task'])) == 18
    assert set(stimuli_train.attributes['task']) == set(range(18))
    assert set(stimuli_val.attributes['task']) == set(range(18))

    assert len(fixations_train.x) == 207970

    assert np.mean(fixations_train.x) == approx(835.8440337548686)
    assert np.mean(fixations_train.y) == approx(509.6030908304083)
    assert np.mean(fixations_train.t) == approx(3.0987979035437805)
    assert np.mean(fixations_train.scanpath_history_length) == approx(3.0987979035437805)

    assert np.std(fixations_train.x) == approx(336.5760343388881)
    assert np.std(fixations_train.y) == approx(193.04654731407436)
    assert np.std(fixations_train.t) == approx(3.8411822348178664)
    assert np.std(fixations_train.scanpath_history_length) == approx(3.8411822348178664)

    assert kurtosis(fixations_train.x) == approx(-0.6283401149747818)
    assert kurtosis(fixations_train.y) == approx(0.15947671647330974)
    assert kurtosis(fixations_train.t) == approx(12.038491881119654)
    assert kurtosis(fixations_train.scanpath_history_length) == approx(12.038491881119654)

    assert skew(fixations_train.x) == approx(0.1706207784149093)
    assert skew(fixations_train.y) == approx(-0.07268825958515616)
    assert skew(fixations_train.t) == approx(2.804671690266736)
    assert skew(fixations_train.scanpath_history_length) == approx(2.804671690266736)

    assert entropy(fixations_train.n) == approx(11.967951796529752)
    assert (fixations_train.n == 0).sum() == 71

    assert len(fixations_val.x) == 31761

    assert np.mean(fixations_val.x) == approx(841.0752652624287)
    assert np.mean(fixations_val.y) == approx(498.3305594911999)
    assert np.mean(fixations_val.t) == approx(3.107994080790907)
    assert np.mean(fixations_val.scanpath_history_length) == approx(3.107994080790907)

    assert np.std(fixations_val.x) == approx(331.6328528765362)
    assert np.std(fixations_val.y) == approx(195.86110035077112)
    assert np.std(fixations_val.t) == approx(3.7502120687824454)
    assert np.std(fixations_val.scanpath_history_length) == approx(3.7502120687824454)

    assert kurtosis(fixations_val.x) == approx(-0.5973130907561486)
    assert kurtosis(fixations_val.y) == approx(-0.2797786304225598)
    assert kurtosis(fixations_val.t) == approx(11.250011182161305)
    assert kurtosis(fixations_val.scanpath_history_length) == approx(11.250011182161305)

    assert skew(fixations_val.x) == approx(0.14886675209256964)
    assert skew(fixations_val.y) == approx(-0.04086275403802345)
    assert skew(fixations_val.t) == approx(2.671653646130074)
    assert skew(fixations_val.scanpath_history_length) == approx(2.671653646130074)

    assert entropy(fixations_val.n) == approx(9.243197427307365)
    assert (fixations_val.n == 0).sum() == 42

    #assert len(fixations_train) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_train, fixations_train))
    #assert len(fixations_val) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_val, fixations_val))


@pytest.mark.slow
@pytest.mark.download
def test_COCO_Search18_no_task_merge_unique_images(location):
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_val, fixations_val = pysaliency.external_datasets.get_COCO_Search18(location=real_location, merge_tasks=False, unique_images=True)
    if location is None:
        assert isinstance(stimuli_train, pysaliency.Stimuli)
        assert not isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.Stimuli)
        assert not isinstance(stimuli_val, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.FileStimuli)
        assert location.join('COCO-Search18_no-task-merge_unique-images/stimuli_train.hdf5').check()
        assert location.join('COCO-Search18_no-task-merge_unique-images/stimuli_validation.hdf5').check()
        assert location.join('COCO-Search18_no-task-merge_unique-images/fixations_train.hdf5').check()
        assert location.join('COCO-Search18_no-task-merge_unique-images/fixations_validation.hdf5').check()

    assert len(stimuli_train) == 4326
    assert len(stimuli_val) == 652
    assert set(stimuli_train.sizes) == {(1050, 1680)}
    assert set(stimuli_val.sizes) == {(1050, 1680)}
    assert len(set(stimuli_train.stimulus_ids)) == 4326
    assert len(set(stimuli_val.stimulus_ids)) == 652

    assert 'task' in stimuli_train.__attributes__
    assert 'task' in stimuli_val.__attributes__
    assert len(np.unique(stimuli_train.attributes['task'])) == 18
    assert len(np.unique(stimuli_val.attributes['task'])) == 18
    assert set(stimuli_train.attributes['task']) == set(range(18))
    assert set(stimuli_val.attributes['task']) == set(range(18))

    assert len(fixations_train.x) == 207970

    assert np.mean(fixations_train.x) == approx(835.8440337548686)
    assert np.mean(fixations_train.y) == approx(509.6030908304083)
    assert np.mean(fixations_train.t) == approx(3.0987979035437805)
    assert np.mean(fixations_train.scanpath_history_length) == approx(3.0987979035437805)

    assert np.std(fixations_train.x) == approx(336.5760343388881)
    assert np.std(fixations_train.y) == approx(193.04654731407436)
    assert np.std(fixations_train.t) == approx(3.8411822348178664)
    assert np.std(fixations_train.scanpath_history_length) == approx(3.8411822348178664)

    assert kurtosis(fixations_train.x) == approx(-0.6283401149747818)
    assert kurtosis(fixations_train.y) == approx(0.15947671647330974)
    assert kurtosis(fixations_train.t) == approx(12.038491881119654)
    assert kurtosis(fixations_train.scanpath_history_length) == approx(12.038491881119654)

    assert skew(fixations_train.x) == approx(0.1706207784149093)
    assert skew(fixations_train.y) == approx(-0.07268825958515616)
    assert skew(fixations_train.t) == approx(2.804671690266736)
    assert skew(fixations_train.scanpath_history_length) == approx(2.804671690266736)

    assert entropy(fixations_train.n) == approx(11.967951796529752)
    assert (fixations_train.n == 0).sum() == 71

    assert len(fixations_val.x) == 31761

    assert np.mean(fixations_val.x) == approx(841.0752652624287)
    assert np.mean(fixations_val.y) == approx(498.3305594911999)
    assert np.mean(fixations_val.t) == approx(3.107994080790907)
    assert np.mean(fixations_val.scanpath_history_length) == approx(3.107994080790907)

    assert np.std(fixations_val.x) == approx(331.6328528765362)
    assert np.std(fixations_val.y) == approx(195.86110035077112)
    assert np.std(fixations_val.t) == approx(3.7502120687824454)
    assert np.std(fixations_val.scanpath_history_length) == approx(3.7502120687824454)

    assert kurtosis(fixations_val.x) == approx(-0.5973130907561486)
    assert kurtosis(fixations_val.y) == approx(-0.2797786304225598)
    assert kurtosis(fixations_val.t) == approx(11.250011182161305)
    assert kurtosis(fixations_val.scanpath_history_length) == approx(11.250011182161305)

    assert skew(fixations_val.x) == approx(0.14886675209256964)
    assert skew(fixations_val.y) == approx(-0.04086275403802345)
    assert skew(fixations_val.t) == approx(2.671653646130074)
    assert skew(fixations_val.scanpath_history_length) == approx(2.671653646130074)

    assert entropy(fixations_val.n) == approx(9.243197427307365)
    assert (fixations_val.n == 0).sum() == 42

    #assert len(fixations_train) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_train, fixations_train))
    #assert len(fixations_val) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_val, fixations_val))