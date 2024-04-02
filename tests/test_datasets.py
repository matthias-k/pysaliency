import numpy as np
import pytest
from imageio import imwrite

import pysaliency


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