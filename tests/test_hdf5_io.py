import numpy as np

import pysaliency
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel


def test_unified_read_hdf5_reads_dataset_files(tmp_path):
    stimuli = pysaliency.Stimuli([np.random.randn(12, 10, 3), np.random.randn(8, 9, 3)])
    path = tmp_path / 'stimuli.hdf5'
    stimuli.to_hdf5(path)

    loaded = pysaliency.read_hdf5(path)
    assert isinstance(loaded, pysaliency.Stimuli)
    assert len(loaded) == len(stimuli)
    np.testing.assert_array_equal(loaded[0].stimulus_data, stimuli[0].stimulus_data)


def test_unified_read_hdf5_reads_baseline_model_files(tmp_path):
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = BaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'baseline_model.hdf5'
    model.to_hdf5(path)

    loaded = pysaliency.read_hdf5(path)
    assert isinstance(loaded, BaselineModel)
    np.testing.assert_allclose(loaded.log_density(stimuli[0]), model.log_density(stimuli[0]))


def test_legacy_datasets_read_hdf5_wrapper_still_works(tmp_path):
    stimuli = pysaliency.Stimuli([np.random.randn(15, 10, 3)])
    path = tmp_path / 'stimuli_legacy.hdf5'
    stimuli.to_hdf5(path)

    loaded = pysaliency.datasets.read_hdf5(path)
    assert isinstance(loaded, pysaliency.Stimuli)
    np.testing.assert_array_equal(loaded[0].stimulus_data, stimuli[0].stimulus_data)


def test_unified_read_hdf5_reads_crossvalidated_baseline_model(tmp_path):
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_model.hdf5'
    model.to_hdf5(path)

    loaded = pysaliency.read_hdf5(path)
    assert isinstance(loaded, CrossvalidatedBaselineModel)
    np.testing.assert_allclose(loaded.log_density(stimuli[0]), model.log_density(stimuli[0]))


def test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg(tmp_path):
    """stimuli= kwarg is forwarded correctly through the dispatcher."""
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_no_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=False)

    loaded = pysaliency.read_hdf5(str(path), stimuli=stimuli)
    assert isinstance(loaded, CrossvalidatedBaselineModel)
    assert loaded.stimuli is stimuli
    np.testing.assert_allclose(loaded.log_density(stimuli[0]), model.log_density(stimuli[0]))


def test_unified_read_hdf5_crossvalidated_baseline_model_stimuli_kwarg_bypasses_cache(tmp_path):
    """Regression: stimuli= kwarg must bypass WeakValueDictionary cache in _read_hdf5_from_file."""
    stimuli = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    fixations = pysaliency.FixationTrains.from_fixation_trains(
        [[1, 2, 3], [4, 8]],
        [[5, 6, 7], [9, 2]],
        [[0, 1, 2], [0, 1]],
        [0, 1],
        [0, 1],
    )
    model = CrossvalidatedBaselineModel(stimuli, fixations, bandwidth=0.1)
    path = tmp_path / 'cv_baseline_no_stimuli.hdf5'
    model.to_hdf5(path, include_stimuli=False)

    loaded = pysaliency.read_hdf5(str(path), stimuli=stimuli)
    assert loaded.stimuli is stimuli

    # Second call with a different stimuli object — must not return the cached first result
    stimuli2 = pysaliency.Stimuli([np.random.randn(20, 20, 3), np.random.randn(20, 20, 3)])
    loaded2 = pysaliency.read_hdf5(str(path), stimuli=stimuli2)
    assert loaded2.stimuli is stimuli2
    np.testing.assert_allclose(loaded2.log_density(stimuli2[0]), CrossvalidatedBaselineModel(stimuli2, fixations, bandwidth=0.1).log_density(stimuli2[0]))
