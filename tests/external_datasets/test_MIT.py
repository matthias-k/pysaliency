import numpy as np
import pytest
from pathlib import Path
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
import pysaliency.external_datasets
from pysaliency.utils import remove_trailing_nans


from tests.test_external_datasets import _location, entropy


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
@pytest.mark.matlab
def test_mit1003(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_mit1003(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('MIT1003/stimuli.hdf5').check()
        assert location.join('MIT1003/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 1003
    for n in range(len(stimuli.stimuli)):
        assert max(stimuli.sizes[n]) == 1024

    assert len(fixations.x) == 104171

    assert np.mean(fixations.x) == approx(487.13683496521253)
    assert np.mean(fixations.y) == approx(392.72728829760155)
    assert np.mean(fixations.t) == approx(1.5039892740461995)
    assert np.mean(fixations.scanpath_history_length) == approx(3.3973754691804823)

    assert np.std(fixations.x) == approx(190.0203102093757)
    assert np.std(fixations.y) == approx(159.99210430350126)
    assert np.std(fixations.t) == approx(0.816414737693668)
    assert np.std(fixations.scanpath_history_length) == approx(2.5433689996843354)

    assert kurtosis(fixations.x) == approx(-0.39272472247196033)
    assert kurtosis(fixations.y) == approx(0.6983793465837596)
    assert kurtosis(fixations.t) == approx(-1.2178525798721818)
    assert kurtosis(fixations.scanpath_history_length) == approx(-0.45897225172578704)

    assert skew(fixations.x) == approx(0.2204976032609953)
    assert skew(fixations.y) == approx(0.6445191904777621)
    assert skew(fixations.t) == approx(0.08125182887100482)
    assert skew(fixations.scanpath_history_length) == approx(0.5047182860999948)

    assert entropy(fixations.n) == approx(9.954348058662386)
    assert (fixations.n == 0).sum() == 121

    assert 'duration_hist' in fixations.__attributes__
    assert 'duration' in fixations.__attributes__
    assert len(fixations.duration_hist) == len(fixations.x)
    assert len(fixations.duration) == len(fixations.x)
    for i in range(len(fixations.x)):
        assert len(remove_trailing_nans(fixations.duration_hist[i])) == len(remove_trailing_nans(fixations.x_hist[i]))


    assert 'durations' in fixations.scanpaths.fixation_attributes

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))

@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
@pytest.mark.matlab
def test_mit1003_onesize(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_mit1003_onesize(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('MIT1003_onesize/stimuli.hdf5').check()
        assert location.join('MIT1003_onesize/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 463
    for n in range(len(stimuli.stimuli)):
        assert stimuli.sizes[n] == (768, 1024)

    assert len(fixations.x) == 48771
    assert (fixations.n == 0).sum() == 121