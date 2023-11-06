import numpy as np
import pytest
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
from tests.test_external_datasets import _location, entropy


@pytest.mark.slow
@pytest.mark.download
def test_NUSEF(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_NUSEF_public(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('NUSEF_public/stimuli.hdf5').check()
        assert location.join('NUSEF_public/fixations.hdf5').check()
        assert location.join('NUSEF_public/src/NUSEF_database.zip').check()

    assert len(stimuli.stimuli) == 429

    assert len(fixations.x) == 66133

    assert np.mean(fixations.x) == approx(452.88481928283653)
    assert np.mean(fixations.y) == approx(337.03301271592267)
    assert np.mean(fixations.t) == approx(2.0420471776571456)
    assert np.mean(fixations.lengths) == approx(4.085887529675049)

    assert np.std(fixations.x) == approx(187.61359889152612)
    assert np.std(fixations.y) == approx(142.59867038067452)
    assert np.std(fixations.t) == approx(1.82140623534086)
    assert np.std(fixations.lengths) == approx(3.4339653884944963)

    assert kurtosis(fixations.x) == approx(0.403419633086465)
    assert kurtosis(fixations.y) == approx(2.0001760382566793)
    assert kurtosis(fixations.t) == approx(5285.812604733467)
    assert kurtosis(fixations.lengths) == approx(0.8320210638515699)

    assert skew(fixations.x) == approx(0.42747360917257937)
    assert skew(fixations.y) == approx(0.7441609934536769)
    assert skew(fixations.t) == approx(39.25751334379433)
    assert skew(fixations.lengths) == approx(0.9874139139443956)

    assert entropy(fixations.n) == approx(8.603204478724775)
    assert (fixations.n == 0).sum() == 132

    # not testing this, there are many out-of-stimulus fixations in the dataset
    # assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))



