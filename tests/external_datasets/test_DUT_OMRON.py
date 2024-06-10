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
def test_DUT_OMRON(location, tmpdir):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_DUT_OMRON(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('DUT-OMRON/stimuli.hdf5').check()
        assert location.join('DUT-OMRON/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 5168

    assert len(fixations.x) == 797542

    assert np.mean(fixations.x) == approx(182.16198519952553)
    assert np.mean(fixations.y) == approx(147.622566585835)
    assert np.mean(fixations.t) == approx(21.965026293286122)
    assert np.mean(fixations.scanpath_history_length) == approx(21.965026293286122)

    assert np.std(fixations.x) == approx(64.01040053828082)
    assert np.std(fixations.y) == approx(58.292098903584176)
    assert np.std(fixations.t) == approx(17.469479262739807)
    assert np.std(fixations.scanpath_history_length) == approx(17.469479262739807)

    assert kurtosis(fixations.x) == approx(-0.0689271960358524)
    assert kurtosis(fixations.y) == approx(0.637871926687533)
    assert kurtosis(fixations.t) == approx(2.914601085582113)
    assert kurtosis(fixations.scanpath_history_length) == approx(2.914601085582113)

    assert skew(fixations.x) == approx(0.23776167825897998)
    assert skew(fixations.y) == approx(0.6328497077003701)
    assert skew(fixations.t) == approx(1.2911168563657345)
    assert skew(fixations.scanpath_history_length) == approx(1.2911168563657345)

    assert entropy(fixations.n) == approx(12.20642017670851)
    assert (fixations.n == 0).sum() == 209

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))