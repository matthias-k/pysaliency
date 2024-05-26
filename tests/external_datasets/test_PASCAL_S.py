import numpy as np
import pytest
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
import pysaliency.external_datasets
from tests.test_external_datasets import _location, entropy


@pytest.mark.slow
@pytest.mark.download
def test_PASCAL_S(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_PASCAL_S(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('PASCAL-S/stimuli.hdf5').check()
        assert location.join('PASCAL-S/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 850

    assert len(fixations.x) == 40314

    assert np.mean(fixations.x) == approx(240.72756362553952)
    assert np.mean(fixations.y) == approx(194.85756809048965)
    assert np.mean(fixations.t) == approx(2.7856823932132757)
    assert np.mean(fixations.scanpath_history_length) == approx(2.7856823932132757)

    assert np.std(fixations.x) == approx(79.57401169717699)
    assert np.std(fixations.y) == approx(65.21296890260112)
    assert np.std(fixations.t) == approx(2.1191752645988675)
    assert np.std(fixations.scanpath_history_length) == approx(2.1191752645988675)

    assert kurtosis(fixations.x) == approx(0.0009226786675387011)
    assert kurtosis(fixations.y) == approx(1.1907544566979986)
    assert kurtosis(fixations.t) == approx(-0.540943536495714)
    assert kurtosis(fixations.scanpath_history_length) == approx(-0.540943536495714)

    assert skew(fixations.x) == approx(0.2112334873314548)
    assert skew(fixations.y) == approx(0.7208733522533084)
    assert skew(fixations.t) == approx(0.4800678710338635)
    assert skew(fixations.scanpath_history_length) == approx(0.4800678710338635)

    assert entropy(fixations.n) == approx(9.711222735065062)
    assert (fixations.n == 0).sum() == 35

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))