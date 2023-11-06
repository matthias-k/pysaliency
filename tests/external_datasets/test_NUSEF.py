import numpy as np
import pytest
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
from tests.test_external_datasets import _location


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

    assert len(stimuli.stimuli) == 444

    assert len(fixations.x) == 71477

    assert np.mean(fixations.x) == approx(459.25215477028985)
    assert np.mean(fixations.y) == approx(337.7105607071453)
    assert np.mean(fixations.t) == approx(2.0601419197783906)
    assert np.mean(fixations.lengths) == approx(4.205604600081145)

    assert np.std(fixations.x) == approx(188.22873273245887)
    assert np.std(fixations.y) == approx(141.41626835405654)
    assert np.std(fixations.t) == approx(1.8835345300302346)
    assert np.std(fixations.lengths) == approx(3.5120574118479095)

    assert kurtosis(fixations.x) == approx(0.40246559702264895)
    assert kurtosis(fixations.y) == approx(2.0149558833607584)
    assert kurtosis(fixations.t) == approx(4500.149257624623)
    assert kurtosis(fixations.lengths) == approx(0.7152102743878679)

    assert skew(fixations.x) == approx(0.3652464937556074)
    assert skew(fixations.y) == approx(0.7127109189315761)
    assert skew(fixations.t) == approx(36.84824400914634)
    assert skew(fixations.lengths) == approx(0.9617232401848484)

    # there are images without any fixations
    #assert entropy(fixations.n) == approx(nan)
    assert (fixations.n == 0).sum() == 132

    # not testing this, there are many out-of-stimulus fixations in the dataset
    # assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))



