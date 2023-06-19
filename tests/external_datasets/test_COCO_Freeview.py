import numpy as np
import pytest
from pytest import approx
import pysaliency
from scipy.stats import kurtosis, skew

from tests.test_external_datasets import _location, entropy


@pytest.mark.slow
@pytest.mark.download
def test_COCO_Freeview(location):
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_val, fixations_val = pysaliency.external_datasets.get_COCO_Freeview(location=real_location)
    if location is None:
        assert isinstance(stimuli_train, pysaliency.Stimuli)
        assert not isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.Stimuli)
        assert not isinstance(stimuli_val, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_val, pysaliency.FileStimuli)
        assert location.join('COCO-Freeview/stimuli_train.hdf5').check()
        assert location.join('COCO-Freeview/stimuli_validation.hdf5').check()
        assert location.join('COCO-Freeview/fixations_train.hdf5').check()
        assert location.join('COCO-Freeview/fixations_validation.hdf5').check()

    assert len(stimuli_train) == 3714
    assert len(stimuli_val) == 623
    assert set(stimuli_train.sizes) == {(1050, 1680)}
    assert set(stimuli_val.sizes) == {(1050, 1680)}

    assert len(fixations_train.x) == 667428

    assert np.mean(fixations_train.x) == approx(855.0507976890392)
    assert np.mean(fixations_train.y) == approx(519.6208629245402)
    assert np.mean(fixations_train.t) == approx(7.575617145220159)
    assert np.mean(fixations_train.lengths) == approx(7.575617145220159)

    assert np.std(fixations_train.x) == approx(296.94267824321696)
    assert np.std(fixations_train.y) == approx(181.42993314294952)
    assert np.std(fixations_train.t) == approx(4.956080545631881)
    assert np.std(fixations_train.lengths) == approx(4.956080545631881)

    assert kurtosis(fixations_train.x) == approx(-0.4800071906527137)
    assert kurtosis(fixations_train.y) == approx(-0.16985576087243315)
    assert kurtosis(fixations_train.t) == approx(-0.7961088597233026)
    assert kurtosis(fixations_train.lengths) == approx(-0.7961088597233026)

    assert skew(fixations_train.x) == approx(0.05151289244179072)
    assert skew(fixations_train.y) == approx(0.12265040006978992)
    assert skew(fixations_train.t) == approx(0.2775958921822995)
    assert skew(fixations_train.lengths) == approx(0.2775958921822995)

    assert entropy(fixations_train.n) == approx(11.775330967227847)
    assert (fixations_train.n == 0).sum() == 165

    # Validation

    assert len(fixations_val.x) == 100391

    assert np.mean(fixations_val.x) == approx(859.6973842276699)
    assert np.mean(fixations_val.y) == approx(519.1442987917244)
    assert np.mean(fixations_val.t) == approx(7.561614088912353)
    assert np.mean(fixations_val.lengths) == approx(7.561614088912353)

    assert np.std(fixations_val.x) == approx(298.007469111755)
    assert np.std(fixations_val.y) == approx(183.67581178519256)
    assert np.std(fixations_val.t) == approx(4.948216910636096)
    assert np.std(fixations_val.lengths) == approx(4.948216910636096)

    assert kurtosis(fixations_val.x) == approx(-0.48170986922459846)
    assert kurtosis(fixations_val.y) == approx(-0.24935255041328297)
    assert kurtosis(fixations_val.t) == approx(-0.7699148004968688)
    assert kurtosis(fixations_val.lengths) == approx(-0.7699148004968688)

    assert skew(fixations_val.x) == approx(0.026197404490588)
    assert skew(fixations_val.y) == approx(0.10752860025117382)
    assert skew(fixations_val.t) == approx(0.2834855455561754)
    assert skew(fixations_val.lengths) == approx(0.2834855455561754)

    assert entropy(fixations_val.n) == approx(9.254923983126101)
    assert (fixations_val.n == 0).sum() == 155


    #assert len(fixations_train) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_train, fixations_train))
    #assert len(fixations_val) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_val, fixations_val))