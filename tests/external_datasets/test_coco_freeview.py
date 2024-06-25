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

    stimuli_train, fixations_train, stimuli_val, fixations_val, stimuli_test = pysaliency.external_datasets.get_COCO_Freeview(location=real_location)
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
    assert len(stimuli_val) == 603
    assert len(stimuli_test) == 1000
    assert set(stimuli_train.sizes) == {(1050, 1680)}
    assert set(stimuli_val.sizes) == {(1050, 1680)}
    assert set(stimuli_test.sizes) == {(1050, 1680)}

    assert len(fixations_train.x) == 572184

    assert np.mean(fixations_train.x) == approx(854.6399011506788)
    assert np.mean(fixations_train.y) == approx(520.0318222809445)
    assert np.mean(fixations_train.t) == approx(7.568133677278638)
    assert np.mean(fixations_train.scanpath_history_length) == approx(7.568133677278638)

    assert np.std(fixations_train.x) == approx(296.0191172854278)
    assert np.std(fixations_train.y) == approx(181.3128347366162)
    assert np.std(fixations_train.t) == approx(4.9536161050175025)
    assert np.std(fixations_train.scanpath_history_length) == approx(4.9536161050175025)

    assert kurtosis(fixations_train.x) == approx(-0.4658856837827998)
    assert kurtosis(fixations_train.y) == approx(-0.17242182386194793)
    assert kurtosis(fixations_train.t) == approx(-0.7932601698667865)
    assert kurtosis(fixations_train.scanpath_history_length) == approx(-0.7932601698667865)

    assert skew(fixations_train.x) == approx(0.04888106495259364)
    assert skew(fixations_train.y) == approx(0.1217343831850603)
    assert skew(fixations_train.t) == approx(0.2791201142040311)
    assert skew(fixations_train.scanpath_history_length) == approx(0.2791201142040311)

    assert entropy(fixations_train.n) == approx(11.853219537063737)
    assert (fixations_train.n == 0).sum() == 165

    # Validation

    assert len(fixations_val.x) == 92821

    assert np.mean(fixations_val.x) == approx(858.7499983839865)
    assert np.mean(fixations_val.y) == approx(519.7572176554874)
    assert np.mean(fixations_val.t) == approx(7.561747880328805)
    assert np.mean(fixations_val.scanpath_history_length) == approx(7.561747880328805)

    assert np.std(fixations_val.x) == approx(298.68282356632267)
    assert np.std(fixations_val.y) == approx(184.22406748940242)
    assert np.std(fixations_val.t) == approx(4.950144502725075)
    assert np.std(fixations_val.scanpath_history_length) == approx(4.950144502725075)

    assert kurtosis(fixations_val.x) == approx(-0.48168521133038)
    assert kurtosis(fixations_val.y) == approx(-0.25828026864894804)
    assert kurtosis(fixations_val.t) == approx(-0.7630800100767541)
    assert kurtosis(fixations_val.scanpath_history_length) == approx(-0.7630800100767541)

    assert skew(fixations_val.x) == approx(0.03072935717644178)
    assert skew(fixations_val.y) == approx(0.1086910594402604)
    assert skew(fixations_val.t) == approx(0.28569302638044036)
    assert skew(fixations_val.scanpath_history_length) == approx(0.28569302638044036)

    assert entropy(fixations_val.n) == approx(9.230606964850315)
    assert (fixations_val.n == 0).sum() == 155


    #assert len(fixations_train) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_train, fixations_train))
    #assert len(fixations_val) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli_val, fixations_val))