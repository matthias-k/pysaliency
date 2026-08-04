import numpy as np
import pytest
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
from tests.test_external_datasets import _location, entropy


@pytest.mark.slow
@pytest.mark.download
def test_DAEMONS(location):
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_validation, fixations_validation, stimuli_test = \
        pysaliency.external_datasets.get_DAEMONS(location=real_location)

    if location is None:
        assert isinstance(stimuli_train, pysaliency.Stimuli)
        assert not isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_validation, pysaliency.Stimuli)
        assert not isinstance(stimuli_validation, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli_train, pysaliency.FileStimuli)
        assert isinstance(stimuli_validation, pysaliency.FileStimuli)
        assert location.join('DAEMONS/stimuli_train.hdf5').check()
        assert location.join('DAEMONS/stimuli_validation.hdf5').check()
        assert location.join('DAEMONS/stimuli_test.hdf5').check()
        assert location.join('DAEMONS/fixations_train.hdf5').check()
        assert location.join('DAEMONS/fixations_validation.hdf5').check()
        # the scanpath archive is kept: it is the only place the column documentation exists
        assert location.join('DAEMONS/src/eye_movement.zip').check()
        # the test scanpaths are withheld, so no test fixations without `test_data`
        assert not location.join('DAEMONS/fixations_test.hdf5').check()

    # train
    assert len(stimuli_train) == 2000
    assert set(stimuli_train.sizes) == {(1080, 1920)}
    assert len(fixations_train.x) == 492367
    assert len(fixations_train.scanpaths) == 19994

    assert np.mean(fixations_train.x) == approx(970.4675494139849)
    assert np.mean(fixations_train.y) == approx(518.878915815939)
    assert np.mean(fixations_train.t) == approx(3.915599739625117)
    assert np.mean(fixations_train.scanpath_history_length) == approx(12.251477048624299)

    assert np.std(fixations_train.x) == approx(452.30634804673684)
    assert np.std(fixations_train.y) == approx(235.73431829076085)
    assert np.std(fixations_train.t) == approx(2.4007398204914177)
    assert np.std(fixations_train.scanpath_history_length) == approx(7.767515307982913)

    assert kurtosis(fixations_train.x) == approx(-0.8289866186030728)
    assert kurtosis(fixations_train.y) == approx(-0.6118464217644393)
    assert kurtosis(fixations_train.t) == approx(-1.1973336001082964)
    assert kurtosis(fixations_train.scanpath_history_length) == approx(-0.8761522613792359)

    assert skew(fixations_train.x) == approx(0.04779524888689353)
    assert skew(fixations_train.y) == approx(0.09605387456738178)
    assert skew(fixations_train.t) == approx(0.03675480059222509)
    assert skew(fixations_train.scanpath_history_length) == approx(0.23957753321683498)

    assert entropy(fixations_train.n) == approx(10.961176405368043)
    assert (fixations_train.n == 0).sum() == 265

    # validation
    assert len(stimuli_validation) == 200
    assert set(stimuli_validation.sizes) == {(1080, 1920)}
    assert len(fixations_validation.x) == 247042
    assert len(fixations_validation.scanpaths) == 9997

    assert np.mean(fixations_validation.x) == approx(975.5040282161087)
    assert np.mean(fixations_validation.y) == approx(513.185525755272)
    assert np.mean(fixations_validation.t) == approx(3.9240375644627226)
    assert np.mean(fixations_validation.scanpath_history_length) == approx(12.278993045716923)

    assert np.std(fixations_validation.x) == approx(455.4877383380231)
    assert np.std(fixations_validation.y) == approx(234.55043965651728)
    assert np.std(fixations_validation.t) == approx(2.4040259653693767)
    assert np.std(fixations_validation.scanpath_history_length) == approx(7.770843395994464)

    assert kurtosis(fixations_validation.x) == approx(-0.8849513566035077)
    assert kurtosis(fixations_validation.y) == approx(-0.6215560722632905)
    assert kurtosis(fixations_validation.t) == approx(-1.20034246303119)
    assert kurtosis(fixations_validation.scanpath_history_length) == approx(-0.8875599533987018)

    assert skew(fixations_validation.x) == approx(0.03096408331607252)
    assert skew(fixations_validation.y) == approx(0.1432487176081993)
    assert skew(fixations_validation.t) == approx(0.0321501556194764)
    assert skew(fixations_validation.scanpath_history_length) == approx(0.23281302779707708)

    assert entropy(fixations_validation.n) == approx(7.641824191116513)
    assert (fixations_validation.n == 0).sum() == 1183

    # test stimuli are public even though the test scanpaths are not
    assert len(stimuli_test) == 200
    assert set(stimuli_test.sizes) == {(1080, 1920)}

    # About 8% of trials contain more than one forced fixation, because observers make small
    # saccades within the forced fixation period. Code that treats the forced fixation as a
    # scanpath's initial condition has to use the attribute rather than assume it is the first
    # fixation only.
    forced_per_scanpath = np.array([
        np.nansum(fixations_train.scanpaths.fixation_attributes['forced_fixations'][index])
        for index in range(len(fixations_train.scanpaths))
    ])
    assert forced_per_scanpath.min() == 1
    assert forced_per_scanpath.max() > 1

    # scanpath attributes that only exist in the public release
    assert 'mask_shown' in fixations_train.scanpaths.scanpath_attributes
    assert 'forced_end' in fixations_train.scanpaths.scanpath_attributes


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.nonfree
def test_DAEMONS_test_split(location):
    """The test scanpaths are withheld for a benchmark and need the non-public saccade tables."""
    real_location = _location(location)

    stimuli_train, fixations_train, stimuli_validation, fixations_validation, stimuli_test = \
        pysaliency.external_datasets.get_DAEMONS(
            location=real_location, test_data='ThirdParty/DAEMONS_final_sac.zip')

    assert location.join('DAEMONS/fixations_test.hdf5').check()
    # the caller's non-public file is kept alongside the built dataset
    assert location.join('DAEMONS/src/DAEMONS_final_sac.zip').check()

    fixations_test = pysaliency.read_hdf5(str(location.join('DAEMONS/fixations_test.hdf5')))

    assert len(fixations_test.x) == 245307
    assert len(fixations_test.scanpaths) == 9997

    assert np.mean(fixations_test.x) == approx(980.2026144420482)
    assert np.mean(fixations_test.y) == approx(516.547855273722)
    assert np.mean(fixations_test.t) == approx(3.920090665981811)

    assert np.std(fixations_test.x) == approx(446.5268414421781)
    assert np.std(fixations_test.y) == approx(237.90957954317085)
    assert np.std(fixations_test.t) == approx(2.400049124254272)

    assert entropy(fixations_test.n) == approx(7.641483886095785)

    # the withheld split covers exactly the stimuli the public loader reports as test stimuli
    assert len(stimuli_test) == 200
