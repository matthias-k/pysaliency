import numpy as np
import pytest
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
import pysaliency.external_datasets
from tests.test_external_datasets import entropy


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_stimuli(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    stimuli_train, stimuli_val, stimuli_test = pysaliency.external_datasets.salicon._get_SALICON_stimuli(location=real_location, name='SALICONfoobar')

    assert isinstance(stimuli_train, pysaliency.FileStimuli)
    assert isinstance(stimuli_val, pysaliency.FileStimuli)
    assert isinstance(stimuli_test, pysaliency.FileStimuli)
    assert location.join('SALICONfoobar/stimuli_train.hdf5').check()
    assert location.join('SALICONfoobar/stimuli_val.hdf5').check()
    assert location.join('SALICONfoobar/stimuli_test.hdf5').check()

    assert len(stimuli_train) == 10000
    assert len(stimuli_val) == 5000
    assert len(stimuli_test) == 5000

    assert set(stimuli_train.sizes) == set([(480, 640)])
    assert set(stimuli_val.sizes) == set([(480, 640)])
    assert set(stimuli_test.sizes) == set([(480, 640)])


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2015_mouse(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2015', fixation_type='mouse')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()
    assert isinstance(fixations_train, pysaliency.Fixations)
    assert not isinstance(fixations_train, pysaliency.FixationTrains)
    assert isinstance(fixations_val, pysaliency.Fixations)
    assert not isinstance(fixations_val, pysaliency.FixationTrains)

    assert len(fixations_train.x) == 68992355

    assert np.mean(fixations_train.x) == approx(313.0925573565361)
    assert np.mean(fixations_train.y) == approx(229.669921428251)
    assert np.mean(fixations_train.t) == approx(2453.3845915246698)
    assert np.mean(fixations_train.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_train.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_train.x) == approx(147.69997888974905)
    assert np.std(fixations_train.y) == approx(96.52066518492143)
    assert np.std(fixations_train.t) == approx(1538.7280458609941)

    assert kurtosis(fixations_train.x) == approx(-0.8543758617424033)
    assert kurtosis(fixations_train.y) == approx(-0.6277250557240337)
    assert kurtosis(fixations_train.t) == approx(19515.32829536525)

    assert skew(fixations_train.x) == approx(0.08274147964197842)
    assert skew(fixations_train.y) == approx(0.10465863071610296)
    assert skew(fixations_train.t) == approx(55.69180106087239)

    assert entropy(fixations_train.n) == approx(13.278169650429593)
    assert (fixations_train.n == 0).sum() == 6928


    assert len(fixations_val.x) == 38846998

    assert np.mean(fixations_val.x) == approx(311.44141923141655)
    assert np.mean(fixations_val.y) == approx(229.10522205602607)
    assert np.mean(fixations_val.t) == approx(2463.950701930687)
    assert np.mean(fixations_val.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_val.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_val.x) == approx(149.34417260369818)
    assert np.std(fixations_val.y) == approx(97.93170200208576)
    assert np.std(fixations_val.t) == approx(1408.3339394913962)

    assert kurtosis(fixations_val.x) == approx(-0.8449322083004356)
    assert kurtosis(fixations_val.y) == approx(-0.6136372253463405)
    assert kurtosis(fixations_val.t) == approx(-1.1157482867740718)

    assert skew(fixations_val.x) == approx(0.08926920530231194)
    assert skew(fixations_val.y) == approx(0.10168032060729842)
    assert skew(fixations_val.t) == approx(0.05444269756551158)

    assert entropy(fixations_val.n) == approx(12.279414832007888)
    assert (fixations_val.n == 0).sum() == 8244

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2015_fixations(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2015', fixation_type='fixations')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()
    assert isinstance(fixations_train, pysaliency.Fixations)
    assert not isinstance(fixations_train, pysaliency.FixationTrains)
    assert isinstance(fixations_val, pysaliency.Fixations)
    assert not isinstance(fixations_val, pysaliency.FixationTrains)


    assert len(fixations_train.x) == 3171533

    assert np.mean(fixations_train.x) == approx(310.93839540689)
    assert np.mean(fixations_train.y) == approx(217.7589979356986)
    assert np.mean(fixations_train.t) == approx(5.020693147446361)
    assert np.mean(fixations_train.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_train.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_train.x) == approx(131.0672366442846)
    assert np.std(fixations_train.y) == approx(86.33526319309237)
    assert np.std(fixations_train.t) == approx(5.2387518223254474)

    assert kurtosis(fixations_train.x) == approx(-0.6327397503173677)
    assert kurtosis(fixations_train.y) == approx(-0.3662318210834883)
    assert kurtosis(fixations_train.t) == approx(5.6123414320267795)

    assert skew(fixations_train.x) == approx(0.10139095797827476)
    assert skew(fixations_train.y) == approx(0.13853441448148346)
    assert skew(fixations_train.t) == approx(1.8891615714930796)

    assert entropy(fixations_train.n) == approx(13.22601241838667)
    assert (fixations_train.n == 0).sum() == 170


    assert len(fixations_val.x) == 1662655

    assert np.mean(fixations_val.x) == approx(308.64650213062845)
    assert np.mean(fixations_val.y) == approx(217.97772358065865)
    assert np.mean(fixations_val.t) == approx(4.808886389539622)
    assert np.mean(fixations_val.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_val.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_val.x) == approx(130.34460214133043)
    assert np.std(fixations_val.y) == approx(85.80831530782285)
    assert np.std(fixations_val.t) == approx(4.999870176048051)

    assert kurtosis(fixations_val.x) == approx(-0.5958648294721907)
    assert kurtosis(fixations_val.y) == approx(-0.31300073559578934)
    assert kurtosis(fixations_val.t) == approx(4.9489750451359225)

    assert skew(fixations_val.x) == approx(0.11714467225615313)
    assert skew(fixations_val.y) == approx(0.12631245881037118)
    assert skew(fixations_val.t) == approx(1.8301317514860862)

    assert entropy(fixations_val.n) == approx(12.234936723301066)
    assert (fixations_val.n == 0).sum() == 259

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2017_mouse(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2017', fixation_type='mouse')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()
    assert isinstance(fixations_train, pysaliency.Fixations)
    assert not isinstance(fixations_train, pysaliency.FixationTrains)
    assert isinstance(fixations_val, pysaliency.Fixations)
    assert not isinstance(fixations_val, pysaliency.FixationTrains)


    assert len(fixations_train.x) == 215286274

    assert np.mean(fixations_train.x) == approx(314.91750797871686)
    assert np.mean(fixations_train.y) == approx(232.38085973332957)
    assert np.mean(fixations_train.t) == approx(2541.6537073654777)
    assert np.mean(fixations_train.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_train.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_train.x) == approx(138.09403491170718)
    assert np.std(fixations_train.y) == approx(93.55417139372516)
    assert np.std(fixations_train.t) == approx(1432.604664553447)

    assert kurtosis(fixations_train.x) == approx(-0.8009690077811422)
    assert kurtosis(fixations_train.y) == approx(-0.638316482844866)
    assert kurtosis(fixations_train.t) == approx(6854.681620924244)

    assert skew(fixations_train.x) == approx(0.06734542626655958)
    assert skew(fixations_train.y) == approx(0.07252065918701057)
    assert skew(fixations_train.t) == approx(17.770454294178407)

    assert entropy(fixations_train.n) == approx(13.274472019581758)
    assert (fixations_train.n == 0).sum() == 24496


    assert len(fixations_val.x) == 121898426

    assert np.mean(fixations_val.x) == approx(313.3112383249313)
    assert np.mean(fixations_val.y) == approx(231.8708303160281)
    assert np.mean(fixations_val.t) == approx(2538.2123597970003)
    assert np.mean(fixations_val.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_val.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_val.x) == approx(139.30115624028937)
    assert np.std(fixations_val.y) == approx(95.24435516821612)
    assert np.std(fixations_val.t) == approx(1395.986706164002)

    assert kurtosis(fixations_val.x) == approx(-0.7932049483979013)
    assert kurtosis(fixations_val.y) == approx(-0.6316552996345393)
    assert kurtosis(fixations_val.t) == approx(-1.1483055562729023)

    assert skew(fixations_val.x) == approx(0.08023882420460927)
    assert skew(fixations_val.y) == approx(0.07703227629250083)
    assert skew(fixations_val.t) == approx(-0.0027158508337847653)

    assert entropy(fixations_val.n) == approx(12.278275960422771)
    assert (fixations_val.n == 0).sum() == 23961

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)


@pytest.mark.slow
@pytest.mark.download
def test_SALICON_fixations_2017_fixations(tmpdir):
    real_location = str(tmpdir)
    location = tmpdir

    fixations_train, fixations_val = pysaliency.external_datasets.salicon._get_SALICON_fixations(
		location=real_location, name='SALICONbar', edition='2017', fixation_type='fixations')

    assert location.join('SALICONbar/fixations_train.hdf5').check()
    assert location.join('SALICONbar/fixations_val.hdf5').check()
    assert isinstance(fixations_train, pysaliency.Fixations)
    assert not isinstance(fixations_train, pysaliency.FixationTrains)
    assert isinstance(fixations_val, pysaliency.Fixations)
    assert not isinstance(fixations_val, pysaliency.FixationTrains)

    assert len(fixations_train.x) == 4598112

    assert np.mean(fixations_train.x) == approx(314.62724265959594)
    assert np.mean(fixations_train.y) == approx(228.43566163677613)
    assert np.mean(fixations_train.t) == approx(4.692611228260643)
    assert np.mean(fixations_train.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_train.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_train.x) == approx(134.1455759990284)
    assert np.std(fixations_train.y) == approx(87.13212105359052)
    assert np.std(fixations_train.t) == approx(3.7300713016372375)

    assert kurtosis(fixations_train.x) == approx(-0.8163385970402013)
    assert kurtosis(fixations_train.y) == approx(-0.615440115290188)
    assert kurtosis(fixations_train.t) == approx(0.7328902767227148)

    assert skew(fixations_train.x) == approx(0.07523280051849487)
    assert skew(fixations_train.y) == approx(0.0854479359829959)
    assert skew(fixations_train.t) == approx(0.8951438604006022)

    assert entropy(fixations_train.n) == approx(13.26103635730998)
    assert (fixations_train.n == 0).sum() == 532


    assert len(fixations_val.x) == 2576914

    assert np.mean(fixations_val.x) == approx(312.8488630198757)
    assert np.mean(fixations_val.y) == approx(227.6883237081253)
    assert np.mean(fixations_val.t) == approx(4.889936955598829)
    assert np.mean(fixations_val.scanpath_history_length) == approx(0.0)
    assert np.max(fixations_val.scanpath_history_length) == approx(0.0)

    assert np.std(fixations_val.x) == approx(133.22242352479964)
    assert np.std(fixations_val.y) == approx(86.71553440419093)
    assert np.std(fixations_val.t) == approx(3.9029124873868466)

    assert kurtosis(fixations_val.x) == approx(-0.7961636859307624)
    assert kurtosis(fixations_val.y) == approx(-0.5897615692354612)
    assert kurtosis(fixations_val.t) == approx(0.7766482713546012)

    assert skew(fixations_val.x) == approx(0.08676607299583787)
    assert skew(fixations_val.y) == approx(0.08801482949432776)
    assert skew(fixations_val.t) == approx(0.9082922185416067)

    assert entropy(fixations_val.n) == approx(12.259608288646687)
    assert (fixations_val.n == 0).sum() == 593

    assert np.all(fixations_train.x >= 0)
    assert np.all(fixations_train.y >= 0)
    assert np.all(fixations_val.x >= 0)
    assert np.all(fixations_val.y >= 0)
    assert np.all(fixations_train.x < 640)
    assert np.all(fixations_train.y < 480)
    assert np.all(fixations_val.x < 640)
    assert np.all(fixations_val.y < 480)