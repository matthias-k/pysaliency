import numpy as np
import pytest
from pathlib import Path
from pytest import approx
from scipy.stats import kurtosis, skew

import pysaliency
import pysaliency.external_datasets
from pysaliency.utils import remove_trailing_nans


def _location(location):
    if location is not None:
        return str(location)
    return location


def entropy(labels):
    counts = np.bincount(labels)
    weights = counts / np.sum(counts)
    return -np.sum(weights*np.log(weights)) / np.log(2)


@pytest.mark.slow
@pytest.mark.download
def test_toronto(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_toronto(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('toronto/stimuli.hdf5').check()
        assert location.join('toronto/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 120
    for n in range(len(stimuli.stimuli)):
        assert stimuli.shapes[n] == (511, 681, 3)
        assert stimuli.sizes[n] == (511, 681)

    assert len(fixations.x) == 11199

    assert np.mean(fixations.x) == approx(345.7466738101616)
    assert np.mean(fixations.y) == approx(244.11393874453077)
    assert np.mean(fixations.t) == approx(0.0)
    assert np.mean(fixations.scanpath_history_length) == approx(0.0)

    assert np.std(fixations.x) == approx(132.7479359296397)
    assert np.std(fixations.y) == approx(82.89667109045186)
    assert np.std(fixations.t) == approx(0.0)
    assert np.std(fixations.scanpath_history_length) == approx(0.0)

    assert kurtosis(fixations.x) == approx(-0.40985986581959066)
    assert kurtosis(fixations.y) == approx(0.2748036777667475)
    assert fixations.t.std() == approx(0)
    assert fixations.scanpath_history_length.std() == approx(0)

    assert skew(fixations.x) == approx(-0.09509166105451604)
    assert skew(fixations.y) == approx(-0.08674038899319877)

    assert entropy(fixations.n) == approx(6.8939709237615405)
    assert (np.array(fixations.n) == 0).sum() == 130

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.matlab
@pytest.mark.skip_octave
def test_cat2000_train_v1_0(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_cat2000_train(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('CAT2000_train/stimuli.hdf5').check()
        assert location.join('CAT2000_train/fixations.hdf5').check()
        assert not list ((Path(location) / 'CAT2000_train' / 'Stimuli').glob('**/Output'))
        assert not list ((Path(location) / 'CAT2000_train' / 'Stimuli').glob('**/*_SaliencyMap.jpg'))

    assert len(stimuli.stimuli) == 2000
    assert set(stimuli.sizes) == {(1080, 1920)}
    assert set(stimuli.attributes.keys()) == {'category'}
    assert np.all(np.array(stimuli.attributes['category'][0:100]) == 0)
    assert np.all(np.array(stimuli.attributes['category'][100:200]) == 1)

    assert len(fixations.x) == 672053

    assert np.mean(fixations.x) == approx(977.9570036886972)
    assert np.mean(fixations.y) == approx(536.8014098590438)
    assert np.mean(fixations.t) == approx(10.95831429961625)
    assert np.mean(fixations.scanpath_history_length) == approx(9.95831429961625)

    assert np.std(fixations.x) == approx(265.521305397389)
    assert np.std(fixations.y) == approx(200.3874894751514)
    assert np.std(fixations.t) == approx(6.881491455270027)
    assert np.std(fixations.scanpath_history_length) == approx(6.881491455270027)

    assert kurtosis(fixations.x) == approx(0.8377433175079028)
    assert kurtosis(fixations.y) == approx(0.15890436764279947)
    assert kurtosis(fixations.t) == approx(0.08351046096368542)
    assert kurtosis(fixations.scanpath_history_length) == approx(0.08351046096368542)

    assert skew(fixations.x) == approx(0.07428576098144545)
    assert skew(fixations.y) == approx(0.27425191693049106)
    assert skew(fixations.t) == approx(0.5874222148956657)
    assert skew(fixations.scanpath_history_length) == approx(0.5874222148956657)

    assert entropy(fixations.n) == approx(10.955266908462857)
    assert (fixations.n == 0).sum() == 307

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.matlab
@pytest.mark.skip_octave
def test_cat2000_train_v1_1(location, matlab):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_cat2000_train(location=real_location, version='1.1')

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('CAT2000_train_v1.1/stimuli.hdf5').check()
        assert location.join('CAT2000_train_v1.1/fixations.hdf5').check()
        assert not list ((Path(location) / 'CAT2000_train_v1.1' / 'Stimuli').glob('**/Output'))
        assert not list ((Path(location) / 'CAT2000_train_v1.1' / 'Stimuli').glob('**/*_SaliencyMap.jpg'))

    assert len(stimuli.stimuli) == 2000
    assert set(stimuli.sizes) == {(1080, 1920)}
    assert set(stimuli.attributes.keys()) == {'category'}
    assert np.all(np.array(stimuli.attributes['category'][0:100]) == 0)
    assert np.all(np.array(stimuli.attributes['category'][100:200]) == 1)

    assert len(fixations.x) == 667804

    assert np.mean(fixations.x) == approx(977.048229720098)
    assert np.mean(fixations.y) == approx(535.7335899455527)
    assert np.mean(fixations.t) == approx(10.888694886523592)
    assert np.mean(fixations.scanpath_history_length) == approx(9.888694886523592)

    assert np.std(fixations.x) == approx(265.7561897117776)
    assert np.std(fixations.y) == approx(200.47021508760227)
    assert np.std(fixations.t) == approx(6.8276447542371805)
    assert np.std(fixations.scanpath_history_length) == approx(6.8276447542371805)

    assert kurtosis(fixations.x) == approx(0.8314129075001575)
    assert kurtosis(fixations.y) == approx(0.16001475266665466)
    assert kurtosis(fixations.t) == approx(0.07131517526032427)
    assert kurtosis(fixations.scanpath_history_length) == approx(0.07131517526032427)

    assert skew(fixations.x) == approx(0.07615972876511597)
    assert skew(fixations.y) == approx(0.2770231691322164)
    assert skew(fixations.t) == approx(0.5813051491385639)
    assert skew(fixations.scanpath_history_length) == approx(0.5813051491385639)

    assert entropy(fixations.n) == approx(10.955097604631638)
    assert (fixations.n == 0).sum() == 304

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
def test_cat2000_test(location):
    real_location = _location(location)

    stimuli = pysaliency.external_datasets.get_cat2000_test(location=real_location)

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('CAT2000_test/stimuli.hdf5').check()
        assert not list ((Path(location) / 'CAT2000_test' / 'Stimuli').glob('**/Output'))
        assert not list ((Path(location) / 'CAT2000_test' / 'Stimuli').glob('**/*_SaliencyMap.jpg'))


    assert len(stimuli.stimuli) == 2000
    assert set(stimuli.sizes) == {(1080, 1920)}
    assert set(stimuli.attributes.keys()) == {'category'}
    assert np.all(np.array(stimuli.attributes['category'][0:100]) == 0)
    assert np.all(np.array(stimuli.attributes['category'][100:200]) == 1)


@pytest.mark.slow
@pytest.mark.download
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


@pytest.mark.slow
@pytest.mark.nonfree
@pytest.mark.skip_octave
def test_koehler(location):
    real_location = _location(location)

    stimuli, fixations_freeviewing, fixations_objectsearch, fixations_saliencysearch \
        = pysaliency.external_datasets.get_koehler(location=real_location, datafile='ThirdParty/Koehler_PublicData.zip')

    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('Koehler/stimuli.hdf5').check()
        assert location.join('Koehler/fixations_freeviewing.hdf5').check()
        assert location.join('Koehler/fixations_objectsearch.hdf5').check()
        assert location.join('Koehler/fixations_saliencysearch.hdf5').check()

    assert len(stimuli.stimuli) == 800
    assert set(stimuli.sizes) == {(405, 405)}

    assert len(fixations_freeviewing.x) == 94600

    assert np.mean(fixations_freeviewing.x) == approx(205.99696617336153)
    assert np.mean(fixations_freeviewing.y) == approx(190.69461945031713)
    assert np.mean(fixations_freeviewing.t) == approx(2.6399788583509514)
    assert np.mean(fixations_freeviewing.scanpath_history_length) == approx(2.6399788583509514)

    assert np.std(fixations_freeviewing.x) == approx(86.7891146642233)
    assert np.std(fixations_freeviewing.y) == approx(67.11495414894833)
    assert np.std(fixations_freeviewing.t) == approx(2.1333811216291982)
    assert np.std(fixations_freeviewing.scanpath_history_length) == approx(2.1333811216291982)

    assert kurtosis(fixations_freeviewing.x) == approx(-0.6927977738542421)
    assert kurtosis(fixations_freeviewing.y) == approx(0.26434562598200007)
    assert kurtosis(fixations_freeviewing.t) == approx(1.0000780305443921)
    assert kurtosis(fixations_freeviewing.scanpath_history_length) == approx(1.0000780305443921)

    assert skew(fixations_freeviewing.x) == approx(0.04283261395632401)
    assert skew(fixations_freeviewing.y) == approx(0.15277972804817913)
    assert skew(fixations_freeviewing.t) == approx(0.8569222723327634)
    assert skew(fixations_freeviewing.scanpath_history_length) == approx(0.8569222723327634)

    assert entropy(fixations_freeviewing.n) == approx(9.638058218898772)
    assert (fixations_freeviewing.n == 0).sum() == 128

    assert len(fixations_objectsearch.x) == 125293

    assert np.mean(fixations_objectsearch.x) == approx(199.05052955871437)
    assert np.mean(fixations_objectsearch.y) == approx(202.8867534499134)
    assert np.mean(fixations_objectsearch.t) == approx(3.9734302794250276)
    assert np.mean(fixations_objectsearch.scanpath_history_length) == approx(3.9734302794250276)

    assert np.std(fixations_objectsearch.x) == approx(88.10778886056328)
    assert np.std(fixations_objectsearch.y) == approx(65.29208873896408)
    assert np.std(fixations_objectsearch.t) == approx(2.902206977368411)
    assert np.std(fixations_objectsearch.scanpath_history_length) == approx(2.902206977368411)

    assert kurtosis(fixations_objectsearch.x) == approx(-0.49120093084140537)
    assert kurtosis(fixations_objectsearch.y) == approx(0.625841808353278)
    assert kurtosis(fixations_objectsearch.t) == approx(-0.33967380087822274)
    assert kurtosis(fixations_objectsearch.scanpath_history_length) == approx(-0.33967380087822274)

    assert skew(fixations_objectsearch.x) == approx(0.12557741560793217)
    assert skew(fixations_objectsearch.y) == approx(-0.005003252610602025)
    assert skew(fixations_objectsearch.t) == approx(0.5297789219605314)
    assert skew(fixations_objectsearch.scanpath_history_length) == approx(0.5297789219605314)

    assert entropy(fixations_objectsearch.n) == approx(9.637156128022387)
    assert (fixations_objectsearch.n == 0).sum() == 140

    assert len(fixations_saliencysearch.x) == 94528

    assert np.mean(fixations_saliencysearch.x) == approx(203.7605894549763)
    assert np.mean(fixations_saliencysearch.y) == approx(193.67308099187542)
    assert np.mean(fixations_saliencysearch.t) == approx(2.7536814488828707)
    assert np.mean(fixations_saliencysearch.scanpath_history_length) == approx(2.7536814488828707)

    assert np.std(fixations_saliencysearch.x) == approx(94.18304559956722)
    assert np.std(fixations_saliencysearch.y) == approx(65.3335501279418)
    assert np.std(fixations_saliencysearch.t) == approx(2.114709138575087)
    assert np.std(fixations_saliencysearch.scanpath_history_length) == approx(2.114709138575087)

    assert kurtosis(fixations_saliencysearch.x) == approx(-0.9085078389136778)
    assert kurtosis(fixations_saliencysearch.y) == approx(0.319385892621745)
    assert kurtosis(fixations_saliencysearch.t) == approx(-0.06720050297739633)
    assert kurtosis(fixations_saliencysearch.scanpath_history_length) == approx(-0.06720050297739633)

    assert skew(fixations_saliencysearch.x) == approx(0.0019227173784863957)
    assert skew(fixations_saliencysearch.y) == approx(0.05728474858602427)
    assert skew(fixations_saliencysearch.t) == approx(0.5866228411986677)
    assert skew(fixations_saliencysearch.scanpath_history_length) == approx(0.5866228411986677)

    assert entropy(fixations_saliencysearch.n) == approx(9.639365034197382)
    assert (fixations_saliencysearch.n == 0).sum() == 103

    assert len(fixations_freeviewing) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations_freeviewing))
    assert len(fixations_objectsearch) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations_objectsearch))
    assert len(fixations_saliencysearch) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations_saliencysearch))


@pytest.mark.slow
@pytest.mark.download
def test_FIGRIM(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_FIGRIM(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('FIGRIM/stimuli.hdf5').check()
        assert location.join('FIGRIM/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 2793
    assert set(stimuli.sizes) == {(1000, 1000)}

    assert len(fixations.x) == 424712

    assert np.mean(fixations.x) == approx(504.64437312814323)
    assert np.mean(fixations.y) == approx(512.8821982896645)
    assert np.mean(fixations.t) == approx(2.9365758443368684)
    assert np.mean(fixations.scanpath_history_length) == approx(2.9365758443368684)

    assert np.std(fixations.x) == approx(158.86601835411133)
    assert np.std(fixations.y) == approx(145.67212772412645)
    assert np.std(fixations.t) == approx(2.1599063813289363)
    assert np.std(fixations.scanpath_history_length) == approx(2.1599063813289363)

    assert kurtosis(fixations.x) == approx(0.5791564709307742)
    assert kurtosis(fixations.y) == approx(0.709663215799134)
    assert kurtosis(fixations.t) == approx(-0.7245566668044039)
    assert kurtosis(fixations.scanpath_history_length) == approx(-0.7245566668044039)

    assert skew(fixations.x) == approx(0.09245444798073615)
    assert skew(fixations.y) == approx(-0.008328881229649684)
    assert skew(fixations.t) == approx(0.37950203945703337)
    assert skew(fixations.scanpath_history_length) == approx(0.37950203945703337)

    assert (fixations.n == 0).sum() == 107

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))


@pytest.mark.slow
@pytest.mark.download
def test_OSIE(location):
    real_location = _location(location)

    stimuli, fixations = pysaliency.external_datasets.get_OSIE(location=real_location)
    if location is None:
        assert isinstance(stimuli, pysaliency.Stimuli)
        assert not isinstance(stimuli, pysaliency.FileStimuli)
    else:
        assert isinstance(stimuli, pysaliency.FileStimuli)
        assert location.join('OSIE/stimuli.hdf5').check()
        assert location.join('OSIE/fixations.hdf5').check()

    assert len(stimuli.stimuli) == 700
    assert set(stimuli.sizes) == {(600, 800)}

    assert len(fixations.x) == 98321

    assert np.mean(fixations.x) == approx(401.466024552232)
    assert np.mean(fixations.y) == approx(283.58293548682377)
    assert np.mean(fixations.t) == approx(4.369971826974909)
    assert np.mean(fixations.scanpath_history_length) == approx(4.369971826974909)

    assert np.std(fixations.x) == approx(171.2760014573171)
    assert np.std(fixations.y) == approx(117.70943331958269)
    assert np.std(fixations.t) == approx(2.9882005810119465)
    assert np.std(fixations.scanpath_history_length) == approx(2.9882005810119465)

    assert kurtosis(fixations.x) == approx(-0.8437559648105699)
    assert kurtosis(fixations.y) == approx(-0.6146702485058717)
    assert kurtosis(fixations.t) == approx(-0.7170874454173752)
    assert kurtosis(fixations.scanpath_history_length) == approx(-0.7170874454173752)

    assert skew(fixations.x) == approx(0.029428873077089763)
    assert skew(fixations.y) == approx(0.170931952813165)
    assert skew(fixations.t) == approx(0.31008498461792156)
    assert skew(fixations.scanpath_history_length) == approx(0.31008498461792156)

    assert entropy(fixations.n) == approx(9.445962418853439)
    assert (fixations.n == 0).sum() == 141

    assert len(fixations) == len(pysaliency.datasets.remove_out_of_stimulus_fixations(stimuli, fixations))
