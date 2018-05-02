from __future__ import absolute_import, print_function, division

import pytest
from pytest import approx

import unittest
import numpy as np
from scipy.stats import kurtosis, skew

import pysaliency
import pysaliency.external_datasets


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
    assert np.mean(fixations.lengths) == approx(0.0)

    assert np.std(fixations.x) == approx(132.7479359296397)
    assert np.std(fixations.y) == approx(82.89667109045186)
    assert np.std(fixations.t) == approx(0.0)
    assert np.std(fixations.lengths) == approx(0.0)

    assert kurtosis(fixations.x) == approx(-0.40985986581959066)
    assert kurtosis(fixations.y) == approx(0.2748036777667475)
    assert kurtosis(fixations.t) == approx(-3.0)
    assert kurtosis(fixations.lengths) == approx(-3.0)

    assert skew(fixations.x) == approx(-0.09509166105451604)
    assert skew(fixations.y) == approx(-0.08674038899319877)
    assert skew(fixations.t) == approx(0.0)
    assert skew(fixations.lengths) == approx(0.0)

    assert entropy(fixations.n) == approx(6.8939709237615405)
    assert (fixations.n == 0).sum() == 130


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
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
    assert np.mean(fixations.lengths) == approx(3.3973754691804823)

    assert np.std(fixations.x) == approx(190.0203102093757)
    assert np.std(fixations.y) == approx(159.99210430350126)
    assert np.std(fixations.t) == approx(0.816414737693668)
    assert np.std(fixations.lengths) == approx(2.5433689996843354)

    assert kurtosis(fixations.x) == approx(-0.39272472247196033)
    assert kurtosis(fixations.y) == approx(0.6983793465837596)
    assert kurtosis(fixations.t) == approx(-1.2178525798721818)
    assert kurtosis(fixations.lengths) == approx(-0.45897225172578704)

    assert skew(fixations.x) == approx(0.2204976032609953)
    assert skew(fixations.y) == approx(0.6445191904777621)
    assert skew(fixations.t) == approx(0.08125182887100482)
    assert skew(fixations.lengths) == approx(0.5047182860999948)

    assert entropy(fixations.n) == approx(9.954348058662386)
    assert (fixations.n == 0).sum() == 121


@pytest.mark.slow
@pytest.mark.download
@pytest.mark.skip_octave
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


if __name__ == '__main__':
    unittest.main()
