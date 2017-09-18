from __future__ import absolute_import, print_function, division

import pytest

import unittest

import pysaliency
import pysaliency.external_datasets


@pytest.fixture(params=["matlab", "octave"])
def matlab(request):
    if request.param == "matlab":
        pysaliency.utils.MatlabOptions.matlab_names = ['matlab', 'matlab.exe']
        pysaliency.utils.MatlabOptions.octave_names = []
    elif request.param == 'octave':
        pysaliency.utils.MatlabOptions.matlab_names = []
        pysaliency.utils.MatlabOptions.octave_names = ['octave', 'octave.exe']

    return request.param


@pytest.fixture(autouse=True)
def skip_by_matlab(request, matlab):
    if request.node.get_marker('skip_octave'):
        if matlab == 'octave':
            pytest.skip('skipped octave')


@pytest.fixture(params=["no_location", "with_location"])
def location(tmpdir, request):
    if request.param == 'no_location':
        return None
    elif request.param == 'with_location':
        return tmpdir
    else:
        raise ValueError(request.param)


def _location(location):
    if location is not None:
        return str(location)
    return location


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
        assert location.join('toronto/stimuli.pydat').check()
        assert location.join('toronto/fixations.pydat').check()

    assert len(stimuli.stimuli) == 120
    for n in range(len(stimuli.stimuli)):
        assert stimuli.shapes[n] == (511, 681, 3)
        assert stimuli.sizes[n] == (511, 681)


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
        assert location.join('MIT1003/stimuli.pydat').check()
        assert location.join('MIT1003/fixations.pydat').check()

    assert len(stimuli.stimuli) == 1003
    for n in range(len(stimuli.stimuli)):
        assert max(stimuli.sizes[n]) == 1024

    assert len(fixations.x) == 104171
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
        assert location.join('MIT1003_onesize/stimuli.pydat').check()
        assert location.join('MIT1003_onesize/fixations.pydat').check()

    assert len(stimuli.stimuli) == 463
    for n in range(len(stimuli.stimuli)):
        assert stimuli.sizes[n] == (768, 1024)

    assert len(fixations.x) == 48771
    assert (fixations.n == 0).sum() == 121


if __name__ == '__main__':
    unittest.main()
