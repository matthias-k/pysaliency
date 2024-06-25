import os
import sys

import pytest

# make sure that package can be found when running `pytest` instead of `python -m pytest`
sys.path.insert(0, os.getcwd())

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     default=False, help="run slow tests")
    parser.addoption("--run-nonfree", action="store_true",
                     default=False, help="run tests requiring nonpublic data")
    parser.addoption("--nomatlab", action="store_true", default=False, help="don't run matlab tests")
    parser.addoption("--nooctave", action="store_true", default=False, help="don't run octave tests")
    parser.addoption("--notheano", action="store_true", default=False, help="don't run slow theano tests")
    parser.addoption("--nodownload", action="store_true", default=False, help="don't download external data")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    run_nonfree = config.getoption('--run-nonfree')
    no_matlab = config.getoption("--nomatlab")
    no_theano = config.getoption("--notheano")
    no_download = config.getoption("--nodownload")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_nonfree = pytest.mark.skip(reason="need --run-nonfree option to run")
    skip_matlab = pytest.mark.skip(reason="skipped because of --nomatlab")
    skip_theano = pytest.mark.skip(reason="skipped because of --notheano")
    skip_download = pytest.mark.skip(reason="skipped because of --nodownload")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if 'nonfree' in item.keywords and not run_nonfree:
            item.add_marker(skip_nonfree)
        if "matlab" in item.keywords and no_matlab:
            item.add_marker(skip_matlab)
        if "theano" in item.keywords and no_theano:
            item.add_marker(skip_theano)
        if "download" in item.keywords and no_download:
            item.add_marker(skip_download)


@pytest.fixture(params=["matlab", "octave"])
def matlab(request, pytestconfig):
    import pysaliency.utils
    if request.param == "matlab":
        pysaliency.utils.MatlabOptions.matlab_names = ['matlab', 'matlab.exe']
        pysaliency.utils.MatlabOptions.octave_names = []
    elif request.param == 'octave':
        if pytestconfig.getoption("--nooctave"):
            pytest.skip("skipped octave due to command line option")
        elif any([marker.name == 'skip_octave' for marker in request.node.own_markers]):
            pytest.skip("skipped octave due to test marker")
        pysaliency.utils.MatlabOptions.matlab_names = []
        pysaliency.utils.MatlabOptions.octave_names = ['octave', 'octave.exe']

    return request.param

#@pytest.fixture(params=["no_location", "with_location"])
#def location(tmpdir, request):
#    if request.param == 'no_location':
#        return None
#    elif request.param == 'with_location':
#        return tmpdir
#    else:
#        raise ValueError(request.param)


# we don't test in memory external datasets anymore
# we'll probably get rid of them anyway
# TODO: remove this fixture, replace with tmpdir
@pytest.fixture()
def location(tmpdir):
    return tmpdir