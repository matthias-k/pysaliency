import pytest

import pysaliency.utils


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     default=False, help="run slow tests")
    parser.addoption("--nomatlab", action="store_true", default=False, help="don't run matlab tests")
    parser.addoption("--notheano", action="store_true", default=False, help="don't run slow theano tests")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    no_matlab = config.getoption("--nomatlab")
    no_theano = config.getoption("--notheano")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_matlab = pytest.mark.skip(reason="skipped because of --nomatlab")
    skip_theano = pytest.mark.skip(reason="skipped because of --notheano")
    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "matlab" in item.keywords and no_matlab:
            item.add_marker(skip_matlab)
        if "theano" in item.keywords and no_theano:
             item.add_marker(skip_theano)


@pytest.fixture(params=["matlab", "octave"])
def matlab(request):
    if request.param == "matlab":
        pysaliency.utils.MatlabOptions.matlab_names = ['matlab', 'matlab.exe']
        pysaliency.utils.MatlabOptions.octave_names = []
    elif request.param == 'octave':
        pysaliency.utils.MatlabOptions.matlab_names = []
        pysaliency.utils.MatlabOptions.octave_names = ['octave', 'octave.exe']

    return request.param


@pytest.fixture()
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
