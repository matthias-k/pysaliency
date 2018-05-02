import pytest

import pysaliency.utils


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


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
