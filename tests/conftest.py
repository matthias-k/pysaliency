import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
from tqdm import tqdm

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


@pytest.fixture(autouse=True)
def cache_requests(monkeypatch):
    """This fixture caches requests to avoid downloading the same file multiple times.

    TODO: There should be an option to disable this fixture, e.g. when we want to test downloading.
    """
    original_get = requests.get

    def mock_get(url, *args, **kwargs):
        cache_dir = Path("download_cache")
        cache_dir.mkdir(exist_ok=True)

        cache_filename = (
            url.replace("http://", "")
            .replace("https://", "")
            .replace("/", "_")
            .replace("?", "_")
            .replace("=", "_")
            .replace("&", "_")
            .replace(":", "_")
            .replace(".", "_")
        )
        cache_file = cache_dir / cache_filename

        print("caching", url, "to", cache_file)

        if not cache_file.exists():
            response = original_get(url, *args, **kwargs)
            total_size = int(response.headers.get('content-length', 0))
            with open(cache_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading file') as progress_bar:
                    for chunk in response.iter_content(32*1024):
                        f.write(chunk)
                        progress_bar.update(len(chunk))

        with open(cache_file, 'rb') as f:
            content = f.read()
        mock_response = MagicMock()
        mock_response.iter_content = lambda chunk_size: [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        mock_response.headers = {'content-length': str(len(content))}
        mock_response.status_code = 200
        return mock_response

    monkeypatch.setattr(requests, "get", mock_get)
